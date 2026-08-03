"""Two paired-samples median effect sizes and confidence intervals.

Implements the methods from the legacy
``stats/Calculator/Medians/Two_Paired_Medians.py`` source file with all R /
rpy2 dependencies replaced by pure-Python equivalents.

Classes
-------
TwoPairedMedians
    Effect sizes and CIs for paired (within-subjects) data.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import binom, iqr, norm, t, wilcoxon
from scipy.stats.mstats import hdmedian  # type: ignore[import]
from statsmodels.stats.descriptivestats import sign_test  # type: ignore[import]

from esek.calculators.medians._optional_deps import (
    biweight_midvariance,
    independent_samples_bootstrap,
)
from esek.utils.nonparametric_ci import (
    hodges_lehmann_ci,
    sign_test_ci,
    wilcoxon_location_ci,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass(frozen=True)
class TwoPairedMediansResult:
    """Result container for two paired-samples median analysis."""

    sample_size: int
    group1_stats: dict[str, float]
    group2_stats: dict[str, float]
    difference_stats: dict[str, float]
    effect_sizes: dict[str, float]
    inferential: dict[str, Any]
    confidence_intervals: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def _pairwise_differences(x: np.ndarray) -> list[float]:
    return [float(b - a) for a, b in itertools.combinations(x, 2)]


def _qn_dispersion(x: np.ndarray) -> float:
    diffs = np.abs(_pairwise_differences(x))
    return 2.2219 * float(np.quantile(diffs, 0.25)) if len(diffs) > 0 else float("nan")


def _group_descriptives(x: np.ndarray) -> dict[str, float]:
    """Compute descriptive statistics and dispersion for one group."""
    n = len(x)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    mean_ad = float(np.mean(np.abs(x - med)))
    pw_avgs = [(x[i] + x[j]) / 2.0 for i in range(n) for j in range(i, n)]
    return {
        "n": float(n),
        "median": med,
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "iqr": float(iqr(x)),
        "mad": mad,
        "mad_corrected": mad * 1.4826,
        "mean_ad": mean_ad,
        "mean_ad_corrected": mean_ad * 1.2533,
        "qn": _qn_dispersion(x),
        "range": float(np.max(x) - np.min(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "pseudo_median": float(np.median(pw_avgs)),
        "harrell_davis": float(hdmedian(x)),
    }


def _median_effect_sizes_paired(
    diff: np.ndarray, population_difference: float
) -> dict[str, float]:
    """Effect sizes for the difference vector."""
    med = float(np.median(diff))
    sd = float(np.std(diff, ddof=1))
    actual_diff = med - population_difference

    mad = float(np.median(np.abs(diff - med)))
    iqr_val = float(iqr(diff))
    bw = float(biweight_midvariance(diff) ** 0.5)
    qn = _qn_dispersion(diff)

    return {
        "delta_iqr": actual_diff / iqr_val if iqr_val != 0 else float("nan"),
        "delta_mad": actual_diff / mad if mad != 0 else float("nan"),
        "delta_mad_corrected": actual_diff / (mad * 1.4826) if mad != 0 else float("nan"),
        "delta_bw": actual_diff / bw if bw != 0 else float("nan"),
        "delta_s": actual_diff / sd if sd != 0 else float("nan"),
        "delta_qn": actual_diff / qn if qn != 0 else float("nan"),
        "median_shift": float(
            np.mean(diff - med + population_difference <= med)
        ),
    }


class TwoPairedMedians:
    """Effect sizes and inference for a two-sample paired median test.

    All R/rpy2 dependencies have been replaced by pure-Python equivalents:
    - ``DescTools::SignTest``   → :func:`sign_test_ci`
    - ``DescTools::MedianCI``  → :func:`hodges_lehmann_ci`
    - ``rigr::wilcoxon`` CIs   → :func:`wilcoxon_location_ci`
    """

    @staticmethod
    def from_data(
        group1: np.ndarray,
        group2: np.ndarray,
        population_difference: float = 0.0,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> TwoPairedMediansResult:
        """Compute all effect sizes and CIs for two paired samples.

        Parameters
        ----------
        group1, group2:
            Equal-length 1-D arrays (paired observations).
        population_difference:
            Hypothesised population median of (group1 - group2).
        confidence_level:
            Desired confidence level, e.g. 0.95.
        n_bootstrap:
            Number of bootstrap replications.

        Returns
        -------
        TwoPairedMediansResult
        """
        g1 = np.asarray(group1, dtype=float)
        g2 = np.asarray(group2, dtype=float)
        if len(g1) != len(g2):
            raise ValueError("group1 and group2 must have the same length.")
        n = len(g1)
        if n < 3:
            raise ValueError("At least 3 pairs are required.")
        if not (0 < confidence_level < 1):
            raise ValueError("confidence_level must be in (0, 1).")

        # ── 1. Descriptive statistics ──────────────────────────────────────
        g1_stats = _group_descriptives(g1)
        g2_stats = _group_descriptives(g2)
        diff = g1 - g2
        diff_stats = _group_descriptives(diff)

        # ── 2. Effect sizes ────────────────────────────────────────────────
        effect_sizes = _median_effect_sizes_paired(diff, population_difference)

        # ── 3. Inferential statistics ──────────────────────────────────────
        diff_median = float(np.median(diff))
        mad_diff = float(np.median(np.abs(diff - diff_median)))
        se_mad = (mad_diff * 1.4826) / np.sqrt(n)
        t_mad = (diff_median - population_difference) / se_mad if se_mad != 0 else float("nan")
        p_mad = float(t.sf(abs(t_mad), n - 1) * 2)

        # Sign test
        s_stat, p_sign, _, _ = sign_test_ci(diff, mu0=population_difference, confidence_level=confidence_level)
        _, p_binom = sign_test(samp=g1, mu0=population_difference)

        # Wilcoxon tests
        try:
            stat_exact, p_wx_exact = wilcoxon(diff - population_difference, method="exact")
        except Exception:
            stat_exact, p_wx_exact = wilcoxon(diff - population_difference, method="approx")
        _, p_wx_approx = wilcoxon(diff - population_difference, method="approx", correction=False)
        _, p_wx_corr = wilcoxon(diff - population_difference, method="approx", correction=True)

        # Price & Bonett CI for difference between medians
        zcrit = float(norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))
        m1, m2 = float(np.median(g1)), float(np.median(g2))

        def _price_bonett_variance(arr: np.ndarray) -> float:
            arr_s = np.sort(arr)
            ns = len(arr_s)
            c = max(1, int(np.round((ns + 1) / 2 - ns ** 0.5)))
            hi = arr_s[min(ns - c, ns - 1)]
            lo = arr_s[max(c - 1, 0)]
            z = float(norm.ppf(1.0 - binom.cdf(c - 1, ns, 0.5)))
            return ((hi - lo) / (2.0 * z)) ** 2

        var1 = _price_bonett_variance(g1)
        var2 = _price_bonett_variance(g2)
        median_corr = (np.sum((g1 < m1) & (g2 < m2)) + 0.25) / (n + 1)
        cov12 = median_corr * np.sqrt(var1 * var2)
        se_diff = np.sqrt(var1 + var2 - 2 * cov12)
        ci_price_lo = (m1 - m2) - zcrit * se_diff
        ci_price_hi = (m1 - m2) + zcrit * se_diff

        inferential = {
            "sign_statistic": s_stat,
            "p_sign_exact": float(p_sign),
            "p_binomial": float(p_binom),
            "wilcoxon_statistic_exact": float(stat_exact),
            "p_wilcoxon_exact": float(p_wx_exact),
            "p_wilcoxon_approx": float(p_wx_approx),
            "p_wilcoxon_approx_corrected": float(p_wx_corr),
            "t_mad_based": float(t_mad),
            "p_mad_based": float(p_mad),
            "se_mad": float(se_mad),
        }

        # ── 4. Confidence intervals ────────────────────────────────────────
        ci: dict[str, Any] = {}

        # Bootstrap CIs for the median of differences
        boot = independent_samples_bootstrap(diff)
        ci_basic = boot.conf_int(lambda x: np.median(x), n_bootstrap, method="basic", size=confidence_level)
        ci_pct = boot.conf_int(lambda x: np.median(x), n_bootstrap, method="percentile", size=confidence_level)
        ci_bc = boot.conf_int(lambda x: np.median(x), n_bootstrap, method="bc", size=confidence_level)
        ci_norm = boot.conf_int(lambda x: np.median(x), n_bootstrap, method="norm", size=confidence_level)

        ci["diff_median_bootstrap_basic"] = (float(ci_basic[0, 0]), float(ci_basic[1, 0]))
        ci["diff_median_bootstrap_percentile"] = (float(ci_pct[0, 0]), float(ci_pct[1, 0]))
        ci["diff_median_bootstrap_bc"] = (float(ci_bc[0, 0]), float(ci_bc[1, 0]))
        ci["diff_median_bootstrap_normal"] = (float(ci_norm[0, 0]), float(ci_norm[1, 0]))

        # Hodges-Lehmann CI for difference
        hl_est, hl_lo, hl_hi = hodges_lehmann_ci(diff, confidence_level=confidence_level)
        ci["diff_median_hodges_lehmann"] = (hl_lo, hl_hi)

        # Sign test CI
        _, _, st_lo, st_hi = sign_test_ci(diff, mu0=population_difference, confidence_level=confidence_level)
        ci["diff_sign_test"] = (st_lo, st_hi)

        # Wilcoxon CIs
        _, _, wx_lo_ex, wx_hi_ex = wilcoxon_location_ci(diff, mu0=population_difference, confidence_level=confidence_level, method="exact")
        _, _, wx_lo_ap, wx_hi_ap = wilcoxon_location_ci(diff, mu0=population_difference, confidence_level=confidence_level, method="approx", correction=True)
        _, _, wx_lo_nc, wx_hi_nc = wilcoxon_location_ci(diff, mu0=population_difference, confidence_level=confidence_level, method="approx", correction=False)
        ci["diff_wilcoxon_exact"] = (wx_lo_ex, wx_hi_ex)
        ci["diff_wilcoxon_approx"] = (wx_lo_ap, wx_hi_ap)
        ci["diff_wilcoxon_approx_corrected"] = (wx_lo_nc, wx_hi_nc)

        # MAD-based CI
        t_crit = float(t.ppf(confidence_level + (1 - confidence_level) / 2, n - 1))
        ci["diff_mad_based"] = (diff_median - se_mad * t_crit, diff_median + se_mad * t_crit)

        # Price-Bonett CI for difference between medians
        ci["diff_price_bonett"] = (ci_price_lo, ci_price_hi)

        # Bootstrap CIs for effect sizes
        def _boot_es(dat: np.ndarray) -> np.ndarray:
            es = _median_effect_sizes_paired(dat, population_difference)
            return np.array(list(es.values()))

        boot_es = independent_samples_bootstrap(diff)
        try:
            ci_es = boot_es.conf_int(_boot_es, n_bootstrap, method="percentile", size=confidence_level)
            for i, key in enumerate(effect_sizes.keys()):
                ci[f"{key}_bootstrap"] = (float(ci_es[0, i]), float(ci_es[1, i]))
        except Exception:
            for key in effect_sizes:
                ci[f"{key}_bootstrap"] = (float("nan"), float("nan"))

        return TwoPairedMediansResult(
            sample_size=n,
            group1_stats=g1_stats,
            group2_stats=g2_stats,
            difference_stats=diff_stats,
            effect_sizes=effect_sizes,
            inferential=inferential,
            confidence_intervals=ci,
            metadata={
                "population_difference": population_difference,
                "confidence_level": confidence_level,
                "n_bootstrap": n_bootstrap,
            },
        )


__all__ = ["TwoPairedMedians", "TwoPairedMediansResult"]
