"""One-sample median effect sizes and confidence intervals.

Implements the methods from the legacy
``stats/Calculator/Medians/One_Sample_Median.py`` source file with all R /
rpy2 dependencies (DescTools, rigr) replaced by pure-Python equivalents.

The R replacements are:
- ``DescTools::SignTest``   → :func:`esek.utils.nonparametric_ci.sign_test_ci`
- ``DescTools::MedianCI``  → :func:`esek.utils.nonparametric_ci.hodges_lehmann_ci`
- ``rigr::wilcoxon`` CIs   → :func:`esek.utils.nonparametric_ci.wilcoxon_location_ci`

Classes
-------
OneSampleMedian
    Static factory methods for computing effect sizes, descriptive stats, and
    confidence intervals for a single sample compared to a population median.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy.stats import biweight_midvariance  # type: ignore[import]
from arch.bootstrap import IndependentSamplesBootstrap  # type: ignore[import]
from scipy.stats import iqr, norm, t, wilcoxon
from scipy.stats.mstats import hdmedian  # type: ignore[import]
from statsmodels.stats.descriptivestats import sign_test  # type: ignore[import]

from esek.utils.nonparametric_ci import (
    hodges_lehmann_ci,
    sign_test_ci,
    wilcoxon_location_ci,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass(frozen=True)
class OneSampleMedianResult:
    """Result container for one-sample median analysis.

    Attributes
    ----------
    sample_size : int
    median : float
    mean : float
    sd : float
    pseudo_median : float
        Hodges-Lehmann location estimator.
    harrell_davis_estimator : float
    dispersion : dict[str, float]
        Spread measures: Range, IQR, MAD, corrected MAD, mean AD, Qn.
    effect_sizes : dict[str, float]
        Six standardised effect sizes.
    inferential : dict[str, Any]
        Test statistics and p-values.
    confidence_intervals : dict[str, Any]
        CI estimates for the median and each effect size.
    metadata : dict[str, Any]
    """

    sample_size: int
    median: float
    mean: float
    sd: float
    pseudo_median: float
    harrell_davis_estimator: float
    dispersion: dict[str, float]
    effect_sizes: dict[str, float]
    inferential: dict[str, Any]
    confidence_intervals: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def _pairwise_differences(x: np.ndarray) -> list[float]:
    """All pairwise differences (x_j - x_i) for j > i."""
    return [float(b - a) for a, b in itertools.combinations(x, 2)]


def _qn_dispersion(x: np.ndarray) -> float:
    """Qn dispersion estimator (Rousseeuw & Croux, 1993).

    Qn = 2.2219 * 25th-percentile of |x_i - x_j| for i < j.
    """
    diffs = np.abs(_pairwise_differences(x))
    return 2.2219 * float(np.quantile(diffs, 0.25)) if len(diffs) > 0 else float("nan")


def _median_effect_sizes(
    data: np.ndarray,
    population_median: float,
) -> dict[str, float]:
    """Seven effect sizes for a one-sample median test.

    Parameters
    ----------
    data:
        1-D array of observations.
    population_median:
        Null hypothesis value for the median.

    Returns
    -------
    dict mapping short name → effect size value.

    References
    ----------
    - Laird, N., & Mosteller, F. (1990). Some statistical methods.
    - Grissom, R. J., & Kim, J. J. (2012). Effect Sizes for Research.
    - Thompson, B. (2007). Interpreting effect sizes.
    - Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the median
      absolute deviation.
    """
    sample_median = float(np.median(data))
    sd = float(np.std(data, ddof=1))
    diff = sample_median - population_median

    iqr_val = float(np.quantile(data, 0.75) - np.quantile(data, 0.25))
    mad = float(np.median(np.abs(data - sample_median)))
    bw = float(biweight_midvariance(data) ** 0.5)
    qn = _qn_dispersion(data)

    return {
        "delta_iqr": diff / iqr_val if iqr_val != 0 else float("nan"),
        "delta_mad": diff / mad if mad != 0 else float("nan"),
        "delta_mad_corrected": diff / (mad * 1.4826) if mad != 0 else float("nan"),
        "delta_bw": diff / bw if bw != 0 else float("nan"),
        "delta_s": diff / sd if sd != 0 else float("nan"),
        "delta_qn": diff / qn if qn != 0 else float("nan"),
        "median_shift": float(np.mean(data - sample_median + population_median <= sample_median)),
    }


class OneSampleMedian:
    """Effect sizes and inference for a one-sample median test.

    All R/rpy2 dependencies have been replaced by pure-Python equivalents.
    The Hodges-Lehmann and Wilcoxon CI methods previously computed via
    ``DescTools::MedianCI`` and ``rigr::wilcoxon`` are now computed internally
    via Walsh averages and the normal approximation to the Wilcoxon distribution.
    """

    @staticmethod
    def from_data(
        data: np.ndarray,
        population_median: float = 0.0,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> OneSampleMedianResult:
        """Compute all effect sizes and CIs for a single sample.

        Parameters
        ----------
        data:
            1-D array of observations.
        population_median:
            Hypothesised population median (μ₀).
        confidence_level:
            Desired confidence level, e.g. 0.95.
        n_bootstrap:
            Number of bootstrap replications for effect-size CIs.

        Returns
        -------
        OneSampleMedianResult
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        if n < 3:
            raise ValueError("At least 3 observations are required.")
        if not (0 < confidence_level < 1):
            raise ValueError("confidence_level must be in (0, 1).")

        # ── 1. Descriptive statistics ──────────────────────────────────────
        sample_median = float(np.median(data))
        sample_mean = float(np.mean(data))
        sd = float(np.std(data, ddof=1))
        mad = float(np.median(np.abs(data - sample_median)))
        mean_ad = float(np.mean(np.abs(data - sample_median)))
        iqr_val = float(iqr(data))
        qn = _qn_dispersion(data)

        # Hodges-Lehmann pseudomedian
        pairwise_avgs = [
            (data[i] + data[j]) / 2.0
            for i in range(n)
            for j in range(i, n)
        ]
        pseudo_median = float(np.median(pairwise_avgs))

        hd_estimator = float(hdmedian(data))

        dispersion = {
            "range": float(np.max(data) - np.min(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "iqr": iqr_val,
            "mean_absolute_deviation": mean_ad,
            "median_absolute_deviation": mad,
            "mad_corrected": mad * 1.4826,
            "mean_ad_corrected": mean_ad * 1.2533,
            "qn": qn,
        }

        # ── 2. Effect sizes ────────────────────────────────────────────────
        effect_sizes = _median_effect_sizes(data, population_median)

        # ── 3. Inferential statistics ──────────────────────────────────────
        se_mad = (mad * 1.4826) / np.sqrt(n)
        t_mad = (sample_median - population_median) / se_mad if se_mad != 0 else float("nan")
        p_mad = float(t.sf(abs(t_mad), n - 1) * 2)

        # Sign test (exact p-value)
        s_stat, p_sign_exact, _, _ = sign_test_ci(
            data, mu0=population_median, confidence_level=confidence_level
        )
        _, p_binom = sign_test(samp=data, mu0=population_median)

        # Wilcoxon tests
        try:
            stat_exact, p_wilcox_exact = wilcoxon(
                data - population_median, method="exact"
            )
        except Exception:
            stat_exact, p_wilcox_exact = wilcoxon(data - population_median, method="approx")

        _, p_wilcox_approx = wilcoxon(
            data - population_median, method="approx", correction=False
        )
        _, p_wilcox_corrected = wilcoxon(
            data - population_median, method="approx", correction=True
        )

        inferential = {
            "sign_statistic": s_stat,
            "p_sign_exact": float(p_sign_exact),
            "p_binomial": float(p_binom),
            "wilcoxon_statistic_exact": float(stat_exact),
            "p_wilcoxon_exact": float(p_wilcox_exact),
            "p_wilcoxon_approx": float(p_wilcox_approx),
            "p_wilcoxon_approx_corrected": float(p_wilcox_corrected),
            "t_mad_based": float(t_mad),
            "p_mad_based": float(p_mad),
            "se_mad": float(se_mad),
        }

        # ── 4. Confidence intervals ────────────────────────────────────────
        ci = {}

        # Bootstrap CIs for the median
        boot = IndependentSamplesBootstrap(data)
        ci_basic = boot.conf_int(
            lambda x: np.median(x), n_bootstrap, method="basic", size=confidence_level
        )
        ci_pct = boot.conf_int(
            lambda x: np.median(x), n_bootstrap, method="percentile", size=confidence_level
        )
        ci_bc = boot.conf_int(
            lambda x: np.median(x), n_bootstrap, method="bc", size=confidence_level
        )
        ci_norm = boot.conf_int(
            lambda x: np.median(x), n_bootstrap, method="norm", size=confidence_level
        )
        ci["median_bootstrap_basic"] = (float(ci_basic[0, 0]), float(ci_basic[1, 0]))
        ci["median_bootstrap_percentile"] = (float(ci_pct[0, 0]), float(ci_pct[1, 0]))
        ci["median_bootstrap_bc"] = (float(ci_bc[0, 0]), float(ci_bc[1, 0]))
        ci["median_bootstrap_normal"] = (float(ci_norm[0, 0]), float(ci_norm[1, 0]))

        # Hodges-Lehmann (DescTools.MedianCI equivalent)
        hl_est, hl_lo, hl_hi = hodges_lehmann_ci(data, confidence_level=confidence_level)
        ci["median_hodges_lehmann"] = (hl_lo, hl_hi)

        # Sign test exact CI
        _, _, sign_lo, sign_hi = sign_test_ci(
            data, mu0=population_median, confidence_level=confidence_level
        )
        ci["median_sign_test_exact"] = (sign_lo, sign_hi)

        # Wilcoxon-based CIs (replaces rigr::wilcoxon CI)
        _, _, wx_lo_exact, wx_hi_exact = wilcoxon_location_ci(
            data, mu0=population_median, confidence_level=confidence_level, method="exact"
        )
        _, _, wx_lo_approx, wx_hi_approx = wilcoxon_location_ci(
            data, mu0=population_median, confidence_level=confidence_level,
            method="approx", correction=True
        )
        _, _, wx_lo_nc, wx_hi_nc = wilcoxon_location_ci(
            data, mu0=population_median, confidence_level=confidence_level,
            method="approx", correction=False
        )
        ci["median_wilcoxon_exact"] = (wx_lo_exact, wx_hi_exact)
        ci["median_wilcoxon_approx"] = (wx_lo_approx, wx_hi_approx)
        ci["median_wilcoxon_approx_corrected"] = (wx_lo_nc, wx_hi_nc)

        # MAD-based CI for median
        t_crit = float(t.ppf(confidence_level + (1 - confidence_level) / 2, n - 1))
        ci["median_mad_based"] = (
            sample_median - se_mad * t_crit,
            sample_median + se_mad * t_crit,
        )

        # Bootstrap CIs for each effect size
        def _boot_es(dat: np.ndarray) -> np.ndarray:
            es = _median_effect_sizes(dat, population_median)
            return np.array(list(es.values()))

        boot_es = IndependentSamplesBootstrap(data)
        try:
            ci_es = boot_es.conf_int(
                _boot_es, n_bootstrap, method="percentile", size=confidence_level
            )
            for i, key in enumerate(effect_sizes.keys()):
                ci[f"{key}_bootstrap"] = (float(ci_es[0, i]), float(ci_es[1, i]))
        except Exception:
            for key in effect_sizes:
                ci[f"{key}_bootstrap"] = (float("nan"), float("nan"))

        return OneSampleMedianResult(
            sample_size=n,
            median=sample_median,
            mean=sample_mean,
            sd=sd,
            pseudo_median=pseudo_median,
            harrell_davis_estimator=hd_estimator,
            dispersion=dispersion,
            effect_sizes=effect_sizes,
            inferential=inferential,
            confidence_intervals=ci,
            metadata={
                "population_median": population_median,
                "confidence_level": confidence_level,
                "n_bootstrap": n_bootstrap,
            },
        )


__all__ = ["OneSampleMedian", "OneSampleMedianResult"]
