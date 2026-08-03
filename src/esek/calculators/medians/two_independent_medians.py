"""Two independent-samples median effect sizes and confidence intervals.

Implements the methods from the legacy
``stats/Calculator/Medians/TwoIndependentMedians/Two_Ind_Medians.py`` source
file.  Although the original source imported DescTools and rigr, they were
never actually called — all computations in that file were pure Python /
SciPy.  This module preserves those computations in a clean OOP interface.

Classes
-------
TwoIndependentMedians
    Effect sizes, CI for the difference between medians, and Mood's median test
    for two independent groups.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import binom, iqr, median_abs_deviation, norm
from scipy.stats.mstats import hdmedian  # type: ignore[import]

from esek.utils.nonparametric_ci import hodges_lehmann_ci

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass(frozen=True)
class TwoIndependentMediansResult:
    """Result container for two independent-samples median analysis."""

    group1_stats: dict[str, float]
    group2_stats: dict[str, float]
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
    n = len(x)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    mean_ad = float(np.mean(np.abs(x - med)))
    pw = [(x[i] + x[j]) / 2.0 for i in range(n) for j in range(i, n)]
    return {
        "n": float(n),
        "median": med,
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "iqr": float(iqr(x)),
        "mad": mad,
        "mad_corrected": mad * 1.4826,
        "mean_ad": mean_ad,
        "qn": _qn_dispersion(x),
        "range": float(np.max(x) - np.min(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "pseudo_median": float(np.median(pw)),
        "harrell_davis": float(hdmedian(x)),
    }


def _price_bonett_ci(
    arr1: np.ndarray,
    arr2: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    """Price-Bonett CI for the difference between two independent medians.

    References
    ----------
    - Price, R. M., & Bonett, D. G. (2002). Distribution-free confidence
      intervals for difference and ratio of medians. *Journal of Statistical
      Computation and Simulation*, 72(2), 119-124.
    """
    zcrit = float(norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))

    def _variance(arr: np.ndarray) -> float:
        ns = len(arr)
        arr_s = np.sort(arr)
        c = max(1, int(np.round((ns + 1) / 2 - ns ** 0.5)))
        hi = arr_s[min(ns - c, ns - 1)]
        lo = arr_s[max(c - 1, 0)]
        z = float(norm.ppf(1.0 - binom.cdf(c - 1, ns, 0.5)))
        return ((hi - lo) / (2.0 * z)) ** 2

    v1 = _variance(arr1)
    v2 = _variance(arr2)
    se = math.sqrt(v1 + v2)
    diff = float(np.median(arr1) - np.median(arr2))
    return diff - zcrit * se, diff + zcrit * se


def _price_bonett_ratio_ci(
    arr1: np.ndarray,
    arr2: np.ndarray,
    confidence_level: float,
) -> tuple[float, float, float]:
    """Price-Bonett CI for the ratio of two independent medians (log-scale).

    Returns
    -------
    tuple[float, float, float]
        (ratio, ci_lower, ci_upper)
    """
    zcrit = float(norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))

    def _log_variance(arr: np.ndarray) -> float:
        ns = len(arr)
        arr_s = np.sort(arr)
        c = max(1, int(np.round((ns + 1) / 2 - ns ** 0.5)))
        hi = arr_s[min(ns - c, ns - 1)]
        lo = arr_s[max(c - 1, 0)]
        log_hi = math.log(max(abs(hi), 1e-300))
        log_lo = math.log(max(abs(lo), 1e-300))
        z = float(norm.ppf(1.0 - binom.cdf(c - 1, ns, 0.5)))
        return ((log_hi - log_lo) / (2.0 * z)) ** 2

    v1 = _log_variance(arr1)
    v2 = _log_variance(arr2)
    log_se = math.sqrt(v1 + v2)
    ratio = float(np.median(arr1) / np.median(arr2))
    return (
        ratio,
        ratio * math.exp(-zcrit * log_se),
        ratio * math.exp(zcrit * log_se),
    )


class TwoIndependentMedians:
    """Effect sizes and inference for two independent samples.

    .. note::
        The original source imported ``DescTools`` and ``rigr`` but never
        called them.  All computations here are pure Python / SciPy.
    """

    @staticmethod
    def from_data(
        group1: np.ndarray,
        group2: np.ndarray,
        population_difference: float = 0.0,
        confidence_level: float = 0.95,
    ) -> TwoIndependentMediansResult:
        """Compute effect sizes and CIs for two independent samples.

        Parameters
        ----------
        group1, group2:
            1-D arrays (may differ in length).
        population_difference:
            Hypothesised difference Median₁ - Median₂ under H₀.
        confidence_level:
            Desired confidence level, e.g. 0.95.

        Returns
        -------
        TwoIndependentMediansResult
        """
        g1 = np.asarray(group1, dtype=float)
        g2 = np.asarray(group2, dtype=float)
        if len(g1) < 2 or len(g2) < 2:
            raise ValueError("Each group must have at least 2 observations.")
        if not (0 < confidence_level < 1):
            raise ValueError("confidence_level must be in (0, 1).")

        n1, n2 = len(g1), len(g2)
        g1_stats = _group_descriptives(g1)
        g2_stats = _group_descriptives(g2)

        m1, m2 = float(np.median(g1)), float(np.median(g2))
        sd1 = float(np.std(g1, ddof=1))
        sd2 = float(np.std(g2, ddof=1))
        mad1 = float(median_abs_deviation(g1))
        mad2 = float(median_abs_deviation(g2))
        mad1_c = mad1 * 1.4826
        mad2_c = mad2 * 1.4826

        diff = (m1 - m2) - population_difference

        # ── Effect sizes ───────────────────────────────────────────────────
        # 1. Thompson (2007) — pooled SD
        pooled_sd = math.sqrt((sd1 ** 2 + sd2 ** 2) / 2.0)
        d_mdns = diff / pooled_sd if pooled_sd != 0 else float("nan")

        # 2. Pooled MAD (Ricca & Blaine)
        mad_pooled = (
            ((n1 - 1) * mad1 + (n2 - 1) * mad2) / (n1 + n2 - 2)
        )
        mad_pooled_c = (
            ((n1 - 1) * mad1_c + (n2 - 1) * mad2_c) / (n1 + n2 - 2)
        )
        d_mad_pooled = diff / mad_pooled if mad_pooled != 0 else float("nan")
        d_mad_pooled_c = diff / mad_pooled_c if mad_pooled_c != 0 else float("nan")

        # 3. Quantile shift (Wilcox)
        all_pairwise = np.subtract.outer(g1, g2).ravel()
        med_comparisons = float(np.median(all_pairwise))
        qs = float(np.mean(all_pairwise - med_comparisons <= med_comparisons))

        effect_sizes = {
            "d_mdns": d_mdns,
            "d_mad_pooled": d_mad_pooled,
            "d_mad_pooled_corrected": d_mad_pooled_c,
            "quantile_symmetric": qs,
        }

        # ── Inferential statistics ─────────────────────────────────────────
        # Mood's median test (via contingency table)
        grand_median = float(np.median(np.concatenate([g1, g2])))
        a = np.sum(g1 > grand_median)
        b = np.sum(g1 <= grand_median)
        c = np.sum(g2 > grand_median)
        d = np.sum(g2 <= grand_median)
        total = n1 + n2
        expected_a = (a + c) * n1 / total
        expected_b = (b + d) * n1 / total
        expected_c = (a + c) * n2 / total
        expected_d = (b + d) * n2 / total

        chi2 = sum(
            (obs - exp) ** 2 / exp if exp > 0 else 0.0
            for obs, exp in [
                (a, expected_a), (b, expected_b), (c, expected_c), (d, expected_d)
            ]
        )
        from scipy.stats import chi2 as chi2_dist
        p_moods = float(chi2_dist.sf(chi2, df=1))

        # Hodges-Lehmann estimator for the difference (Walsh averages of all
        # pairwise differences g1_i - g2_j)
        hl_diff_est, hl_diff_lo, hl_diff_hi = hodges_lehmann_ci(
            all_pairwise, confidence_level=confidence_level
        )

        inferential = {
            "chi2_moods_median": float(chi2),
            "p_moods_median": p_moods,
            "grand_median": grand_median,
            "hodges_lehmann_diff": hl_diff_est,
        }

        # ── Confidence intervals ───────────────────────────────────────────
        ci: dict[str, Any] = {}

        # Price-Bonett CI for difference
        pb_lo, pb_hi = _price_bonett_ci(g1, g2, confidence_level)
        ci["diff_price_bonett"] = (pb_lo, pb_hi)

        # Price-Bonett CI for ratio
        ratio, ratio_lo, ratio_hi = _price_bonett_ratio_ci(g1, g2, confidence_level)
        ci["ratio_price_bonett"] = (ratio_lo, ratio_hi)
        ci["ratio"] = ratio

        # Hodges-Lehmann CI for difference
        ci["diff_hodges_lehmann"] = (hl_diff_lo, hl_diff_hi)

        return TwoIndependentMediansResult(
            group1_stats=g1_stats,
            group2_stats=g2_stats,
            effect_sizes=effect_sizes,
            inferential=inferential,
            confidence_intervals=ci,
            metadata={
                "population_difference": population_difference,
                "confidence_level": confidence_level,
            },
        )


__all__ = ["TwoIndependentMedians", "TwoIndependentMediansResult"]
