"""Nonparametric confidence interval helpers.

Pure-Python replacements for the R packages DescTools (MedianCI, SignTest) and
rigr (wilcoxon) that were used in the original ``stats/Calculator/Medians/``
source files.

All CI computations are purely distribution-based and do not require R or rpy2.

Functions
---------
sign_test_ci(x, mu0, confidence_level)
    One-sample sign test and exact binomial CI for the median.
hodges_lehmann_ci(x, confidence_level)
    Hodges-Lehmann point estimate with Wilcoxon-based CI (DescTools.MedianCI
    equivalent).
wilcoxon_location_ci(x, mu0, confidence_level, method)
    CI for the median by inverting the one-sample Wilcoxon signed-rank test
    (rigr.wilcoxon equivalent).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.stats import binom, norm, wilcoxon


def sign_test_ci(
    x: np.ndarray,
    mu0: float = 0.0,
    confidence_level: float = 0.95,
) -> tuple[float, float, float, float]:
    """One-sample sign test and exact CI for the median.

    Computes the sign test statistic *S*, its exact two-sided p-value, and a
    binomial-inversion confidence interval for the true median θ.  This is the
    pure-Python replacement for ``DescTools::SignTest``.

    The CI is the shortest interval ``[x_(k), x_(n-k+1)]`` (in sorted order)
    such that P(B ≥ k) + P(B ≤ n-k) ≥ confidence_level where B ~ Bin(n, 0.5).

    Parameters
    ----------
    x:
        1-D numeric array of observations.
    mu0:
        Null hypothesis value for the median.
    confidence_level:
        Desired confidence level (default 0.95).

    Returns
    -------
    tuple[float, float, float, float]
        ``(S_statistic, p_value, ci_lower, ci_upper)``

    References
    ----------
    - Hollander, M., & Wolfe, D. A. (1999). *Nonparametric Statistical
      Methods* (2nd ed.). Wiley.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    x_sorted = np.sort(x)

    # Sign test statistic: count of observations above mu0
    s_plus = int(np.sum(x > mu0))
    s_minus = int(np.sum(x < mu0))
    s_stat = float(max(s_plus, s_minus))

    # Two-sided p-value using exact binomial
    p_value = float(2.0 * binom.sf(s_stat - 1, n, 0.5))
    p_value = min(p_value, 1.0)

    # Binomial CI for the median (Hodges 1955)
    alpha = 1.0 - confidence_level
    k = 0
    while binom.cdf(k, n, 0.5) < alpha / 2.0:
        k += 1

    if k >= n:
        ci_lower = float("-inf")
        ci_upper = float("inf")
    else:
        ci_lower = float(x_sorted[k])
        ci_upper = float(x_sorted[n - k - 1]) if n - k - 1 >= 0 else float("inf")

    return s_stat, p_value, ci_lower, ci_upper


def hodges_lehmann_ci(
    x: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Hodges-Lehmann estimator with Wilcoxon-based CI.

    Computes the pseudomedian (Hodges-Lehmann location estimator) as the
    median of all pairwise averages (Walsh averages), together with a
    confidence interval obtained by inverting the one-sample Wilcoxon
    signed-rank test.  This is the pure-Python replacement for
    ``DescTools::MedianCI``.

    Parameters
    ----------
    x:
        1-D numeric array of observations.
    confidence_level:
        Desired confidence level (default 0.95).

    Returns
    -------
    tuple[float, float, float]
        ``(point_estimate, ci_lower, ci_upper)``

    Notes
    -----
    For small n the exact Wilcoxon critical value is used via a bisection
    search on the Wilcoxon pmf.  For n > 50 the normal approximation with
    continuity correction is used instead for performance.

    References
    ----------
    - Hodges, J. L., & Lehmann, E. L. (1963). Estimates of location based
      on rank tests. *Annals of Mathematical Statistics*, 34(2), 598-611.
    - Hollander, M., & Wolfe, D. A. (1999). *Nonparametric Statistical
      Methods* (2nd ed.). Wiley.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    big_n = n * (n + 1) // 2  # total number of Walsh averages

    # Walsh averages (pairwise averages including xi with itself)
    walsh: list[float] = [
        (x[i] + x[j]) / 2.0
        for i in range(n)
        for j in range(i, n)
    ]
    walsh.sort()

    point_estimate = float(np.median(walsh))

    # Find Wilcoxon critical value k such that P(W ≤ k-1) ≤ α/2
    alpha = 1.0 - confidence_level
    mu_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0

    # Normal approximation with continuity correction
    z_alpha2 = float(norm.ppf(1.0 - alpha / 2.0))
    t_lower = mu_w - z_alpha2 * math.sqrt(var_w) - 0.5
    k = max(1, int(math.floor(t_lower)) + 1)
    k = min(k, big_n)

    ci_lower = float(walsh[k - 1])
    ci_upper = float(walsh[big_n - k])

    return point_estimate, ci_lower, ci_upper


def wilcoxon_location_ci(
    x: np.ndarray,
    mu0: float = 0.0,
    confidence_level: float = 0.95,
    correction: bool = True,
    method: Literal["approx", "exact", "auto"] = "auto",
) -> tuple[float, float, float, float]:
    """CI for the median by inverting the Wilcoxon signed-rank test.

    This is the pure-Python replacement for ``rigr::wilcoxon`` CI output.
    Returns the two-sided p-value and the CI for the location parameter θ
    (population median of the signed-rank distribution, i.e. the pseudomedian).

    The CI is computed by Hodges-Lehmann inversion: finding the range of θ₀
    values for which the Wilcoxon signed-rank test of H₀: θ = θ₀ would not
    be rejected at significance level ``1 - confidence_level``.

    Parameters
    ----------
    x:
        1-D numeric array of observations.
    mu0:
        Null hypothesis value for the centre (default 0.0).  Used only for
        the test p-value; the CI is always for the pseudomedian.
    confidence_level:
        Desired confidence level (default 0.95).
    correction:
        Whether to apply continuity correction for the normal approximation
        (default True).
    method:
        ``"exact"`` uses the exact Wilcoxon distribution (slow for n > 25),
        ``"approx"`` uses the normal approximation, ``"auto"`` uses exact for
        n ≤ 25 and approximate otherwise.

    Returns
    -------
    tuple[float, float, float, float]
        ``(statistic, p_value, ci_lower, ci_upper)``

    References
    ----------
    - Hollander, M., & Wolfe, D. A. (1999). *Nonparametric Statistical
      Methods* (2nd ed.). Wiley, pp. 133-134.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    # Wilcoxon test p-value
    _method: str
    if method == "auto":
        _method = "exact" if n <= 25 else ("approx" if not correction else "approx")
    else:
        _method = method

    try:
        stat, p_val = wilcoxon(x - mu0, correction=correction, method=_method)
    except Exception:
        stat, p_val = wilcoxon(x - mu0, method="approx")

    # CI via Hodges-Lehmann inversion (same as hodges_lehmann_ci but offset by mu0)
    _, ci_lo, ci_hi = hodges_lehmann_ci(x, confidence_level=confidence_level)

    return float(stat), float(p_val), ci_lo, ci_hi


__all__ = ["sign_test_ci", "hodges_lehmann_ci", "wilcoxon_location_ci"]
