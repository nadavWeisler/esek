"""Mathematical helper functions for ESEK.

Pure numerical helpers used internally by calculators and CI methods.
No side effects, no formatting, no printing.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def bias_correction_factor(df: float) -> float:
    """Compute the bias-correction factor *J* for Hedges' g.

    Uses the log-gamma form for numerical stability:

        J(df) = exp(lgamma(df/2) − log(sqrt(df/2)) − lgamma((df−1)/2))

    Parameters
    ----------
    df:
        Degrees of freedom (must be > 1).

    Returns
    -------
    float
        Bias-correction factor J, which is slightly less than 1.
    """
    if df <= 1:
        raise ValueError(f"df must be > 1 for bias correction, got {df}.")
    return math.exp(
        math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2)
    )


def pooled_sd(sd1: float, sd2: float, n1: int, n2: int) -> float:
    """Compute the pooled standard deviation for two independent groups.

    Uses the exact formula:

        s_p = sqrt(((n₁−1)·s₁² + (n₂−1)·s₂²) / (n₁+n₂−2))

    Parameters
    ----------
    sd1, sd2:
        Standard deviations of groups 1 and 2.
    n1, n2:
        Sample sizes of groups 1 and 2.

    Returns
    -------
    float
    """
    df = n1 + n2 - 2
    if df <= 0:
        raise ValueError(
            f"Combined degrees of freedom must be > 0 (n1={n1}, n2={n2})."
        )
    return math.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / df)


def harmonic_mean_n(n1: int, n2: int) -> float:
    """Compute the harmonic mean of two sample sizes.

        h = 2 / (1/n₁ + 1/n₂)

    Parameters
    ----------
    n1, n2:
        Sample sizes.

    Returns
    -------
    float
    """
    return 2.0 / (1.0 / n1 + 1.0 / n2)


def winsorized_variance(
    x: list[float] | NDArray[np.floating],
    trimming_level: float = 0.2,
) -> float:
    """Compute the Winsorized variance of a sample.

    Parameters
    ----------
    x:
        The data array.
    trimming_level:
        Fraction to trim from each tail (default 0.20 = 20 %).

    Returns
    -------
    float
        Winsorized variance (sample variance, ddof=1).
    """
    y = np.sort(x)
    n = len(x)
    ibot = int(np.floor(trimming_level * n)) + 1
    itop = n - ibot + 1
    xbot = y[ibot - 1]
    xtop = y[itop - 1]
    y = np.where(y <= xbot, xbot, y)
    y = np.where(y >= xtop, xtop, y)
    return float(np.std(y, ddof=1) ** 2)


def winsorized_correlation(
    x: list[float] | NDArray[np.floating],
    y: list[float] | NDArray[np.floating],
    trimming_level: float = 0.2,
) -> dict[str, float]:
    """Compute the Winsorized correlation between two arrays.

    Parameters
    ----------
    x, y:
        Paired data arrays of equal length.
    trimming_level:
        Fraction trimmed from each tail (default 0.20).

    Returns
    -------
    dict with keys: ``cor``, ``cov``, ``p.value``, ``n``, ``test_statistic``.
    """
    sample_size = len(x)
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    k = int(np.floor(trimming_level * sample_size))
    x_lower, x_upper = x_sorted[k], x_sorted[sample_size - k - 1]
    y_lower, y_upper = y_sorted[k], y_sorted[sample_size - k - 1]
    x_w = np.clip(x, x_lower, x_upper)
    y_w = np.clip(y, y_lower, y_upper)
    cor = float(np.corrcoef(x_w, y_w)[0, 1])
    cov = float(np.cov(x_w, y_w)[0, 1])
    t_stat = cor * np.sqrt((sample_size - 2) / (1 - cor**2))
    p_value = float(
        2 * (1 - stats.t.cdf(abs(t_stat), sample_size - 2 * k - 2))
    )
    return {
        "cor": cor,
        "cov": cov,
        "p.value": p_value,
        "n": sample_size,
        "test_statistic": float(t_stat),
    }


def density(x: float) -> float:
    """x² · φ(x) — a weighting density used in numerical integration.

    Parameters
    ----------
    x:
        Input value.

    Returns
    -------
    float
    """
    return float(np.array(x) ** 2 * stats.norm.pdf(np.array(x)))


def area_under_function(
    f,
    a: float,
    b: float,
    *,
    limit: int = 10,
    eps: float = 1e-5,
) -> float:
    """Recursively compute ∫[a,b] f(x) dx using adaptive Simpson's rule.

    Parameters
    ----------
    f:
        Integrand callable ``f(x) -> float``.
    a, b:
        Integration limits.
    limit:
        Maximum recursion depth.
    eps:
        Absolute tolerance for the adaptive refinement.

    Returns
    -------
    float
    """

    def _simpson(fa: float, fm: float, fb: float, h: float) -> float:
        return (fa + 4 * fm + fb) * h / 6

    def _recurse(a_: float, b_: float, fa_: float, fb_: float, fm_: float, depth: int) -> float:
        mid = (a_ + b_) / 2
        h = b_ - a_
        whole = _simpson(fa_, fm_, fb_, h)
        lm = (a_ + mid) / 2
        rm = (mid + b_) / 2
        flm = f(lm)
        frm = f(rm)
        left = _simpson(fa_, flm, fm_, h / 2)
        right = _simpson(fm_, frm, fb_, h / 2)
        if abs(left + right - whole) < eps or depth == 0:
            return left + right
        return _recurse(a_, mid, fa_, fm_, flm, depth - 1) + _recurse(
            mid, b_, fm_, fb_, frm, depth - 1
        )

    fa = f(a)
    fb = f(b)
    fm = f((a + b) / 2)
    return _recurse(a, b, fa, fb, fm, limit)
