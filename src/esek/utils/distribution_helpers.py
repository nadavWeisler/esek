"""Distribution-based helpers: NCP search, pivotal CI, Fisher CI.

All functions work with scipy distributions (t, nct, ncf, norm).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def pivotal_ci_t(
    t_score: float,
    df: float,
    sample_size: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Compute the pivotal (non-central t) confidence interval for Cohen's d.

    Uses a bisection search on the non-central t CDF to find the NCP bounds
    that bracket the observed t-score at the given confidence level.

    Parameters
    ----------
    t_score:
        Observed t-statistic (may be negative).
    df:
        Degrees of freedom.
    sample_size:
        Total sample size (used to scale NCP → d).
    confidence_level:
        Desired confidence level, e.g. 0.95.

    Returns
    -------
    tuple[float, float]
        ``(ci_lower, ci_upper)`` for Cohen's d.
    """
    is_negative = t_score < 0
    if is_negative:
        t_score = abs(t_score)

    upper_limit = 1 - (1 - confidence_level) / 2
    lower_limit = (1 - confidence_level) / 2

    # --- lower NCP bracket ---
    lower_criterion = [-t_score, t_score / 2, t_score]
    while stats.nct.cdf(t_score, df, lower_criterion[0]) < upper_limit:
        lower_criterion = [lower_criterion[0] - t_score, lower_criterion[0], lower_criterion[2]]

    # --- upper NCP bracket ---
    upper_criterion = [t_score, 2 * t_score, 3 * t_score]
    while stats.nct.cdf(t_score, df, upper_criterion[0]) < lower_limit:
        if stats.nct.cdf(t_score, df) < lower_limit:
            upper_criterion = [upper_criterion[0] / 4, upper_criterion[0], upper_criterion[2]]
    while stats.nct.cdf(t_score, df, upper_criterion[2]) > lower_limit:
        upper_criterion = [upper_criterion[0], upper_criterion[2], upper_criterion[2] + t_score]

    # Bisect for lower CI
    lower_ci = 0.0
    diff = 1.0
    while diff > 1e-5:
        mid = lower_criterion[1]
        if stats.nct.cdf(t_score, df, mid) < upper_limit:
            lower_criterion = [lower_criterion[0], (lower_criterion[0] + mid) / 2, mid]
        else:
            lower_criterion = [mid, (mid + lower_criterion[2]) / 2, lower_criterion[2]]
        diff = abs(stats.nct.cdf(t_score, df, lower_criterion[1]) - upper_limit)
        lower_ci = lower_criterion[1] / math.sqrt(sample_size)

    # Bisect for upper CI
    upper_ci = 0.0
    diff = 1.0
    while diff > 1e-5:
        mid = upper_criterion[1]
        if stats.nct.cdf(t_score, df, mid) < lower_limit:
            upper_criterion = [upper_criterion[0], (upper_criterion[0] + mid) / 2, mid]
        else:
            upper_criterion = [mid, (mid + upper_criterion[2]) / 2, upper_criterion[2]]
        diff = abs(stats.nct.cdf(t_score, df, upper_criterion[1]) - lower_limit)
        upper_ci = upper_criterion[1] / math.sqrt(sample_size)

    if is_negative:
        return -upper_ci, -lower_ci
    return lower_ci, upper_ci


def ci_ncp(
    effect_size: float,
    sample_size: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Non-central parameter (NCP) confidence interval for a one-sample t-test.

    Converts the NCP bounds back to Cohen's d scale.

    Parameters
    ----------
    effect_size:
        Cohen's d estimate.
    sample_size:
        Sample size (n ≥ 2).
    confidence_level:
        Desired confidence level, e.g. 0.95.

    Returns
    -------
    tuple[float, float]
        ``(ci_lower, ci_upper)`` for Cohen's d.
    """
    df = sample_size - 1
    if df <= 0:
        raise ValueError(f"sample_size must be > 1 for NCP CI, got {sample_size}.")
    ncp = effect_size * math.sqrt(sample_size)
    q_low = float(stats.nct.ppf(0.5 - confidence_level / 2, df, nc=ncp))
    q_high = float(stats.nct.ppf(0.5 + confidence_level / 2, df, nc=ncp))
    ci_low = q_low / ncp * effect_size
    ci_high = q_high / ncp * effect_size
    return ci_low, ci_high


def compute_fisher_confidence_interval(
    correlation: float,
    standard_error: float,
    z_critical: float,
) -> tuple[float, float]:
    """Fisher z CI for a correlation coefficient.

    Transforms to Fisher z, applies the margin, and transforms back.

    Parameters
    ----------
    correlation:
        Sample correlation (clamped to ±0.999999 internally).
    standard_error:
        Standard error of the Fisher z estimate.
    z_critical:
        Critical z-value (e.g. 1.96 for 95 %).

    Returns
    -------
    tuple[float, float]
        ``(ci_lower, ci_upper)`` bounded to [−1, 1].
    """
    safe = max(min(correlation, 0.999999), -0.999999)
    fz = math.atanh(safe)
    margin = z_critical * standard_error
    lower = max(math.tanh(fz - margin), -1.0)
    upper = min(math.tanh(fz + margin), 1.0)
    return lower, upper
