"""Confidence-interval methods for odds ratios.

CIs are computed on the log-OR scale and then exponentiated.
"""

from __future__ import annotations

import math

from scipy import stats


def log_scale_ci(
    or_: float,
    se_log_or: float,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """CI for an odds ratio using the log-scale (Wald) method.

    Constructs the CI on log(OR) scale and exponentiates:

        CI = (exp(log(OR) − z·SE), exp(log(OR) + z·SE))

    Parameters
    ----------
    or_:
        Point estimate of the odds ratio (must be > 0).
    se_log_or:
        Standard error of log(OR).
    confidence_level:
        Desired CI level, e.g. 0.95.

    Returns
    -------
    (ci_lower, ci_upper) — both on the OR (not log-OR) scale.

    Raises
    ------
    ValueError
        If *or_* is not strictly positive.
    """
    if or_ <= 0.0:
        raise ValueError(f"or_ must be > 0, got {or_}.")
    log_or = math.log(or_)
    z = float(stats.norm.ppf(0.5 + confidence_level / 2))
    ci_lower = math.exp(log_or - z * se_log_or)
    ci_upper = math.exp(log_or + z * se_log_or)
    return ci_lower, ci_upper
