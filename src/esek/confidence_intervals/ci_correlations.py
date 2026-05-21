"""Confidence-interval methods for correlation-based effect sizes.

References
----------
- Fisher (1921) On the "probable error" of a coefficient of correlation
"""

from __future__ import annotations

import math

from scipy import stats


def fisher_z_ci(
    r: float,
    n: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Fisher *z′* CI for Pearson's *r*.

    Transforms *r* to Fisher z′, constructs a symmetric CI in z′ space
    (variance ≈ 1/(n−3)), then transforms back.

    Parameters
    ----------
    r:
        Pearson correlation (clamped to ±0.999999 internally).
    n:
        Sample size.
    confidence_level:
        Desired CI level, e.g. 0.95.

    Returns
    -------
    (ci_lower, ci_upper) bounded to [−1, 1].

    Raises
    ------
    ValueError
        If *n* ≤ 3 (variance formula requires n > 3).
    """
    if n <= 3:
        raise ValueError(f"n must be > 3 for Fisher z CI, got n={n}.")
    safe_r = max(min(float(r), 0.999999), -0.999999)
    fz = math.atanh(safe_r)
    se = 1.0 / math.sqrt(n - 3)
    z = float(stats.norm.ppf(0.5 + confidence_level / 2))
    ci_lower = max(math.tanh(fz - z * se), -1.0)
    ci_upper = min(math.tanh(fz + z * se), 1.0)
    return ci_lower, ci_upper
