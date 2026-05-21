"""Confidence intervals for dispersion measures.

Migrated and refactored from:
``stats/CI_Constructor/8_DispersionMeasures/Dispersion Measures CI.py``
in the ``dev`` branch.

Provides CI for:
- Mean Absolute Deviation (MAD) from median — log-scale delta method
- Sample standard deviation — chi² exact CI

Statistical assumptions:
    - MAD CI uses the delta method on the log-scale (first-order approximation).
      May be inaccurate for small samples (n < 20) or highly non-normal data.
    - SD CI uses the exact chi²-based pivotal method, valid under normality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm, chi2


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MADCIResult:
    """CI for the Mean Absolute Deviation (from the median).

    Attributes:
        mad: Sample MAD (= mean(|xᵢ − median(x)|)).
        mad_corrected: Bias-corrected MAD (× n/(n−1)).
        ci_low: Lower confidence bound for corrected MAD.
        ci_high: Upper confidence bound for corrected MAD.
        se: Asymptotic standard error (delta method on log scale).
        n: Sample size.
        confidence_level: Nominal CI level.
        metadata: Additional quantities.
    """

    mad: float
    mad_corrected: float
    ci_low: float
    ci_high: float
    se: float
    n: int
    confidence_level: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SDCIResult:
    """Exact chi²-based CI for the sample standard deviation.

    Attributes:
        sd: Sample SD (ddof=1).
        ci_low: Lower confidence bound for SD.
        ci_high: Upper confidence bound for SD.
        variance: Sample variance (sd²).
        var_ci_low: Lower bound for variance.
        var_ci_high: Upper bound for variance.
        n: Sample size.
        df: Degrees of freedom (n−1).
        confidence_level: Nominal CI level.
        metadata: Additional quantities.

    Notes:
        This CI assumes approximate normality. For heavy-tailed distributions,
        bootstrap or robust alternatives are preferred.
    """

    sd: float
    ci_low: float
    ci_high: float
    variance: float
    var_ci_low: float
    var_ci_high: float
    n: int
    df: int
    confidence_level: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def mad_ci(
    data: "np.ndarray | list[float]",
    confidence_level: float = 0.95,
) -> MADCIResult:
    """CI for the Mean Absolute Deviation from the median.

    Uses the delta method on the log scale (Gastwirth, 1982 approximation):

        SE_log(MAD) = sqrt(((mean - median) / MAD)² + (SD / MAD)² − 1) / sqrt(n)
        CI = exp(log(MAD_corrected) ± z * SE)

    Parameters:
        data: 1-D array-like of numeric values.
        confidence_level: Nominal CI level (default 0.95).

    Returns:
        :class:`MADCIResult`.

    Raises:
        ValueError: For invalid inputs.

    Note:
        The CI may be inaccurate for n < 20 or heavy-tailed distributions.
        Consider bootstrap CIs for those cases.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError("data must be 1-dimensional.")
    n = len(arr)
    if n < 3:
        raise ValueError(f"Sample size must be ≥ 3 for MAD CI (got {n}).")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1) (got {confidence_level}).")

    median = float(np.median(arr))
    mad = float(np.mean(np.abs(arr - median)))
    if mad == 0.0:
        # Degenerate case: all values equal or constant within rounding
        raise ValueError("MAD is zero — data has no spread. Cannot compute CI.")

    mad_corrected = mad * n / (n - 1)
    sd = float(np.std(arr, ddof=1))
    mean = float(np.mean(arr))

    se = float(math.sqrt(
        max(((mean - median) / mad)**2 + (sd / mad)**2 - 1.0, 0.0) / n
    ))
    z_crit = float(norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))

    log_mad = math.log(mad_corrected)
    ci_low = math.exp(log_mad - z_crit * se)
    ci_high = math.exp(log_mad + z_crit * se)

    return MADCIResult(
        mad=round(float(mad), 6),
        mad_corrected=round(float(mad_corrected), 6),
        ci_low=round(ci_low, 6),
        ci_high=round(ci_high, 6),
        se=round(se, 6),
        n=int(n),
        confidence_level=confidence_level,
        metadata={"mean": round(mean, 6), "median": round(median, 6), "sd": round(sd, 6)},
    )


def sd_ci(
    data: "np.ndarray | list[float]",
    confidence_level: float = 0.95,
) -> SDCIResult:
    """Exact chi²-based CI for the sample standard deviation.

    Under normality, (n−1)·s²/σ² ~ χ²(n−1). The pivotal CI for σ is:

        σ ∈ (s·sqrt(df / χ²_{α/2}), s·sqrt(df / χ²_{1−α/2}))

    Parameters:
        data: 1-D array-like of numeric values.
        confidence_level: Nominal CI level (default 0.95).

    Returns:
        :class:`SDCIResult`.

    Raises:
        ValueError: For invalid inputs.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError("data must be 1-dimensional.")
    n = len(arr)
    if n < 2:
        raise ValueError(f"Sample size must be ≥ 2 for SD CI (got {n}).")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1) (got {confidence_level}).")

    df = n - 1
    s = float(np.std(arr, ddof=1))
    variance = s**2
    alpha = 1.0 - confidence_level

    chi2_lo = float(chi2.ppf(alpha / 2.0, df))
    chi2_hi = float(chi2.ppf(1.0 - alpha / 2.0, df))

    var_ci_lo = df * variance / chi2_hi
    var_ci_hi = df * variance / chi2_lo

    return SDCIResult(
        sd=round(s, 6),
        ci_low=round(math.sqrt(var_ci_lo), 6),
        ci_high=round(math.sqrt(var_ci_hi), 6),
        variance=round(variance, 6),
        var_ci_low=round(var_ci_lo, 6),
        var_ci_high=round(var_ci_hi, 6),
        n=int(n),
        df=int(df),
        confidence_level=confidence_level,
    )
