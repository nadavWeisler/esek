"""Confidence-interval methods for correlation-based effect sizes.

References
----------
- Fisher (1921) On the "probable error" of a coefficient of correlation
- Bonett & Wright (2000) — Spearman CI
- Fieller, Hartley & Pearson (1957) — Spearman CI
- Caruso & Cliff (1997) — Spearman CI
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import ncx2


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


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpearmanCIResult:
    """Multiple CI estimates for Spearman ρ.

    The recommended default CI is Bonett & Wright (2000) with z-critical value.

    Attributes:
        rho: Spearman ρ point estimate.
        n: Sample size.
        confidence_level: Nominal CI level.
        ci_bonett_wright_z: Bonett & Wright (2000) CI using z-critical.
        ci_bonett_wright_t: Bonett & Wright (2000) CI using t-critical.
        ci_fieller: Fieller et al. (1957) CI.
        ci_fisher_z: Fisher transformation CI using z-critical.
        ci_fisher_t: Fisher transformation CI using t-critical.
        metadata: Additional SE values.
    """

    rho: float
    n: int
    confidence_level: float
    ci_bonett_wright_z: tuple[float, float]
    ci_bonett_wright_t: tuple[float, float]
    ci_fieller: tuple[float, float]
    ci_fisher_z: tuple[float, float]
    ci_fisher_t: tuple[float, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CramerVCIResult:
    """Confidence interval for Cramér's V.

    Attributes:
        cramer_v: Cramér's V point estimate.
        chi_square: χ² statistic used.
        n: Sample size.
        df: Degrees of freedom for χ².
        confidence_level: Nominal CI level.
        ci: NCP-based CI (lower, upper).
    """

    cramer_v: float
    chi_square: float
    n: int
    df: int
    confidence_level: float
    ci: tuple[float, float]


@dataclass(frozen=True)
class CohensWCIResult:
    """Confidence interval for Cohen's *w* (and φ for 2×2 tables).

    Attributes:
        cohens_w: Cohen's *w* point estimate.
        chi_square: χ² statistic used.
        n: Sample size.
        df: Degrees of freedom for χ².
        confidence_level: Nominal CI level.
        ci: NCP-based CI (lower, upper).
    """

    cohens_w: float
    chi_square: float
    n: int
    df: int
    confidence_level: float
    ci: tuple[float, float]


@dataclass(frozen=True)
class ContingencyCoefficientCIResult:
    """Confidence interval for Pearson's contingency coefficient *C*.

    Attributes:
        contingency_coefficient: Contingency coefficient point estimate.
        chi_square: χ² statistic used.
        n: Sample size.
        df: Degrees of freedom for χ².
        confidence_level: Nominal CI level.
        ci: NCP-based CI (lower, upper).
    """

    contingency_coefficient: float
    chi_square: float
    n: int
    df: int
    confidence_level: float
    ci: tuple[float, float]


# ---------------------------------------------------------------------------
# Internal NCP χ² CI
# ---------------------------------------------------------------------------


def _ncp_chi2_ci(
    chival: float,
    df: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Non-central χ² CI via bisection.

    Returns ``(ncp_lower, ncp_upper)``.
    """
    alpha = 1.0 - confidence_level
    ulim = 1.0 - alpha / 2
    llim = alpha / 2

    # lower NCP
    lo = [1e-3, chival / 2.0, chival]
    if ncx2.cdf(chival, df, lo[0]) < ulim:
        ncp_lo = 0.0
    else:
        diff = 1.0
        while diff > 1e-5:
            if ncx2.cdf(chival, df, lo[1]) < ulim:
                lo = [lo[0], (lo[0] + lo[1]) / 2.0, lo[1]]
            else:
                lo = [lo[1], (lo[1] + lo[2]) / 2.0, lo[2]]
            diff = abs(ncx2.cdf(chival, df, lo[1]) - ulim)
        ncp_lo = lo[1]

    # upper NCP
    hi = [chival, 2.0 * chival, 3.0 * chival]
    while ncx2.cdf(chival, df, hi[0]) < llim:
        hi = [hi[0] / 4.0, hi[0], hi[2]]
    while ncx2.cdf(chival, df, hi[2]) > llim:
        hi = [hi[0], hi[2], hi[2] + chival]
    diff = 1.0
    while diff > 1e-5:
        if ncx2.cdf(chival, df, hi[1]) < llim:
            hi = [hi[0], (hi[0] + hi[1]) / 2.0, hi[1]]
        else:
            hi = [hi[1], (hi[1] + hi[2]) / 2.0, hi[2]]
        diff = abs(ncx2.cdf(chival, df, hi[1]) - llim)
    ncp_hi = hi[1]

    return ncp_lo, ncp_hi


# ---------------------------------------------------------------------------
# Spearman CI
# ---------------------------------------------------------------------------


def spearman_ci(
    rho: float,
    n: int,
    confidence_level: float = 0.95,
) -> SpearmanCIResult:
    """Confidence intervals for Spearman's ρ using multiple methods.

    The recommended default is the Bonett & Wright (2000) CI with z-critical
    value, which outperforms Fisher-only in simulation studies.

    Parameters:
        rho: Spearman ρ (must be in (−1, 1)).
        n: Sample size (must be ≥ 5 for Bonett-Wright).
        confidence_level: Nominal CI level.

    Returns:
        :class:`SpearmanCIResult` with 5 CI variants.
    """
    if not (-1.0 < rho < 1.0):
        raise ValueError(f"rho must be in (−1, 1) (got {rho}).")
    if n < 5:
        raise ValueError(f"n must be ≥ 5 for Spearman CI (got {n}).")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1) (got {confidence_level}).")

    z_rho = math.atanh(rho)
    z_crit = stats.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)
    t_crit = stats.t.ppf(1.0 - (1.0 - confidence_level) / 2.0, n - 2)

    se_fieller = math.sqrt(1.06 / (n - 3))
    se_caruso_cliff = math.sqrt(1.0 / (n - 2)) + abs(z_rho) / (6.0 * n + 4.0 * math.sqrt(n))
    se_bonett_wright = math.sqrt((1.0 + rho / 2.0) / (n - 3))
    se_fisher = math.sqrt(1.0 / (n - 3))

    def _ci(se: float, crit: float) -> tuple[float, float]:
        return (
            max(math.tanh(z_rho - crit * se), -1.0),
            min(math.tanh(z_rho + crit * se), 1.0),
        )

    return SpearmanCIResult(
        rho=float(rho),
        n=n,
        confidence_level=confidence_level,
        ci_bonett_wright_z=_ci(se_bonett_wright, z_crit),
        ci_bonett_wright_t=_ci(se_bonett_wright, t_crit),
        ci_fieller=_ci(se_fieller, z_crit),
        ci_fisher_z=_ci(se_fisher, z_crit),
        ci_fisher_t=_ci(se_fisher, t_crit),
        metadata={
            "se_bonett_wright": round(se_bonett_wright, 6),
            "se_fieller": round(se_fieller, 6),
            "se_fisher": round(se_fisher, 6),
            "fisher_z": round(z_rho, 6),
        },
    )


# ---------------------------------------------------------------------------
# Cramér's V CI
# ---------------------------------------------------------------------------


def cramer_v_ci(
    cramer_v: float,
    n: int,
    df: int,
    confidence_level: float = 0.95,
) -> CramerVCIResult:
    """Non-central χ²–based CI for Cramér's V.

    Converts V → χ², finds the NCP CI, then converts back to V.

    Parameters:
        cramer_v: Cramér's V value (≥ 0).
        n: Total sample size.
        df: Degrees of freedom = (rows − 1) or (cols − 1) for 2-level tables.
        confidence_level: Nominal CI level.

    Returns:
        :class:`CramerVCIResult`.
    """
    if cramer_v < 0:
        raise ValueError(f"cramer_v must be ≥ 0 (got {cramer_v}).")
    if n < 2:
        raise ValueError(f"n must be ≥ 2 (got {n}).")
    if df < 1:
        raise ValueError(f"df must be ≥ 1 (got {df}).")

    chi_sq = cramer_v**2 * n * df
    ncp_lo, ncp_hi = _ncp_chi2_ci(chi_sq, df, confidence_level)
    lo_v = math.sqrt(ncp_lo / n / df) if n * df > 0 else 0.0
    hi_v = math.sqrt(ncp_hi / n / df)
    return CramerVCIResult(
        cramer_v=float(cramer_v),
        chi_square=float(chi_sq),
        n=n,
        df=df,
        confidence_level=confidence_level,
        ci=(round(lo_v, 6), round(hi_v, 6)),
    )


# ---------------------------------------------------------------------------
# Cohen's w CI
# ---------------------------------------------------------------------------


def cohens_w_ci(
    cohens_w: float,
    n: int,
    df: int,
    confidence_level: float = 0.95,
) -> CohensWCIResult:
    """Non-central χ²–based CI for Cohen's *w*.

    Converts *w* → χ² = *w*²·*n*, finds the NCP CI, then converts back with
    ``√(λ / n)``.

    Parameters
    ----------
    cohens_w:
        Cohen's *w* value (≥ 0).
    n:
        Total sample size.
    df:
        Degrees of freedom for the χ² distribution.
    confidence_level:
        Nominal CI level in ``(0, 1)``.
    """
    if cohens_w < 0:
        raise ValueError(f"cohens_w must be ≥ 0 (got {cohens_w}).")
    if n < 2:
        raise ValueError(f"n must be ≥ 2 (got {n}).")
    if df < 1:
        raise ValueError(f"df must be ≥ 1 (got {df}).")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1) (got {confidence_level}).")

    chi_sq = float(cohens_w) ** 2 * n
    ncp_lo, ncp_hi = _ncp_chi2_ci(chi_sq, df, confidence_level)
    lo_w = math.sqrt(ncp_lo / n) if n > 0 else 0.0
    hi_w = math.sqrt(ncp_hi / n)
    return CohensWCIResult(
        cohens_w=float(cohens_w),
        chi_square=float(chi_sq),
        n=n,
        df=df,
        confidence_level=confidence_level,
        ci=(round(lo_w, 6), round(hi_w, 6)),
    )


# ---------------------------------------------------------------------------
# Contingency coefficient CI
# ---------------------------------------------------------------------------


def contingency_coefficient_ci(
    contingency_coefficient: float,
    n: int,
    df: int,
    confidence_level: float = 0.95,
) -> ContingencyCoefficientCIResult:
    """Non-central χ²–based CI for Pearson's contingency coefficient *C*.

    Converts *C* → χ² = (*C*²·*n*) / (1 − *C*²), finds the NCP CI, then
    converts back with ``√(λ / (λ + n))``.

    Parameters
    ----------
    contingency_coefficient:
        Contingency coefficient in ``[0, 1)``.
    n:
        Total sample size.
    df:
        Degrees of freedom for the χ² distribution.
    confidence_level:
        Nominal CI level in ``(0, 1)``.
    """
    c_val = float(contingency_coefficient)
    if not (0.0 <= c_val < 1.0):
        raise ValueError(
            f"contingency_coefficient must be in [0, 1) (got {contingency_coefficient})."
        )
    if n < 2:
        raise ValueError(f"n must be ≥ 2 (got {n}).")
    if df < 1:
        raise ValueError(f"df must be ≥ 1 (got {df}).")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1) (got {confidence_level}).")

    denom = 1.0 - c_val**2
    if denom <= 0.0:
        raise ValueError("contingency_coefficient must be strictly less than 1.")

    chi_sq = (c_val**2 * n) / denom
    ncp_lo, ncp_hi = _ncp_chi2_ci(chi_sq, df, confidence_level)
    lo_c = math.sqrt(ncp_lo / (ncp_lo + n)) if (ncp_lo + n) > 0 else 0.0
    hi_c = math.sqrt(ncp_hi / (ncp_hi + n))
    return ContingencyCoefficientCIResult(
        contingency_coefficient=c_val,
        chi_square=float(chi_sq),
        n=n,
        df=df,
        confidence_level=confidence_level,
        ci=(round(lo_c, 6), round(hi_c, 6)),
    )

