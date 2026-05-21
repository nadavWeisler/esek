"""Confidence-interval methods for ANOVA effect sizes.

Provides CIs for partial η², partial ω², and partial ε² via the
non-central F distribution (NCP bisection method).

Migrated from:
``stats/CI_Constructor/3_EtaSquareFamily/CI_Constructor_eta.py``
in the ``dev`` branch.

References:
    - Fleishman (1980)
    - Steiger & Fouladi (1997)
    - Smithson (2003) — cautions on using ε² CIs in practice
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from scipy.stats import ncf


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EtaSquaredCIResult:
    """Confidence intervals for partial η² and related ANOVA effect sizes.

    Attributes:
        partial_eta_squared: Point estimate (input or derived from F).
        f_statistic: The F statistic used to compute CIs.
        df1: Numerator degrees of freedom.
        df2: Denominator degrees of freedom.
        confidence_level: Nominal CI level.
        ci_partial_eta_sq_fleishman: CI via Fleishman (1980) NCP method.
        ci_partial_eta_sq_f_method: CI via F-conversion method.
        ci_partial_omega_sq: CI for partial ω² via F-conversion.
        ci_partial_epsilon_sq: CI for partial ε² via F-conversion.
        metadata: Additional quantities.

    Notes:
        Smithson (2003) cautions against interpreting ε² CIs at face value —
        they are derived transformations and may not behave as expected.
    """

    partial_eta_squared: float
    f_statistic: float
    df1: int
    df2: int
    confidence_level: float
    ci_partial_eta_sq_fleishman: tuple[float, float]
    ci_partial_eta_sq_f_method: tuple[float, float]
    ci_partial_omega_sq: tuple[float, float]
    ci_partial_epsilon_sq: tuple[float, float]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal NCP-F bisection
# ---------------------------------------------------------------------------


def _ncp_f_ci(
    f_statistic: float,
    df1: int,
    df2: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Non-central F CI via bisection on the NCP parameter.

    Returns ``(ncp_lower, ncp_upper)`` such that::

        P(F(df1, df2, ncp_lo) ≥ f) = 1 − α/2
        P(F(df1, df2, ncp_hi) ≥ f) = α/2
    """
    alpha = 1.0 - confidence_level
    upper_tail = 1.0 - alpha / 2
    lower_tail = alpha / 2
    tol_lo, tol_hi = 1e-7, 1e-5

    # --- lower NCP ---
    lo = [1e-3, f_statistic / 2.0, f_statistic]
    if ncf.cdf(f_statistic, df1, df2, lo[0]) < upper_tail:
        ncp_lo = 0.0
    else:
        while ncf.cdf(f_statistic, df1, df2, lo[2]) > upper_tail:
            lo = [lo[0], lo[2], lo[2] + f_statistic]
        diff = 1.0
        while diff > tol_lo:
            if ncf.cdf(f_statistic, df1, df2, lo[1]) < upper_tail:
                lo = [lo[0], (lo[0] + lo[1]) / 2.0, lo[1]]
            else:
                lo = [lo[1], (lo[1] + lo[2]) / 2.0, lo[2]]
            diff = abs(ncf.cdf(f_statistic, df1, df2, lo[1]) - upper_tail)
        ncp_lo = lo[1]

    # --- upper NCP ---
    hi = [f_statistic, 2.0 * f_statistic, 3.0 * f_statistic]
    while ncf.cdf(f_statistic, df1, df2, hi[0]) < lower_tail:
        hi = [hi[0] / 4.0, hi[0], hi[2]]
    while ncf.cdf(f_statistic, df1, df2, hi[2]) > lower_tail:
        hi = [hi[0], hi[2], hi[2] + f_statistic]
    diff = 1.0
    while diff > tol_hi:
        if ncf.cdf(f_statistic, df1, df2, hi[1]) < lower_tail:
            hi = [hi[0], (hi[0] + hi[1]) / 2.0, hi[1]]
        else:
            hi = [hi[1], (hi[1] + hi[2]) / 2.0, hi[2]]
        diff = abs(ncf.cdf(f_statistic, df1, df2, hi[1]) - lower_tail)
    ncp_hi = hi[1]

    return ncp_lo, ncp_hi


# ---------------------------------------------------------------------------
# Public calculator
# ---------------------------------------------------------------------------


class EtaSquaredCI:
    """Confidence intervals for partial η², ω², and ε² effect sizes.

    Example::

        result = EtaSquaredCI.from_f(f_statistic=4.2, df1=2, df2=57)
        print(result.ci_partial_eta_sq_fleishman)
        print(result.ci_partial_omega_sq)
    """

    @staticmethod
    def from_f(
        f_statistic: float,
        df1: int,
        df2: int,
        confidence_level: float = 0.95,
    ) -> EtaSquaredCIResult:
        """Compute CIs from an F-statistic.

        Parameters:
            f_statistic: Observed F value (must be ≥ 0).
            df1: Numerator df (≥ 1).
            df2: Denominator df (≥ 1).
            confidence_level: Nominal level, e.g. 0.95.

        Returns:
            :class:`EtaSquaredCIResult`.
        """
        if f_statistic < 0:
            raise ValueError(f"f_statistic must be ≥ 0 (got {f_statistic}).")
        if df1 < 1 or df2 < 1:
            raise ValueError(f"Degrees of freedom must be ≥ 1 (df1={df1}, df2={df2}).")
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in (0, 1) (got {confidence_level}).")

        eta_sq = (f_statistic * df1) / (f_statistic * df1 + df2)
        return EtaSquaredCI._compute(f_statistic, eta_sq, df1, df2, confidence_level)

    @staticmethod
    def from_partial_eta_squared(
        partial_eta_sq: float,
        df1: int,
        df2: int,
        confidence_level: float = 0.95,
    ) -> EtaSquaredCIResult:
        """Compute CIs from a partial η² value.

        Converts partial η² back to an F-statistic then applies the same
        NCP-based algorithm.  Formula: F = η²·df2 / ((1 − η²)·df1).

        Parameters:
            partial_eta_sq: Partial η² value in (0, 1).
            df1: Numerator df (≥ 1).
            df2: Denominator df (≥ 1).
            confidence_level: Nominal level.

        Returns:
            :class:`EtaSquaredCIResult`.
        """
        if not (0.0 < partial_eta_sq < 1.0):
            raise ValueError(f"partial_eta_sq must be in (0, 1) (got {partial_eta_sq}).")
        if df1 < 1 or df2 < 1:
            raise ValueError(f"Degrees of freedom must be ≥ 1 (df1={df1}, df2={df2}).")
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in (0, 1) (got {confidence_level}).")

        f_stat = (-partial_eta_sq * df2) / (partial_eta_sq * df1 - df1)
        return EtaSquaredCI._compute(f_stat, partial_eta_sq, df1, df2, confidence_level)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute(
        f_statistic: float,
        eta_sq: float,
        df1: int,
        df2: int,
        confidence_level: float,
    ) -> EtaSquaredCIResult:
        ncp_lo, ncp_hi = _ncp_f_ci(f_statistic, df1, df2, confidence_level)

        # Method 1 — Fleishman (1980) / "R-square method"
        # η² = NCP / (NCP + df1 + df2 + 1)
        lo_eta_fleishman = ncp_lo / (ncp_lo + df1 + df2 + 1)
        hi_eta_fleishman = ncp_hi / (ncp_hi + df1 + df2 + 1)

        # Method 2 — F-converted method
        # η² = (NCP/df1 · df1) / (NCP/df1 · df1 + df2)
        # simplifies to: NCP / (NCP + df2)
        lo_eta_f = ncp_lo / (ncp_lo + df2) if (ncp_lo + df2) > 0 else 0.0
        hi_eta_f = ncp_hi / (ncp_hi + df2) if (ncp_hi + df2) > 0 else 0.0

        # Partial ω² CI (bias-corrected η²)
        # ω² = (NCP/df1 − 1) · df1 / ((NCP/df1) · df1 + df2 + 1)
        def _omega(ncp: float) -> float:
            return ((ncp / df1 - 1.0) * df1) / (ncp + df2 + 1.0) if (ncp + df2 + 1.0) > 0 else math.nan

        # Partial ε² CI
        # ε² = (NCP/df1 − 1) · df1 / ((NCP/df1) · df1 + df2)
        def _epsilon(ncp: float) -> float:
            return ((ncp / df1 - 1.0) * df1) / (ncp + df2) if (ncp + df2) > 0 else math.nan

        return EtaSquaredCIResult(
            partial_eta_squared=round(float(eta_sq), 6),
            f_statistic=round(float(f_statistic), 6),
            df1=int(df1),
            df2=int(df2),
            confidence_level=confidence_level,
            ci_partial_eta_sq_fleishman=(round(lo_eta_fleishman, 6), round(hi_eta_fleishman, 6)),
            ci_partial_eta_sq_f_method=(round(lo_eta_f, 6), round(hi_eta_f, 6)),
            ci_partial_omega_sq=(round(_omega(ncp_lo), 6), round(_omega(ncp_hi), 6)),
            ci_partial_epsilon_sq=(round(_epsilon(ncp_lo), 6), round(_epsilon(ncp_hi), 6)),
            metadata={"ncp_lower": ncp_lo, "ncp_upper": ncp_hi},
        )
