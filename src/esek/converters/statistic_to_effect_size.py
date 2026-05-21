"""Converters from common test statistics to Cohen's d effect sizes.

Migrated and refactored from:
``stats/Converter/1_ConvertFromDifferentStatisticsToEffectSizes/From_Statistic_to_an_Effect_Size.py``
in the ``dev`` branch.

Formulas follow:
    - Rosenthal (1994) — z → d
    - Cohen (1988) — t → d
    - Lakens (2013) — practical guidance

Notes on statistical assumptions:
    - ``z_one_sample`` and ``z_paired``: d = z / √n.  Same formula; both
      included for clarity since the interpretation differs.
    - ``z_independent``: d = 2z / √N · √(n̄ · N / (2 · n₁ · n₂)) where N
      = n₁ + n₂ and n̄ is the harmonic mean.  Simplifies to the formula in
      the original code.
    - ``t_one_sample``: d = t / √df (df = n − 1).
    - ``t_paired``: d = t / √n.
    - ``t_independent``: d = t · √(1/n₁ + 1/n₂) (exact pooled-SD formula).
      The original code contained a bug (used undefined ``sample_size``); this
      implementation uses the correct formula.

Bias correction (Hedges' g) uses the log-gamma approximation for numerical
stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatisticConversionResult:
    """Result of converting a test statistic to an effect size.

    Attributes:
        cohens_d: Cohen's d estimate.
        hedges_g: Bias-corrected d (Hedges' g).  ``None`` if df is unavailable.
        input_statistic: Name of the input statistic (e.g. ``"t"``).
        input_value: Numeric value of the input statistic.
        design: Design type (``"one_sample"``, ``"paired"``, or ``"independent"``).
        metadata: Additional quantities (sample sizes, df, etc.).
    """

    cohens_d: float
    hedges_g: float | None
    input_statistic: str
    input_value: float
    design: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _bias_correction(df: float) -> float:
    """Compute Hedges' bias-correction factor J(df).

    Uses the log-gamma form for numerical stability across all df.

    J(df) = exp(lgamma(df/2) − log(√(df/2)) − lgamma((df−1)/2))

    References:
        Hedges & Olkin (1985) — equation 4.23.
    """
    if df < 1:
        return math.nan
    return math.exp(
        math.lgamma(df / 2.0)
        - math.log(math.sqrt(df / 2.0))
        - math.lgamma((df - 1.0) / 2.0)
    )


# ---------------------------------------------------------------------------
# Public calculator class
# ---------------------------------------------------------------------------


class StatisticToEffectSize:
    """Convert test statistics (z, t) to standardised effect sizes.

    All methods are static and return a :class:`StatisticConversionResult`.

    Example::

        result = StatisticToEffectSize.from_t_one_sample(t=2.5, n=30)
        print(result.cohens_d, result.hedges_g)
    """

    # ------------------------------------------------------------------
    # z → Cohen's d
    # ------------------------------------------------------------------

    @staticmethod
    def from_z_one_sample(z: float, n: int) -> StatisticConversionResult:
        """Convert a one-sample z-score to Cohen's d.

        Formula:  d = z / √n

        Parameters:
            z: The observed z-statistic.
            n: Sample size (n ≥ 2).

        Returns:
            :class:`StatisticConversionResult`.
        """
        _validate_n(n, min_n=2)
        d = z / math.sqrt(n)
        return StatisticConversionResult(
            cohens_d=round(d, 6),
            hedges_g=None,  # no df available for this formula
            input_statistic="z",
            input_value=float(z),
            design="one_sample",
            metadata={"n": n},
        )

    @staticmethod
    def from_z_paired(z: float, n: int) -> StatisticConversionResult:
        """Convert a paired-samples z-score to Cohen's d.

        Formula:  d = z / √n  (same as one-sample; included for clarity)

        Parameters:
            z: The observed z-statistic.
            n: Number of pairs (n ≥ 2).

        Returns:
            :class:`StatisticConversionResult`.
        """
        _validate_n(n, min_n=2)
        d = z / math.sqrt(n)
        return StatisticConversionResult(
            cohens_d=round(d, 6),
            hedges_g=None,
            input_statistic="z",
            input_value=float(z),
            design="paired",
            metadata={"n": n},
        )

    @staticmethod
    def from_z_independent(z: float, n1: int, n2: int) -> StatisticConversionResult:
        """Convert an independent-samples z-score to Cohen's d.

        Formula:  d = (2z / √N) · √(n̄ · N / (2 · n₁ · n₂))
        where N = n₁ + n₂, n̄ = harmonic_mean(n₁, n₂).

        This simplifies to the formula used in the original implementation
        and is equivalent to Rosenthal (1994), formula 2.6.

        Parameters:
            z: The observed z-statistic.
            n1: Sample size of group 1 (n₁ ≥ 2).
            n2: Sample size of group 2 (n₂ ≥ 2).

        Returns:
            :class:`StatisticConversionResult`.
        """
        _validate_n(n1, min_n=2, label="n1")
        _validate_n(n2, min_n=2, label="n2")
        total = n1 + n2
        harmonic_mean = 2.0 * n1 * n2 / total  # harmonic mean × 2 = 2n1n2/(n1+n2)
        d = (2.0 * z / math.sqrt(total)) * math.sqrt(harmonic_mean / (2.0 * n1 * n2 / total))
        # simplifies: d = z * sqrt(total) / sqrt(2 * n1 * n2 / total * total/2)
        # = 2z/sqrt(N) * sqrt(n_bar * N / (2*n1*n2))
        return StatisticConversionResult(
            cohens_d=round(d, 6),
            hedges_g=None,
            input_statistic="z",
            input_value=float(z),
            design="independent",
            metadata={"n1": n1, "n2": n2, "total_n": total},
        )

    # ------------------------------------------------------------------
    # t → Cohen's d
    # ------------------------------------------------------------------

    @staticmethod
    def from_t_one_sample(t: float, n: int) -> StatisticConversionResult:
        """Convert a one-sample t-statistic to Cohen's d and Hedges' g.

        Formula:  d = t / √df  where df = n − 1.

        Parameters:
            t: The observed t-statistic.
            n: Sample size (n ≥ 2).

        Returns:
            :class:`StatisticConversionResult` with Hedges' g.
        """
        _validate_n(n, min_n=2)
        df = n - 1
        d = t / math.sqrt(df)
        j = _bias_correction(df)
        g = j * d
        return StatisticConversionResult(
            cohens_d=round(d, 6),
            hedges_g=round(g, 6),
            input_statistic="t",
            input_value=float(t),
            design="one_sample",
            metadata={"n": n, "df": df, "bias_correction_j": round(j, 6)},
        )

    @staticmethod
    def from_t_paired(t: float, n: int) -> StatisticConversionResult:
        """Convert a paired-samples t-statistic to Cohen's d.

        Formula:  d = t / √n

        Note: This gives dav (Cohen's d for average SD), not drm (which also
        accounts for the correlation between pairs).  Use dav when the
        between-pair correlation is unknown.

        Parameters:
            t: The observed t-statistic.
            n: Number of pairs (n ≥ 2).

        Returns:
            :class:`StatisticConversionResult`.
        """
        _validate_n(n, min_n=2)
        d = t / math.sqrt(n)
        return StatisticConversionResult(
            cohens_d=round(d, 6),
            hedges_g=None,
            input_statistic="t",
            input_value=float(t),
            design="paired",
            metadata={"n": n},
        )

    @staticmethod
    def from_t_independent(t: float, n1: int, n2: int) -> StatisticConversionResult:
        """Convert an independent-samples t-statistic to Cohen's d.

        Formula:  d = t · √(1/n₁ + 1/n₂)

        This is the exact formula when both groups use the pooled SD denominator
        (Student's t-test, equal variances assumed).  See Borenstein et al.
        (2009), eq. 4.18.

        Note: The original source code contained a bug (referenced an undefined
        variable ``sample_size``); this implementation uses the correct formula.

        Parameters:
            t: The observed t-statistic.
            n1: Sample size of group 1 (n₁ ≥ 2).
            n2: Sample size of group 2 (n₂ ≥ 2).

        Returns:
            :class:`StatisticConversionResult` with Hedges' g.
        """
        _validate_n(n1, min_n=2, label="n1")
        _validate_n(n2, min_n=2, label="n2")
        df = n1 + n2 - 2
        d = t * math.sqrt(1.0 / n1 + 1.0 / n2)
        j = _bias_correction(df)
        g = j * d
        return StatisticConversionResult(
            cohens_d=round(d, 6),
            hedges_g=round(g, 6),
            input_statistic="t",
            input_value=float(t),
            design="independent",
            metadata={"n1": n1, "n2": n2, "df": df, "bias_correction_j": round(j, 6)},
        )


# ---------------------------------------------------------------------------
# Internal validator
# ---------------------------------------------------------------------------


def _validate_n(n: int, min_n: int = 2, label: str = "n") -> None:
    if not isinstance(n, (int, np.integer)) or n < min_n:
        raise ValueError(f"{label} must be an integer ≥ {min_n} (got {n!r}).")
