"""Effect-size conversion functions for the *r* (correlation) family.

All functions return :class:`~esek.results.base.ConversionResult`.
"""

from __future__ import annotations

import math

from ..core.exceptions import InvalidInputError
from ..core.validation import validate_sample_size
from ..results.base import ConversionResult


def r_to_d(r: float, n1: int, n2: int) -> ConversionResult:
    """Convert Pearson *r* to Cohen's *d*.

    Inverse of :func:`~esek.converters.d_conversions.d_to_r`:

        d = 2r / √(1 − r²) · √((n₁+n₂)²/(4·n₁·n₂))

    Parameters
    ----------
    r:
        Pearson correlation (−1 ≤ r ≤ 1, but |r| must be < 1).
    n1, n2:
        Sample sizes of the two groups.

    Returns
    -------
    ConversionResult

    Raises
    ------
    InvalidInputError
        If |r| >= 1 (undefined conversion) or n1/n2 are invalid.
    """
    validate_sample_size(n1, "n1")
    validate_sample_size(n2, "n2")
    if abs(r) >= 1.0:
        raise InvalidInputError(
            f"r must satisfy |r| < 1 for this conversion, got r = {r}."
        )
    # Factor that accounts for unequal group sizes
    size_factor = math.sqrt((n1 + n2) ** 2 / (4.0 * n1 * n2))
    d = (2.0 * r / math.sqrt(1.0 - r**2)) * size_factor
    return ConversionResult(
        input_type="r",
        output_type="d",
        input_value=float(r),
        output_value=d,
        method="r_to_d unequal-n formula",
        metadata={"n1": n1, "n2": n2},
    )


def r_to_d_equal_n(r: float) -> ConversionResult:
    """Convert Pearson *r* to Cohen's *d* assuming equal group sizes.

        d = 2r / √(1 − r²)

    Parameters
    ----------
    r:
        Pearson correlation (|r| must be < 1).

    Returns
    -------
    ConversionResult
    """
    if abs(r) >= 1.0:
        raise InvalidInputError(
            f"r must satisfy |r| < 1 for this conversion, got r = {r}."
        )
    d = 2.0 * r / math.sqrt(1.0 - r**2)
    return ConversionResult(
        input_type="r",
        output_type="d",
        input_value=float(r),
        output_value=d,
        method="r_to_d equal-n approximation",
    )


def r_to_fisher_z(r: float) -> ConversionResult:
    """Apply Fisher's *z′* transformation to Pearson *r*.

        z′ = 0.5 · ln((1+r)/(1−r)) = atanh(r)

    This transformation stabilises the variance for CI construction
    (variance of z′ ≈ 1/(n−3)).

    Parameters
    ----------
    r:
        Pearson correlation (|r| must be < 1).

    Returns
    -------
    ConversionResult
    """
    if abs(r) >= 1.0:
        raise InvalidInputError(
            f"r must satisfy |r| < 1 for Fisher z, got r = {r}."
        )
    z = math.atanh(r)
    return ConversionResult(
        input_type="r",
        output_type="Fisher z",
        input_value=float(r),
        output_value=z,
        method="Fisher (1921) z-transformation",
    )


def fisher_z_to_r(z: float) -> ConversionResult:
    """Inverse Fisher *z′* transformation back to Pearson *r*.

        r = tanh(z′)

    Parameters
    ----------
    z:
        Fisher z′-transformed correlation.

    Returns
    -------
    ConversionResult
    """
    r = math.tanh(z)
    return ConversionResult(
        input_type="Fisher z",
        output_type="r",
        input_value=float(z),
        output_value=r,
        method="inverse Fisher z-transformation (tanh)",
    )
