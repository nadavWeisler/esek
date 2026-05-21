"""Effect-size conversion functions for the *d* family.

All public functions return a :class:`~esek.results.base.ConversionResult`
so callers always receive a typed, immutable result object rather than a
raw scalar.

Formulas follow:
- Cohen (1988) *Statistical Power Analysis for the Behavioral Sciences*
- Borenstein et al. (2009) *Introduction to Meta-Analysis*
- Lakens (2013) *Calculating and reporting effect sizes...*
"""

from __future__ import annotations

import math

from ..core.exceptions import InvalidInputError
from ..core.validation import validate_positive, validate_sample_size
from ..results.base import ConversionResult


def d_to_r(d: float, n1: int, n2: int) -> ConversionResult:
    """Convert Cohen's *d* to Pearson *r* (point-biserial).

    Uses the formula:

        r = d / √(d² + (n₁+n₂)²/(n₁·n₂))

    which is exact when group sizes are known.

    Parameters
    ----------
    d:
        Cohen's d effect size.
    n1, n2:
        Sample sizes of the two groups.

    Returns
    -------
    ConversionResult
        ``input_type="d"``, ``output_type="r"``.

    Raises
    ------
    InvalidInputError
        If *n1* or *n2* are not positive integers.
    """
    validate_sample_size(n1, "n1")
    validate_sample_size(n2, "n2")
    correction = (n1 + n2) ** 2 / (n1 * n2)
    r = d / math.sqrt(d**2 + correction)
    return ConversionResult(
        input_type="d",
        output_type="r",
        input_value=float(d),
        output_value=r,
        method="Cohen (1988) exact formula",
        metadata={"n1": n1, "n2": n2, "correction_factor": correction},
    )


def d_to_r_equal_n(d: float, n: int) -> ConversionResult:
    """Convert Cohen's *d* to Pearson *r* assuming equal group sizes.

    Simplified formula for n₁ = n₂ = n:

        r = d / √(d² + 4)

    Parameters
    ----------
    d:
        Cohen's d effect size.
    n:
        Size of each group (both groups must have the same size).

    Returns
    -------
    ConversionResult
    """
    validate_sample_size(n, "n")
    r = d / math.sqrt(d**2 + 4.0)
    return ConversionResult(
        input_type="d",
        output_type="r",
        input_value=float(d),
        output_value=r,
        method="d_to_r equal-n approximation",
        metadata={"n": n},
    )


def d_to_odds_ratio(d: float) -> ConversionResult:
    """Convert Cohen's *d* to odds ratio (OR) via the logistic approximation.

    Formula (Cox 1970; Sanchez-Meca et al. 2003):

        log(OR) = d · π / √3

    Parameters
    ----------
    d:
        Cohen's d effect size.

    Returns
    -------
    ConversionResult
        ``output_value`` is the OR (not log-OR).
    """
    log_or = d * math.pi / math.sqrt(3.0)
    or_ = math.exp(log_or)
    return ConversionResult(
        input_type="d",
        output_type="OR",
        input_value=float(d),
        output_value=or_,
        method="logistic approximation (Cox 1970)",
        metadata={"log_OR": log_or},
    )


def d_to_cohens_f(d: float) -> ConversionResult:
    """Convert Cohen's *d* to Cohen's *f* (ANOVA effect size).

    For the two-group case:

        f = d / 2

    Parameters
    ----------
    d:
        Cohen's d effect size.

    Returns
    -------
    ConversionResult
    """
    f = d / 2.0
    return ConversionResult(
        input_type="d",
        output_type="f",
        input_value=float(d),
        output_value=f,
        method="d/2 (two-group ANOVA)",
    )


def d_to_r_squared(d: float, n1: int, n2: int) -> ConversionResult:
    """Convert Cohen's *d* to coefficient of determination *r²*.

    Parameters
    ----------
    d, n1, n2:
        See :func:`d_to_r`.

    Returns
    -------
    ConversionResult
    """
    r_result = d_to_r(d, n1, n2)
    r2 = r_result.output_value**2
    return ConversionResult(
        input_type="d",
        output_type="r²",
        input_value=float(d),
        output_value=r2,
        method="r² = (d_to_r)²",
        metadata={"r": r_result.output_value, "n1": n1, "n2": n2},
    )


def d_to_eta_squared(d: float, n1: int, n2: int) -> ConversionResult:
    """Convert Cohen's *d* to eta-squared (η²) for the two-group case.

    Uses η² = r² (they are equivalent for the two-group contrast).

    Parameters
    ----------
    d, n1, n2:
        See :func:`d_to_r`.

    Returns
    -------
    ConversionResult
    """
    r_result = d_to_r(d, n1, n2)
    eta2 = r_result.output_value**2
    return ConversionResult(
        input_type="d",
        output_type="η²",
        input_value=float(d),
        output_value=eta2,
        method="η² = r² (two-group equivalence)",
        metadata={"r": r_result.output_value, "n1": n1, "n2": n2},
    )
