"""Effect-size conversion functions for the odds-ratio family.

All functions return :class:`~esek.results.base.ConversionResult`.
"""

from __future__ import annotations

import math

from ..core.exceptions import InvalidInputError
from ..results.base import ConversionResult


def odds_ratio_to_d(or_: float) -> ConversionResult:
    """Convert an odds ratio (OR) to Cohen's *d*.

    Inverse of the logistic approximation:

        d = ln(OR) · √3 / π

    Parameters
    ----------
    or_:
        Odds ratio (must be > 0).

    Returns
    -------
    ConversionResult

    Raises
    ------
    InvalidInputError
        If *or_* is not strictly positive.
    """
    if or_ <= 0.0:
        raise InvalidInputError(f"Odds ratio must be > 0, got {or_}.")
    log_or = math.log(or_)
    d = log_or * math.sqrt(3.0) / math.pi
    return ConversionResult(
        input_type="OR",
        output_type="d",
        input_value=float(or_),
        output_value=d,
        method="logistic approximation (Cox 1970)",
        metadata={"log_OR": log_or},
    )


def log_odds_ratio_to_d(log_or: float) -> ConversionResult:
    """Convert a log odds ratio (log-OR) to Cohen's *d*.

    Parameters
    ----------
    log_or:
        Natural log of the odds ratio.

    Returns
    -------
    ConversionResult
    """
    d = log_or * math.sqrt(3.0) / math.pi
    return ConversionResult(
        input_type="log-OR",
        output_type="d",
        input_value=float(log_or),
        output_value=d,
        method="logistic approximation (Cox 1970)",
        metadata={"OR": math.exp(log_or)},
    )


def odds_ratio_to_r(or_: float, n1: int, n2: int) -> ConversionResult:
    """Convert an odds ratio to Pearson *r* via *d*.

    First converts OR → d (logistic approximation), then d → r.

    Parameters
    ----------
    or_:
        Odds ratio (must be > 0).
    n1, n2:
        Sample sizes used for the d → r conversion.

    Returns
    -------
    ConversionResult
    """
    from ..converters.d_conversions import d_to_r  # avoid circular at module level

    if or_ <= 0.0:
        raise InvalidInputError(f"Odds ratio must be > 0, got {or_}.")
    d_result = odds_ratio_to_d(or_)
    r_result = d_to_r(d_result.output_value, n1, n2)
    return ConversionResult(
        input_type="OR",
        output_type="r",
        input_value=float(or_),
        output_value=r_result.output_value,
        method="OR → d (logistic approx.) → r",
        metadata={"d": d_result.output_value, "n1": n1, "n2": n2},
    )
