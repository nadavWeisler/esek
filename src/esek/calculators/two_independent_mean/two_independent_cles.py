"""Parametric common-language effect sizes for two independent means.

This module provides a typed migration of the legacy two-independent CLES
calculator. It focuses on the parametric common-language effect sizes derived
from three standardized mean differences:

- Cohen's d_s
- Hedges' g_s
- Cohen's d_pop

Each common-language measure is returned together with central and pivotal
confidence intervals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from scipy.stats import norm, t

from ...core import InvalidInputError
from ...core.validation import (
    validate_confidence_level,
    validate_not_nan,
    validate_proportion,
    validate_sample_size,
)
from ...utils.distribution_helpers import pivotal_ci_t


@dataclass(frozen=True)
class ConfidenceInterval:
    """A simple immutable confidence interval."""

    lower: float
    upper: float


@dataclass(frozen=True)
class CLESMeasureResult:
    """A parametric common-language effect size with two CI types."""

    value: float
    central_ci: ConfidenceInterval
    pivotal_ci: ConfidenceInterval


@dataclass(frozen=True)
class StandardizerCLESResult:
    """CLES results for a single standardized mean difference."""

    effect_size: float
    standard_error: float
    central_effect_ci: ConfidenceInterval
    pivotal_effect_ci: ConfidenceInterval
    u1: CLESMeasureResult
    u2: CLESMeasureResult
    u3: CLESMeasureResult
    cl: CLESMeasureResult
    pov: CLESMeasureResult


@dataclass(frozen=True)
class TwoIndependentCLESResult:
    """Complete result for the two-independent parametric CLES calculator."""

    t_score: float
    p_value: float
    degrees_of_freedom: int
    n1: int
    n2: int
    confidence_level: float
    cohens_ds: StandardizerCLESResult
    hedges_gs: StandardizerCLESResult
    cohens_dpop: StandardizerCLESResult


class TwoIndependentCLES:
    """Calculate parametric CLES measures for two independent means."""

    @staticmethod
    def from_t_score(
        t_score: float,
        n1: int,
        n2: int,
        confidence_level: float = 0.95,
    ) -> TwoIndependentCLESResult:
        """Create parametric CLES results from an independent-samples t-statistic.

        Parameters
        ----------
        t_score:
            Observed independent-samples t-statistic.
        n1, n2:
            Sample sizes for the two groups.
        confidence_level:
            Confidence level in ``(0, 1)``.
        """
        validate_not_nan(t_score, name="t_score")
        validate_sample_size(n1, name="n1")
        validate_sample_size(n2, name="n2")
        validate_confidence_level(confidence_level)

        if n1 < 2 or n2 < 2:
            raise InvalidInputError("'n1' and 'n2' must each be at least 2.")

        df = n1 + n2 - 2
        if df <= 2:
            raise InvalidInputError(
                "'n1 + n2' must be greater than 4 for central effect-size intervals."
            )

        cl = float(confidence_level)
        total_n = n1 + n2
        t_value = float(t_score)
        p_value = min(float(t.sf(abs(t_value), df) * 2.0), 0.99999)

        ds = t_value * math.sqrt((1.0 / n1) + (1.0 / n2))
        correction = _hedges_correction(df)
        gs = ds * correction
        dpop = ds / math.sqrt(df / total_n)

        ds_central_low, ds_central_high, ds_se = _central_ci_from_d(ds, n1, n2, cl)
        gs_central_low, gs_central_high, gs_se = _central_ci_from_d(gs, n1, n2, cl)
        dpop_central_low, dpop_central_high, dpop_se = _central_ci_from_d(
            dpop, n1, n2, cl
        )

        scale = math.sqrt(total_n / (n1 * n2))
        ds_pivotal_ncp_low, ds_pivotal_ncp_high = pivotal_ci_t(t_value, df, total_n, cl)
        ds_pivotal_low = ds_pivotal_ncp_low * scale
        ds_pivotal_high = ds_pivotal_ncp_high * scale
        gs_pivotal_low = ds_pivotal_low * correction
        gs_pivotal_high = ds_pivotal_high * correction

        t_score_dpop = dpop / math.sqrt((1.0 / n1) + (1.0 / n2))
        dpop_pivotal_ncp_low, dpop_pivotal_ncp_high = pivotal_ci_t(
            t_score_dpop,
            df,
            total_n,
            cl,
        )
        dpop_pivotal_low = dpop_pivotal_ncp_low * scale
        dpop_pivotal_high = dpop_pivotal_ncp_high * scale

        return TwoIndependentCLESResult(
            t_score=t_value,
            p_value=p_value,
            degrees_of_freedom=df,
            n1=n1,
            n2=n2,
            confidence_level=cl,
            cohens_ds=_build_standardizer_result(
                effect_size=ds,
                standard_error=ds_se,
                central_effect_ci=(ds_central_low, ds_central_high),
                pivotal_effect_ci=(ds_pivotal_low, ds_pivotal_high),
            ),
            hedges_gs=_build_standardizer_result(
                effect_size=gs,
                standard_error=gs_se,
                central_effect_ci=(gs_central_low, gs_central_high),
                pivotal_effect_ci=(gs_pivotal_low, gs_pivotal_high),
            ),
            cohens_dpop=_build_standardizer_result(
                effect_size=dpop,
                standard_error=dpop_se,
                central_effect_ci=(dpop_central_low, dpop_central_high),
                pivotal_effect_ci=(dpop_pivotal_low, dpop_pivotal_high),
            ),
        )


def _build_standardizer_result(
    *,
    effect_size: float,
    standard_error: float,
    central_effect_ci: tuple[float, float],
    pivotal_effect_ci: tuple[float, float],
) -> StandardizerCLESResult:
    """Build all five parametric CLES measures for one standardizer."""
    central_interval = ConfidenceInterval(*central_effect_ci)
    pivotal_interval = ConfidenceInterval(*pivotal_effect_ci)

    return StandardizerCLESResult(
        effect_size=effect_size,
        standard_error=standard_error,
        central_effect_ci=central_interval,
        pivotal_effect_ci=pivotal_interval,
        u1=_build_abs_measure(effect_size, central_effect_ci, pivotal_effect_ci, _u1_from_abs_d),
        u2=_build_abs_measure(effect_size, central_effect_ci, pivotal_effect_ci, _u2_from_abs_d),
        u3=_build_abs_measure(effect_size, central_effect_ci, pivotal_effect_ci, _u3_from_abs_d),
        cl=_build_signed_measure(effect_size, central_effect_ci, pivotal_effect_ci, _cl_from_d),
        pov=_build_abs_measure(
            effect_size,
            central_effect_ci,
            pivotal_effect_ci,
            _pov_from_abs_d,
            decreasing=True,
        ),
    )


def _build_abs_measure(
    effect_size: float,
    central_effect_ci: tuple[float, float],
    pivotal_effect_ci: tuple[float, float],
    transform: Callable[[float], float],
    *,
    decreasing: bool = False,
) -> CLESMeasureResult:
    """Transform an absolute-d based measure and its confidence intervals."""
    value = transform(abs(effect_size))
    central_low_abs, central_high_abs = _absolute_ci_bounds(*central_effect_ci)
    pivotal_low_abs, pivotal_high_abs = _absolute_ci_bounds(*pivotal_effect_ci)

    if decreasing:
        central_ci = _proportion_interval(
            transform(central_high_abs),
            transform(central_low_abs),
        )
        pivotal_ci = _proportion_interval(
            transform(pivotal_high_abs),
            transform(pivotal_low_abs),
        )
    else:
        central_ci = _proportion_interval(
            transform(central_low_abs),
            transform(central_high_abs),
        )
        pivotal_ci = _proportion_interval(
            transform(pivotal_low_abs),
            transform(pivotal_high_abs),
        )

    return CLESMeasureResult(value=value, central_ci=central_ci, pivotal_ci=pivotal_ci)


def _build_signed_measure(
    effect_size: float,
    central_effect_ci: tuple[float, float],
    pivotal_effect_ci: tuple[float, float],
    transform: Callable[[float], float],
) -> CLESMeasureResult:
    """Transform a signed-d based measure and its confidence intervals."""
    return CLESMeasureResult(
        value=transform(effect_size),
        central_ci=_proportion_interval(
            transform(central_effect_ci[0]),
            transform(central_effect_ci[1]),
        ),
        pivotal_ci=_proportion_interval(
            transform(pivotal_effect_ci[0]),
            transform(pivotal_effect_ci[1]),
        ),
    )


def _central_ci_from_d(
    effect_size: float,
    n1: int,
    n2: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    """Central CI for a two-independent standardized mean difference."""
    df = n1 + n2 - 2
    if df <= 2:
        raise InvalidInputError("Degrees of freedom must be greater than 2.")

    inverse_n_sum = (1.0 / n1) + (1.0 / n2)
    correction_factor = _hedges_correction(df)
    variance = (
        (df / (df - 2.0))
        * inverse_n_sum
        * (1.0 + (effect_size**2) / (2.0 * inverse_n_sum))
        - ((effect_size**2) / (correction_factor**2))
    )
    if variance < 0.0:
        variance = 0.0

    standard_error = math.sqrt(variance)
    z_critical = float(norm.ppf(confidence_level + ((1.0 - confidence_level) / 2.0)))
    return (
        effect_size - standard_error * z_critical,
        effect_size + standard_error * z_critical,
        standard_error,
    )


def _hedges_correction(df: int) -> float:
    """Return Hedges' small-sample correction factor."""
    return math.exp(
        math.lgamma(df / 2.0)
        - math.log(math.sqrt(df / 2.0))
        - math.lgamma((df - 1.0) / 2.0)
    )


def _absolute_ci_bounds(lower: float, upper: float) -> tuple[float, float]:
    """Return the minimum and maximum absolute value inside an interval."""
    low = float(lower)
    high = float(upper)
    if low > high:
        low, high = high, low
    min_abs = 0.0 if low <= 0.0 <= high else min(abs(low), abs(high))
    max_abs = max(abs(low), abs(high))
    return min_abs, max_abs


def _proportion_interval(lower: float, upper: float) -> ConfidenceInterval:
    """Create a validated probability interval."""
    validate_proportion(lower, name="lower")
    validate_proportion(upper, name="upper")
    return ConfidenceInterval(lower=float(lower), upper=float(upper))


def _u1_from_abs_d(abs_d: float) -> float:
    """Cohen's U1 from |d|."""
    u2 = _u2_from_abs_d(abs_d)
    if u2 == 0.0:
        return 0.0
    value = (2.0 * u2 - 1.0) / u2
    return float(max(0.0, min(1.0, value)))


def _u2_from_abs_d(abs_d: float) -> float:
    """Cohen's U2 from |d|."""
    return float(norm.cdf(abs_d / 2.0))


def _u3_from_abs_d(abs_d: float) -> float:
    """Cohen's U3 from |d|."""
    return float(norm.cdf(abs_d))


def _cl_from_d(d_value: float) -> float:
    """McGraw-Wong common-language effect size from signed d."""
    return float(norm.cdf(d_value / math.sqrt(2.0)))


def _pov_from_abs_d(abs_d: float) -> float:
    """Proportion of overlap from |d|."""
    return float(2.0 * norm.cdf(-(abs_d / 2.0)))
