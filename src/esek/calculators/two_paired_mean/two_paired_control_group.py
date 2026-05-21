"""Pre-post control-group effect sizes for paired mean designs.

This module migrates the legacy pre-post control-group calculator. The effect
size is standardized by the pre-test standard deviation, following the approach
summarized by Lipsey and Wilson (2001).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import norm, t

from ...core import InvalidInputError, StatisticalComputationError
from ...core.validation import (
    validate_confidence_level,
    validate_groups_equal_length,
    validate_non_empty,
    validate_not_nan,
    validate_sample_size,
    validate_standard_deviation,
)
from ...utils.distribution_helpers import pivotal_ci_t


@dataclass(frozen=True)
class ConfidenceInterval:
    """A simple immutable confidence interval."""

    lower: float
    upper: float


@dataclass(frozen=True)
class EffectSizeEstimate:
    """A standardized mean difference with central and pivotal intervals."""

    value: float
    standard_error: float
    central_ci: ConfidenceInterval
    pivotal_ci: ConfidenceInterval


@dataclass(frozen=True)
class PrePostControlGroupResult:
    """Result for a pre-post control-group effect-size analysis."""

    mean_pre: float
    mean_post: float
    sd_pre: float
    sd_post: float
    correlation: float
    n: int
    confidence_level: float
    population_difference: float
    mean_difference: float
    difference_standard_error: float
    t_score: float
    degrees_of_freedom: int
    p_value: float
    correction_factor: float
    cohens_d: EffectSizeEstimate
    hedges_g: EffectSizeEstimate


class TwoPairedControlGroup:
    """Calculate pre-post control-group effect sizes."""

    @staticmethod
    def from_parameters(
        mean_pre: float,
        mean_post: float,
        sd_pre: float,
        sd_post: float,
        r: float,
        n: int,
        confidence_level: float,
        pop_diff: float = 0.0,
    ) -> PrePostControlGroupResult:
        """Create results from summary statistics.

        Parameters
        ----------
        mean_pre, mean_post:
            Pre-test and post-test means.
        sd_pre, sd_post:
            Pre-test and post-test standard deviations.
        r:
            Correlation between the pre and post measurements.
        n:
            Number of paired observations.
        confidence_level:
            Confidence level in ``(0, 1)``.
        pop_diff:
            Reference population difference.
        """
        validate_not_nan(mean_pre, name="mean_pre")
        validate_not_nan(mean_post, name="mean_post")
        validate_not_nan(pop_diff, name="pop_diff")
        validate_standard_deviation(sd_pre, name="sd_pre")
        validate_standard_deviation(sd_post, name="sd_post")
        validate_sample_size(n, name="n")
        validate_confidence_level(confidence_level)
        _validate_correlation(r)

        if n < 4:
            raise InvalidInputError(
                "'n' must be at least 4 for the requested inferential and CI formulas."
            )

        mean_difference = float(mean_post) - float(mean_pre)
        adjusted_difference = mean_difference - float(pop_diff)
        difference_sd = _paired_difference_sd(float(sd_pre), float(sd_post), float(r))
        difference_se = difference_sd / math.sqrt(n)
        if difference_se == 0.0:
            raise StatisticalComputationError(
                "The paired-difference standard error is zero; the t-statistic is undefined."
            )

        df = n - 2
        t_score = adjusted_difference / difference_se
        p_value = min(float(t.sf(abs(t_score), df) * 2.0), 0.99999)

        cohens_d_value = adjusted_difference / float(sd_pre)
        df_effect = n - 1
        correction = _hedges_correction(df_effect)
        hedges_g_value = cohens_d_value * correction

        t_score_pre = adjusted_difference / (float(sd_pre) / math.sqrt(n))
        d_pivotal_low, d_pivotal_high = pivotal_ci_t(
            t_score_pre,
            df_effect,
            n,
            float(confidence_level),
        )
        d_central_low, d_central_high, d_standard_error = _central_ci_from_d(
            cohens_d_value,
            n,
            float(confidence_level),
        )
        g_central_low, g_central_high, g_standard_error = _central_ci_from_d(
            hedges_g_value,
            n,
            float(confidence_level),
        )

        return PrePostControlGroupResult(
            mean_pre=float(mean_pre),
            mean_post=float(mean_post),
            sd_pre=float(sd_pre),
            sd_post=float(sd_post),
            correlation=float(r),
            n=n,
            confidence_level=float(confidence_level),
            population_difference=float(pop_diff),
            mean_difference=mean_difference,
            difference_standard_error=difference_sd,
            t_score=t_score,
            degrees_of_freedom=df,
            p_value=p_value,
            correction_factor=correction,
            cohens_d=EffectSizeEstimate(
                value=cohens_d_value,
                standard_error=d_standard_error,
                central_ci=ConfidenceInterval(d_central_low, d_central_high),
                pivotal_ci=ConfidenceInterval(d_pivotal_low, d_pivotal_high),
            ),
            hedges_g=EffectSizeEstimate(
                value=hedges_g_value,
                standard_error=g_standard_error,
                central_ci=ConfidenceInterval(g_central_low, g_central_high),
                pivotal_ci=ConfidenceInterval(
                    d_pivotal_low * correction,
                    d_pivotal_high * correction,
                ),
            ),
        )

    @staticmethod
    def from_data(
        control_data: Sequence[float],
        experimental_data: Sequence[float],
        confidence_level: float,
        pop_diff: float = 0.0,
    ) -> PrePostControlGroupResult:
        """Create results from paired raw data.

        Parameters
        ----------
        control_data:
            Baseline / pre-test scores.
        experimental_data:
            Follow-up / post-test scores for the same units.
        confidence_level:
            Confidence level in ``(0, 1)``.
        pop_diff:
            Reference population difference.
        """
        validate_non_empty(control_data, name="control_data")
        validate_non_empty(experimental_data, name="experimental_data")
        validate_groups_equal_length(
            control_data,
            experimental_data,
            name1="control_data",
            name2="experimental_data",
        )

        pre = np.asarray(control_data, dtype=float)
        post = np.asarray(experimental_data, dtype=float)
        n = int(pre.size)
        validate_sample_size(n, name="n")

        mean_pre = float(np.mean(pre))
        mean_post = float(np.mean(post))
        sd_pre = float(np.std(pre, ddof=1))
        sd_post = float(np.std(post, ddof=1))
        correlation = float(np.corrcoef(pre, post)[0, 1])

        return TwoPairedControlGroup.from_parameters(
            mean_pre=mean_pre,
            mean_post=mean_post,
            sd_pre=sd_pre,
            sd_post=sd_post,
            r=correlation,
            n=n,
            confidence_level=confidence_level,
            pop_diff=pop_diff,
        )


def _validate_correlation(r: float) -> None:
    """Validate a correlation coefficient."""
    validate_not_nan(r, name="r")
    r_value = float(r)
    if not -1.0 <= r_value <= 1.0:
        raise InvalidInputError(f"'r' must be in [-1, 1], got {r_value}.")


def _paired_difference_sd(sd_pre: float, sd_post: float, correlation: float) -> float:
    """Return the standard deviation of the paired difference scores."""
    variance = (sd_pre**2) + (sd_post**2) - (2.0 * correlation * sd_pre * sd_post)
    if variance < 0.0:
        if variance > -1e-12:
            variance = 0.0
        else:
            raise StatisticalComputationError(
                "Computed a negative paired-difference variance."
            )
    return math.sqrt(variance)


def _central_ci_from_d(
    effect_size: float,
    sample_size: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    """Central CI for one-sample / paired-design Cohen's d."""
    df = sample_size - 1
    if df <= 2:
        raise InvalidInputError("Degrees of freedom must be greater than 2.")

    correction_factor = _hedges_correction(df)
    variance = (
        (df / (df - 2.0)) * (1.0 / sample_size) * (1.0 + (effect_size**2) * sample_size)
        - ((effect_size**2) / (correction_factor**2))
    )
    if variance < 0.0:
        if variance > -1e-12:
            variance = 0.0
        else:
            raise StatisticalComputationError(
                "Computed a negative variance while building the central CI."
            )

    
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
