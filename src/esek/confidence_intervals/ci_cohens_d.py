"""Confidence intervals for Cohen's d.

Implements the non-R helpers from the legacy Cohen's d CI source file using a
clean, typed API. The public methods cover one-sample, paired, and independent
standardised mean differences using central, pivotal, and non-central-t based
approaches.

References
----------
- Hedges & Olkin (1985) *Statistical Methods for Meta-Analysis*
- Morris (2000) on paired-sample standardised mean differences
- Algina & Keselman (2003) on paired-sample noncentral-t intervals
- Goulet-Pelletier & Cousineau (2018, 2021) on repeated-measures effect sizes
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import gmean

from esek.utils.distribution_helpers import qlambdap as _qlambdap


@dataclass(frozen=True)
class CohensDCIResult:
    """Typed result for a Cohen's d confidence interval."""

    d: float
    ci_low: float
    ci_high: float
    se: float | None
    method: str
    design: str
    confidence_level: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_confidence_level(confidence_level: float) -> None:
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be in (0, 1) (got {confidence_level})."
        )


def _validate_sample_size(n: int, *, minimum: int = 2, name: str = "n") -> None:
    if n < minimum:
        raise ValueError(f"{name} must be >= {minimum} (got {n}).")


def _validate_two_sample_sizes(n1: int, n2: int, *, minimum: int = 2) -> None:
    _validate_sample_size(n1, minimum=minimum, name="n1")
    _validate_sample_size(n2, minimum=minimum, name="n2")


def _validate_correlation(r: float) -> None:
    if not (-1.0 < r < 1.0):
        raise ValueError(f"r must be in (-1, 1) (got {r}).")


def _validate_sd(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0 (got {value}).")


def _sqrt_nonnegative(value: float, label: str) -> float:
    if value < -1e-12:
        raise ValueError(f"{label} became negative ({value}).")
    return math.sqrt(max(value, 0.0))


def _bias_correction(df: float) -> float:
    return math.exp(
        math.lgamma(df / 2.0)
        - math.log(math.sqrt(df / 2.0))
        - math.lgamma((df - 1.0) / 2.0)
    )


def _z_critical(confidence_level: float) -> float:
    return float(stats.norm.ppf(confidence_level + (1.0 - confidence_level) / 2.0))


def _build_result(
    *,
    d: float,
    ci_low: float,
    ci_high: float,
    se: float | None,
    method: str,
    design: str,
    confidence_level: float,
    metadata: dict[str, Any] | None = None,
) -> CohensDCIResult:
    return CohensDCIResult(
        d=float(d),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        se=None if se is None else float(se),
        method=method,
        design=design,
        confidence_level=float(confidence_level),
        metadata={} if metadata is None else metadata,
    )


def _multiple_se_results(
    *,
    d: float,
    confidence_level: float,
    design: str,
    df: int,
    se_values: dict[str, float],
    metadata: dict[str, Any] | None = None,
) -> list[CohensDCIResult]:
    z_crit = _z_critical(confidence_level)
    base_metadata = {} if metadata is None else dict(metadata)
    return [
        _build_result(
            d=d,
            ci_low=d - z_crit * se,
            ci_high=d + z_crit * se,
            se=se,
            method=f"central_{name}",
            design=design,
            confidence_level=confidence_level,
            metadata={**base_metadata, "df": df, "se_method": name},
        )
        for name, se in se_values.items()
    ]


def _calculate_central_ci_from_cohens_d_one_sample(
    cohens_d: float,
    sample_size: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    standard_error_es = math.sqrt(
        (1.0 / sample_size) + ((cohens_d**2) / (2.0 * sample_size))
    )
    z_critical_value = _z_critical(confidence_level)
    ci_lower = cohens_d - standard_error_es * z_critical_value
    ci_upper = cohens_d + standard_error_es * z_critical_value
    return ci_lower, ci_upper, standard_error_es


def _nct_ci_t(
    t_score: float,
    df: int,
    confidence_level: float,
    *,
    tolerance: float = 1e-5,
) -> tuple[float, float]:
    """Faithful NCP bisection from the legacy source.

    Returns the lower and upper non-centrality parameters for an observed t.
    """

    is_negative = t_score < 0
    observed_t = abs(float(t_score))
    step = observed_t if observed_t > 0 else 1.0

    upper_limit = 1.0 - (1.0 - confidence_level) / 2.0
    lower_limit = (1.0 - confidence_level) / 2.0

    lower_criterion = [-step, observed_t / 2.0, step]
    upper_criterion = [step, 2.0 * step, 3.0 * step]

    while stats.nct.cdf(observed_t, df, lower_criterion[0]) < upper_limit:
        lower_criterion = [
            lower_criterion[0] - step,
            lower_criterion[0],
            lower_criterion[2],
        ]

    while stats.nct.cdf(observed_t, df, upper_criterion[0]) < lower_limit:
        if stats.nct.cdf(observed_t, df) < lower_limit:
            upper_criterion = [
                upper_criterion[0] / 4.0,
                upper_criterion[0],
                upper_criterion[2],
            ]
        else:
            break

    while stats.nct.cdf(observed_t, df, upper_criterion[2]) > lower_limit:
        upper_criterion = [
            upper_criterion[0],
            upper_criterion[2],
            upper_criterion[2] + step,
        ]

    lower_ci = 0.0
    diff_lower = 1.0
    while diff_lower > tolerance:
        if stats.nct.cdf(observed_t, df, lower_criterion[1]) < upper_limit:
            lower_criterion = [
                lower_criterion[0],
                (lower_criterion[0] + lower_criterion[1]) / 2.0,
                lower_criterion[1],
            ]
        else:
            lower_criterion = [
                lower_criterion[1],
                (lower_criterion[1] + lower_criterion[2]) / 2.0,
                lower_criterion[2],
            ]
        diff_lower = abs(stats.nct.cdf(observed_t, df, lower_criterion[1]) - upper_limit)
        lower_ci = lower_criterion[1]

    upper_ci = 0.0
    diff_upper = 1.0
    while diff_upper > tolerance:
        if stats.nct.cdf(observed_t, df, upper_criterion[1]) < lower_limit:
            upper_criterion = [
                upper_criterion[0],
                (upper_criterion[0] + upper_criterion[1]) / 2.0,
                upper_criterion[1],
            ]
        else:
            upper_criterion = [
                upper_criterion[1],
                (upper_criterion[1] + upper_criterion[2]) / 2.0,
                upper_criterion[2],
            ]
        diff_upper = abs(stats.nct.cdf(observed_t, df, upper_criterion[1]) - lower_limit)
        upper_ci = upper_criterion[1]

    if is_negative:
        return -upper_ci, -lower_ci
    return lower_ci, upper_ci


def _pivotal_ci_t(
    t_score: float,
    df: int,
    sample_size: int,
    confidence_level: float,
) -> tuple[float, float]:
    lower_ncp, upper_ncp = _nct_ci_t(t_score, df, confidence_level)
    scale = math.sqrt(sample_size)
    return lower_ncp / scale, upper_ncp / scale


def _calculate_central_ci_paired_samples_t_test(
    effect_size: float,
    sample_size: int,
    confidence_level: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    df = sample_size - 1
    correction_factor = _bias_correction(df)
    se_true = _sqrt_nonnegative(
        ((df / (df - 2.0)) * (1.0 / sample_size) * (1.0 + effect_size**2 * sample_size))
        - (effect_size**2 / correction_factor**2),
        "paired true SE variance",
    )
    morris_correction = 1.0 - (3.0 / (4.0 * (df - 1.0) - 1.0))
    se_morris = _sqrt_nonnegative(
        ((df / (df - 2.0)) * (1.0 / sample_size) * (1.0 + effect_size**2 * sample_size))
        - (effect_size**2 / morris_correction**2),
        "paired Morris SE variance",
    )
    se_hedges = math.sqrt((1.0 / sample_size) + effect_size**2 / (2.0 * df))
    se_hedges_olkin = math.sqrt(
        (1.0 / sample_size) + effect_size**2 / (2.0 * sample_size)
    )
    se_mle = math.sqrt(se_hedges * ((df + 2.0) / df))
    se_large_n = math.sqrt((1.0 / sample_size) * (1.0 + effect_size**2 / 8.0))
    se_small_n = math.sqrt(se_large_n * ((df + 1.0) / (df - 1.0)))
    z_critical_value = _z_critical(confidence_level)
    ci_lower = effect_size - se_true * z_critical_value
    ci_upper = effect_size + se_true * z_critical_value
    return (
        ci_lower,
        ci_upper,
        se_true,
        se_morris,
        se_hedges,
        se_hedges_olkin,
        se_mle,
        se_large_n,
        se_small_n,
    )


def _calculate_se_pooled_paired_samples_t_test(
    effect_size: float,
    sample_size: int,
    correlation: float,
    confidence_level: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    df = sample_size - 1
    correction_factor = _bias_correction(df)
    a_value = sample_size / (2.0 * (1.0 - correlation))
    se_true = _sqrt_nonnegative(
        ((df / (df - 2.0)) * (1.0 / a_value) * (1.0 + effect_size**2 * a_value))
        - (effect_size**2 / correction_factor**2),
        "paired pooled true SE variance",
    )
    morris_correction = 1.0 - (3.0 / (4.0 * (df - 1.0) - 1.0))
    se_morris = _sqrt_nonnegative(
        ((df / (df - 2.0)) * (1.0 / a_value) * (1.0 + effect_size**2 * a_value))
        - (effect_size**2 / morris_correction**2),
        "paired pooled Morris SE variance",
    )
    se_hedges = math.sqrt((1.0 / a_value) + effect_size**2 / (2.0 * df))
    se_hedges_olkin = math.sqrt((1.0 / a_value) + effect_size**2 / (2.0 * sample_size))
    se_mle = math.sqrt(se_hedges * ((df + 2.0) / df))
    se_large_n = math.sqrt((1.0 / a_value) * (1.0 + effect_size**2 / 8.0))
    se_small_n = math.sqrt(se_large_n * ((df + 1.0) / (df - 1.0)))
    z_critical_value = _z_critical(confidence_level)
    ci_lower = effect_size - se_true * z_critical_value
    ci_upper = effect_size + se_true * z_critical_value
    return (
        ci_lower,
        ci_upper,
        se_true,
        se_morris,
        se_hedges,
        se_hedges_olkin,
        se_mle,
        se_large_n,
        se_small_n,
    )


def _ci_ncp_one_sample(
    effect_size: float,
    sample_size: int,
    confidence_level: float,
) -> tuple[float, float]:
    ncp_value = effect_size * math.sqrt(sample_size)
    if abs(ncp_value) < 1e-12:
        lower_ncp, upper_ncp = _nct_ci_t(0.0, sample_size - 1, confidence_level)
        scale = math.sqrt(sample_size)
        return lower_ncp / scale, upper_ncp / scale
    ci_ncp_low = (
        stats.nct.ppf(0.5 - confidence_level / 2.0, sample_size - 1, nc=ncp_value)
        / ncp_value
        * effect_size
    )
    ci_ncp_high = (
        stats.nct.ppf(0.5 + confidence_level / 2.0, sample_size - 1, nc=ncp_value)
        / ncp_value
        * effect_size
    )
    return float(ci_ncp_low), float(ci_ncp_high)


def _ci_mag_paired_samples(
    effect_size: float,
    sd1: float,
    sd2: float,
    sample_size: int,
    correlation: float,
    confidence_level: float,
) -> tuple[float, float]:
    corrected_correlation = correlation * (
        gmean([sd1**2, sd2**2]) / np.mean((sd1**2, sd2**2))
    )
    df = sample_size - 1
    correction = _bias_correction(df)
    lambda_value = float(
        effect_size * correction**2 * math.sqrt(sample_size / (2.0 * (1.0 - corrected_correlation)))
    )
    scale = math.sqrt(sample_size / (2.0 * (1.0 - corrected_correlation)))
    lower = stats.nct.ppf(0.5 - confidence_level / 2.0, df=df, nc=lambda_value) / scale
    upper = stats.nct.ppf(0.5 + confidence_level / 2.0, df=df, nc=lambda_value) / scale
    return float(lower), float(upper)


def _ci_morris_paired_samples(
    effect_size: float,
    sample_size: int,
    correlation: float,
    confidence_level: float,
) -> tuple[float, float]:
    df = sample_size - 1
    correction = _bias_correction(df)
    variance = (
        ((df / (df - 2.0)) * 2.0 * (1.0 - correlation) / sample_size)
        * (1.0 + effect_size**2 * sample_size / (2.0 * (1.0 - correlation)))
        - effect_size**2 / correction**2
    ) * correction**2
    z_critical_value = _z_critical(confidence_level)
    margin = _sqrt_nonnegative(variance, "Morris paired variance") * z_critical_value
    return effect_size - margin, effect_size + margin


def _ci_t_algina_keselman(
    effect_size: float,
    sd1: float,
    sd2: float,
    sample_size: int,
    correlation: float,
    confidence_level: float,
) -> tuple[float, float]:
    corrected_correlation = correlation * (
        gmean([sd1**2, sd2**2]) / np.mean((sd1**2, sd2**2))
    )
    constant = math.sqrt(sample_size / (2.0 * (1.0 - corrected_correlation)))
    lower_ncp, upper_ncp = _nct_ci_t(
        effect_size * constant,
        sample_size - 1,
        confidence_level,
    )
    return lower_ncp / constant, upper_ncp / constant


def _calculate_central_ci_from_cohens_d_two_samples(
    cohens_d: float,
    sample_size_1: int,
    sample_size_2: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    standard_error_es = math.sqrt(
        ((sample_size_1 + sample_size_2) / (sample_size_1 * sample_size_2))
        + ((cohens_d**2) / (2.0 * (sample_size_1 + sample_size_2)))
    )
    z_critical_value = _z_critical(confidence_level)
    ci_lower = cohens_d - standard_error_es * z_critical_value
    ci_upper = cohens_d + standard_error_es * z_critical_value
    return ci_lower, ci_upper, standard_error_es


def _calculate_central_ci_from_cohens_d_two_independent_sample_t_test(
    effect_size: float,
    sample_size1: int,
    sample_size2: int,
    confidence_level: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    sample_size = sample_size1 + sample_size2
    df = sample_size - 2
    correction_factor = _bias_correction(df)
    harmonic_sample_size = 2.0 / (1.0 / sample_size1 + 1.0 / sample_size2)
    a_value = harmonic_sample_size / 2.0
    se_true = _sqrt_nonnegative(
        ((df / (df - 2.0)) * (1.0 / a_value) * (1.0 + effect_size**2 * a_value))
        - (effect_size**2 / correction_factor**2),
        "independent true SE variance",
    )
    morris_correction = 1.0 - (3.0 / (4.0 * (df - 1.0) - 1.0))
    se_morris = _sqrt_nonnegative(
        ((df / (df - 2.0)) * (1.0 / a_value) * (1.0 + effect_size**2 * a_value))
        - (effect_size**2 / morris_correction**2),
        "independent Morris SE variance",
    )
    se_hedges = math.sqrt((1.0 / a_value) + effect_size**2 / (2.0 * df))
    se_hedges_olkin = math.sqrt((1.0 / a_value) + effect_size**2 / (2.0 * sample_size))
    se_mle = math.sqrt(
        (2.0 / harmonic_sample_size)
        * ((df + 2.0) / df)
        * (1.0 + (effect_size**2 * a_value / (2.0 * df)))
    )
    se_large_n = math.sqrt((2.0 / harmonic_sample_size) * (1.0 + effect_size**2 / 8.0))
    se_small_n = math.sqrt(
        ((df + 1.0) / (df - 1.0))
        * (2.0 / harmonic_sample_size)
        * (1.0 + effect_size**2 / 8.0)
    )
    z_critical_value = _z_critical(confidence_level)
    ci_lower = effect_size - se_true * z_critical_value
    ci_upper = effect_size + se_true * z_critical_value
    return (
        ci_lower,
        ci_upper,
        se_true,
        se_morris,
        se_hedges,
        se_hedges_olkin,
        se_mle,
        se_large_n,
        se_small_n,
    )


class CohensDCI:
    """Confidence intervals for Cohen's d family effect sizes.

    Each method returns a :class:`CohensDCIResult` or a list of results when the
    legacy source exposed several standard-error-based central intervals.
    """

    @staticmethod
    def one_sample_z(
        d: float,
        n: int,
        confidence_level: float = 0.95,
    ) -> CohensDCIResult:
        """Z-based central CI for one-sample Cohen's d.

        Uses the Hedges & Olkin (1985) large-sample SE formula from the legacy
        source helper ``calculate_central_ci_from_cohens_d_one_sample``.
        """
        _validate_sample_size(n, minimum=2)
        _validate_confidence_level(confidence_level)
        ci_low, ci_high, se = _calculate_central_ci_from_cohens_d_one_sample(d, n, confidence_level)
        return _build_result(
            d=d,
            ci_low=ci_low,
            ci_high=ci_high,
            se=se,
            method="z_central",
            design="one_sample",
            confidence_level=confidence_level,
            metadata={"n": n},
        )

    @staticmethod
    def paired_z(
        d: float,
        n: int,
        confidence_level: float = 0.95,
    ) -> CohensDCIResult:
        """Z-based central CI for paired-sample Cohen's d_z.

        This matches the one-sample large-sample formula in the legacy source,
        where paired d_z is treated as a one-sample effect size on differences.
        """
        _validate_sample_size(n, minimum=2)
        _validate_confidence_level(confidence_level)
        ci_low, ci_high, se = _calculate_central_ci_from_cohens_d_one_sample(d, n, confidence_level)
        return _build_result(
            d=d,
            ci_low=ci_low,
            ci_high=ci_high,
            se=se,
            method="z_central",
            design="paired",
            confidence_level=confidence_level,
            metadata={"n": n},
        )

    @staticmethod
    def independent_z(
        d: float,
        n1: int,
        n2: int,
        confidence_level: float = 0.95,
    ) -> CohensDCIResult:
        """Z-based central CI for independent-samples Cohen's d.

        Uses the Hedges & Olkin (1985) SE approximation from the legacy helper
        ``calculate_central_ci_from_cohens_d_two_samples``.
        """
        _validate_two_sample_sizes(n1, n2, minimum=2)
        _validate_confidence_level(confidence_level)
        ci_low, ci_high, se = _calculate_central_ci_from_cohens_d_two_samples(
            d,
            n1,
            n2,
            confidence_level,
        )
        return _build_result(
            d=d,
            ci_low=ci_low,
            ci_high=ci_high,
            se=se,
            method="z_central",
            design="independent",
            confidence_level=confidence_level,
            metadata={"n1": n1, "n2": n2},
        )

    @staticmethod
    def one_sample_t_central(
        d: float,
        n: int,
        confidence_level: float = 0.95,
    ) -> list[CohensDCIResult]:
        """Central CIs for one-sample d using seven SE estimators.

        Returns seven results in this order: true, Morris, Hedges,
        Hedges-Olkin, MLE, Large-N, Small-N.
        """
        _validate_sample_size(n, minimum=4)
        _validate_confidence_level(confidence_level)
        _, _, se_true, se_morris, se_hedges, se_hedges_olkin, se_mle, se_large_n, se_small_n = (
            _calculate_central_ci_paired_samples_t_test(d, n, confidence_level)
        )
        return _multiple_se_results(
            d=d,
            confidence_level=confidence_level,
            design="one_sample",
            df=n - 1,
            se_values={
                "true": se_true,
                "morris": se_morris,
                "hedges": se_hedges,
                "hedges_olkin": se_hedges_olkin,
                "mle": se_mle,
                "large_n": se_large_n,
                "small_n": se_small_n,
            },
            metadata={"n": n},
        )

    @staticmethod
    def one_sample_t_ncp(
        d: float,
        n: int,
        confidence_level: float = 0.95,
    ) -> CohensDCIResult:
        """Noncentral-t CI for one-sample d.

        Uses the legacy ``CI_NCP_one_Sample`` approach, with a zero-effect
        fallback through the legacy NCP bisection logic.
        """
        _validate_sample_size(n, minimum=2)
        _validate_confidence_level(confidence_level)
        ci_low, ci_high = _ci_ncp_one_sample(d, n, confidence_level)
        return _build_result(
            d=d,
            ci_low=ci_low,
            ci_high=ci_high,
            se=None,
            method="t_ncp",
            design="one_sample",
            confidence_level=confidence_level,
            metadata={"n": n, "df": n - 1, "ncp": d * math.sqrt(n)},
        )

    @staticmethod
    def one_sample_t_pivotal(
        t_stat: float,
        n: int,
        confidence_level: float = 0.95,
    ) -> CohensDCIResult:
        """Pivotal noncentral-t CI for one-sample d.

        Faithfully follows the legacy ``Pivotal_ci_t`` bisection on the NCT CDF.
        The observed point estimate is ``d = t / sqrt(n)``.
        """
        _validate_sample_size(n, minimum=2)
        _validate_confidence_level(confidence_level)
        ci_low, ci_high = _pivotal_ci_t(t_stat, n - 1, n, confidence_level)
        d = t_stat / math.sqrt(n)
        return _build_result(
            d=d,
            ci_low=ci_low,
            ci_high=ci_high,
            se=None,
            method="t_pivotal",
            design="one_sample",
            confidence_level=confidence_level,
            metadata={"n": n, "df": n - 1, "t_stat": t_stat},
        )

    @staticmethod
    def paired_t_central(
        d: float,
        n: int,
        confidence_level: float = 0.95,
    ) -> list[CohensDCIResult]:
        """Central CIs for paired d_z using seven SE estimators.

        Returns seven results in this order: true, Morris, Hedges,
        Hedges-Olkin, MLE, Large-N, Small-N.
        """
        _validate_sample_size(n, minimum=4)
        _validate_confidence_level(confidence_level)
        _, _, se_true, se_morris, se_hedges, se_hedges_olkin, se_mle, se_large_n, se_small_n = (
            _calculate_central_ci_paired_samples_t_test(d, n, confidence_level)
        )
        return _multiple_se_results(
            d=d,
            confidence_level=confidence_level,
            design="paired",
            df=n - 1,
            se_values={
                "true": se_true,
                "morris": se_morris,
                "hedges": se_hedges,
                "hedges_olkin": se_hedges_olkin,
                "mle": se_mle,
                "large_n": se_large_n,
                "small_n": se_small_n,
            },
            metadata={"n": n, "variant": "dz"},
        )

    @staticmethod
    def paired_t_pooled_central(
        d: float,
        n: int,
        r: float,
        confidence_level: float = 0.95,
    ) -> list[CohensDCIResult]:
        """Central CIs for paired pooled-SD d using seven SE estimators.

        Uses the Goulet-Pelletier & Cousineau pooled-SE formulas that depend on
        the within-pair correlation ``r``. Returns the same seven SE variants as
        :meth:`paired_t_central`.
        """
        _validate_sample_size(n, minimum=4)
        _validate_confidence_level(confidence_level)
        _validate_correlation(r)
        _, _, se_true, se_morris, se_hedges, se_hedges_olkin, se_mle, se_large_n, se_small_n = (
            _calculate_se_pooled_paired_samples_t_test(d, n, r, confidence_level)
        )
        return _multiple_se_results(
            d=d,
            confidence_level=confidence_level,
            design="paired",
            df=n - 1,
            se_values={
                "true": se_true,
                "morris": se_morris,
                "hedges": se_hedges,
                "hedges_olkin": se_hedges_olkin,
                "mle": se_mle,
                "large_n": se_large_n,
                "small_n": se_small_n,
            },
            metadata={"n": n, "r": r, "variant": "pooled"},
        )

    @staticmethod
    def paired_t_ncp(
        d: float,
        n: int,
        confidence_level: float = 0.95,
    ) -> CohensDCIResult:
        """Noncentral-t CI for paired d_z.

        In the legacy source the paired d_z NCP interval is identical to the
        one-sample case because the test is performed on paired differences.
        """
        _validate_sample_size(n, minimum=2)
        _validate_confidence_level(confidence_level)
        ci_low, ci_high = _ci_ncp_one_sample(d, n, confidence_level)
        return _build_result(
            d=d,
            ci_low=ci_low,
            ci_high=ci_high,
            se=None,
            method="t_ncp",
            design="paired",
            confidence_level=confidence_level,
            metadata={"n": n, "df": n - 1, "variant": "dz"},
        )

    @staticmethod
    def paired_t_morris(
        d: float,
        n: int,
        r: float,
        confidence_level: float = 0.95,
    ) -> CohensDCIResult:
        """Morris (2000) CI for a paired pooled-SD effect size.

        Assumes a repeated-measures design and requires the correlation between
        paired observations.
        """
        _validate_sample_size(n, minimum=4)
        _validate_confidence_level(confidence_level)
        _validate_correlation(r)
        ci_low, ci_high = _ci_morris_paired_samples(d, n, r, confidence_level)
        return _build_result(
            d=d,
            ci_low=ci_low,
            ci_high=ci_high,
            se=None,
            method="t_morris",
            design="paired",
            confidence_level=confidence_level,
            metadata={"n": n, "df": n - 1, "r": r},
        )

    @staticmethod
    def paired_t_mag(
        d: float,
        sd1: float,
        sd2: float,
        n: int,
        r: float,
        confidence_level: float = 0.95,
    ) -> CohensDCIResult:
        """MAG CI for paired d_av.

        Combines Morris (2000), Algina & Keselman (2003), and
        Goulet-Pelletier & Cousineau logic from the legacy source. This method
        does not require rpy2.
        """
        _validate_sample_size(n, minimum=2)
        _validate_confidence_level(confidence_level)
        _validate_correlation(r)
        _validate_sd(sd1, "sd1")
        _validate_sd(sd2, "sd2")
        ci_low, ci_high = _ci_mag_paired_samples(d, sd1, sd2, n, r, confidence_level)
        corrected_correlation = r * (gmean([sd1**2, sd2**2]) / np.mean((sd1**2, sd2**2)))
        return _build_result(
            d=d,
            ci_low=ci_low,
            ci_high=ci_high,
            se=None,
            method="t_mag",
            design="paired",
            confidence_level=confidence_level,
            metadata={
                "n": n,
                "df": n - 1,
                "r": r,
                "sd1": sd1,
                "sd2": sd2,
                "corrected_correlation": float(corrected_correlation),
            },
        )

    @staticmethod
    def paired_t_algina_keselman(
        d: float,
        sd1: float,
        sd2: float,
        n: int,
        r: float,
        confidence_level: float = 0.95,
    ) -> CohensDCIResult:
        """Algina & Keselman (2003) NCT CI for paired d_av.

        Uses the corrected correlation from the legacy source and the same NCP
        bisection used elsewhere in this module.
        """
        _validate_sample_size(n, minimum=2)
        _validate_confidence_level(confidence_level)
        _validate_correlation(r)
        _validate_sd(sd1, "sd1")
        _validate_sd(sd2, "sd2")
        ci_low, ci_high = _ci_t_algina_keselman(d, sd1, sd2, n, r, confidence_level)
        corrected_correlation = r * (gmean([sd1**2, sd2**2]) / np.mean((sd1**2, sd2**2)))
        return _build_result(
            d=d,
            ci_low=ci_low,
            ci_high=ci_high,
            se=None,
            method="t_algina_keselman",
            design="paired",
            confidence_level=confidence_level,
            metadata={
                "n": n,
                "df": n - 1,
                "r": r,
                "sd1": sd1,
                "sd2": sd2,
                "corrected_correlation": float(corrected_correlation),
            },
        )

    @staticmethod
    def paired_t_lambda_prime(
        d: float,
        sd1: float,
        sd2: float,
        n: int,
        r: float,
        confidence_level: float = 0.95,
    ) -> CohensDCIResult:
        """Adjusted lambda-prime CI for paired d_av (Goulet-Pelletier & Cousineau, 2021).

        Uses Lecoutre's lambda-prime distribution as implemented in the *sadists*
        R package but computed here via pure-Python numerical integration.  The
        method adjusts for the correlation between measurements and applies the
        bias-correction factor for the *original* df to the lambda parameter, and
        the bias-correction factor for the *corrected* df to the scale-back step.

        .. note::
            The original R source in the *dev* branch contained an apparent
            formula bug: the denominator was ``2*(1-r_c) / c2`` rather than
            ``scale * c2`` where ``scale = sqrt(n / (2*(1-r_c)))``.  This
            implementation uses the corrected formula, which produces CIs that
            contain the observed d value and match the intended statistical
            behaviour described in the reference paper.

        Parameters
        ----------
        d:
            Paired Cohen's d_av estimate.
        sd1, sd2:
            Standard deviations of the two conditions.
        n:
            Number of pairs.
        r:
            Pearson correlation between the two conditions.
        confidence_level:
            Desired confidence level (default 0.95).

        Returns
        -------
        CohensDCIResult

        References
        ----------
        - Goulet-Pelletier, J.-C., & Cousineau, D. (2021). A review of
          effect sizes and their confidence intervals, Part I. *The Quantitative
          Methods for Psychology*, 17(1), 51-75.
        - Lecoutre, B. (1999). Two useful distributions for Bayesian predictive
          procedures under normal models. *Journal of Statistical Planning and
          Inference*, 79(1), 93-105.
        """
        _validate_sample_size(n, minimum=3)
        _validate_confidence_level(confidence_level)
        _validate_correlation(r)
        _validate_sd(sd1, "sd1")
        _validate_sd(sd2, "sd2")

        corrected_r = r * (gmean([sd1**2, sd2**2]) / np.mean([sd1**2, sd2**2]))
        df = n - 1
        df_corrected = 2.0 / (1.0 + r**2) * df
        # Bias correction for uncorrected df and corrected df
        c1 = _bias_correction(df)
        c2 = _bias_correction(df_corrected)
        scale = math.sqrt(n / (2.0 * (1.0 - corrected_r)))
        lambda_val = float(d * c1 * scale)
        alpha = 1.0 - confidence_level
        ci_low = float(
            _qlambdap(alpha / 2.0, df=df_corrected, t=lambda_val) / (scale * c2)
        )
        ci_high = float(
            _qlambdap(1.0 - alpha / 2.0, df=df_corrected, t=lambda_val) / (scale * c2)
        )
        return _build_result(
            d=d,
            ci_low=ci_low,
            ci_high=ci_high,
            se=None,
            method="t_lambda_prime",
            design="paired",
            confidence_level=confidence_level,
            metadata={
                "n": n,
                "df": df,
                "df_corrected": df_corrected,
                "r": r,
                "sd1": sd1,
                "sd2": sd2,
                "corrected_correlation": float(corrected_r),
                "lambda_val": lambda_val,
            },
        )

    @staticmethod
    def paired_t_t_prime(
        d: float,
        sd1: float,
        sd2: float,
        n: int,
        r: float,
        confidence_level: float = 0.95,
    ) -> CohensDCIResult:
        """t-prime CI for paired d_av (Goulet-Pelletier & Cousineau, 2021).

        Uses Lecoutre's lambda-prime distribution (sadists' ``qlambdap``) with
        the corrected degrees of freedom.  Unlike :meth:`paired_t_lambda_prime`,
        this method applies only a single bias correction (to the corrected df)
        and scales directly by the t-test constant.

        Parameters
        ----------
        d:
            Paired Cohen's d_av estimate.
        sd1, sd2:
            Standard deviations of the two conditions.
        n:
            Number of pairs.
        r:
            Pearson correlation between the two conditions.
        confidence_level:
            Desired confidence level (default 0.95).

        Returns
        -------
        CohensDCIResult

        References
        ----------
        - Goulet-Pelletier, J.-C., & Cousineau, D. (2021). A review of
          effect sizes and their confidence intervals, Part I. *The Quantitative
          Methods for Psychology*, 17(1), 51-75.
        """
        _validate_sample_size(n, minimum=3)
        _validate_confidence_level(confidence_level)
        _validate_correlation(r)
        _validate_sd(sd1, "sd1")
        _validate_sd(sd2, "sd2")

        corrected_r = r * (gmean([sd1**2, sd2**2]) / np.mean([sd1**2, sd2**2]))
        df = n - 1
        df_corrected = 2.0 / (1.0 + r**2) * df
        c = _bias_correction(df_corrected)
        scale = math.sqrt(n / (2.0 * (1.0 - corrected_r)))
        lambda_val = float(d * c * scale)
        alpha = 1.0 - confidence_level
        ci_low = float(
            _qlambdap(alpha / 2.0, df=df_corrected, t=lambda_val) / scale
        )
        ci_high = float(
            _qlambdap(1.0 - alpha / 2.0, df=df_corrected, t=lambda_val) / scale
        )
        return _build_result(
            d=d,
            ci_low=ci_low,
            ci_high=ci_high,
            se=None,
            method="t_t_prime",
            design="paired",
            confidence_level=confidence_level,
            metadata={
                "n": n,
                "df": df,
                "df_corrected": df_corrected,
                "r": r,
                "sd1": sd1,
                "sd2": sd2,
                "corrected_correlation": float(corrected_r),
                "lambda_val": lambda_val,
            },
        )

    @staticmethod
    def independent_t_central(
        d: float,
        n1: int,
        n2: int,
        confidence_level: float = 0.95,
    ) -> list[CohensDCIResult]:
        """Central CIs for independent-samples d using seven SE estimators.

        Returns seven results in this order: true, Morris, Hedges,
        Hedges-Olkin, MLE, Large-N, Small-N.
        """
        _validate_two_sample_sizes(n1, n2, minimum=2)
        if n1 + n2 <= 4:
            raise ValueError(f"n1 + n2 must be > 4 (got {n1 + n2}).")
        _validate_confidence_level(confidence_level)
        _, _, se_true, se_morris, se_hedges, se_hedges_olkin, se_mle, se_large_n, se_small_n = (
            _calculate_central_ci_from_cohens_d_two_independent_sample_t_test(
                d,
                n1,
                n2,
                confidence_level,
            )
        )
        return _multiple_se_results(
            d=d,
            confidence_level=confidence_level,
            design="independent",
            df=n1 + n2 - 2,
            se_values={
                "true": se_true,
                "morris": se_morris,
                "hedges": se_hedges,
                "hedges_olkin": se_hedges_olkin,
                "mle": se_mle,
                "large_n": se_large_n,
                "small_n": se_small_n,
            },
            metadata={"n1": n1, "n2": n2},
        )

    @staticmethod
    def independent_t_pivotal(
        t_stat: float,
        n1: int,
        n2: int,
        confidence_level: float = 0.95,
    ) -> CohensDCIResult:
        """Pivotal NCT CI for independent-samples d.

        The legacy source inverts the noncentral t distribution for the test
        statistic, then rescales the resulting NCP bounds to Cohen's d using
        ``sqrt((n1 + n2) / (n1 * n2))``.
        """
        _validate_two_sample_sizes(n1, n2, minimum=2)
        _validate_confidence_level(confidence_level)
        total_n = n1 + n2
        constant = math.sqrt(total_n / (n1 * n2))
        lower_ncp, upper_ncp = _nct_ci_t(t_stat, total_n - 2, confidence_level)
        d = t_stat * constant
        return _build_result(
            d=d,
            ci_low=lower_ncp * constant,
            ci_high=upper_ncp * constant,
            se=None,
            method="t_pivotal",
            design="independent",
            confidence_level=confidence_level,
            metadata={
                "n1": n1,
                "n2": n2,
                "df": total_n - 2,
                "t_stat": t_stat,
                "ncp_low": lower_ncp,
                "ncp_high": upper_ncp,
            },
        )

    @staticmethod
    def independent_t_ncp(
        t_stat: float,
        n1: int,
        n2: int,
        confidence_level: float = 0.95,
    ) -> CohensDCIResult:
        """NCP-based CI for independent-samples d.

        Returns the same d-scale interval as the legacy NCT inversion while also
        storing the raw NCP bounds in ``metadata``.
        """
        _validate_two_sample_sizes(n1, n2, minimum=2)
        _validate_confidence_level(confidence_level)
        total_n = n1 + n2
        constant = math.sqrt(total_n / (n1 * n2))
        lower_ncp, upper_ncp = _nct_ci_t(t_stat, total_n - 2, confidence_level)
        d = t_stat * constant
        return _build_result(
            d=d,
            ci_low=lower_ncp * constant,
            ci_high=upper_ncp * constant,
            se=None,
            method="t_ncp",
            design="independent",
            confidence_level=confidence_level,
            metadata={
                "n1": n1,
                "n2": n2,
                "df": total_n - 2,
                "t_stat": t_stat,
                "ncp_low": lower_ncp,
                "ncp_high": upper_ncp,
            },
        )


__all__ = ["CohensDCI", "CohensDCIResult"]
