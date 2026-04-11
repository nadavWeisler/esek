"""
Confidence interval utility functions for the ESEK package.

This module provides functions for computing confidence intervals for various
effect sizes, including Cohen's d, Hedges' g, and related statistics for
one-sample, paired, and two-sample tests.
"""

import math
import numpy as np
from scipy import stats


def ci_from_cohens_simple(
    cohens_d: float, sample_size: float, confidence_level: float
) -> tuple[float, float, float]:
    """
    Calculate the confidence intervals and standard error for Cohen's d effect size in a
    one-sample Z-test.

    This function calculates the confidence intervals of the effect size (Cohen's d) for a
    one-sample Z-test or two dependent samples test using the Hedges and Olkin (1985)
    formula to estimate the standard error.

    Parameters
    ----------
    cohens_d : float
        The calculated Cohen's d effect size
    sample_size : float
        The size of the sample
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%)

    Returns
    -------
    tuple
        A tuple containing:
        - ci_lower (float): Lower bound of the confidence interval
        - ci_upper (float): Upper bound of the confidence interval
        - standard_error_es (float): Standard error of the effect size

    Notes
    -----
    Since the effect size in the population and its standard deviation are unknown,
    we estimate it based on the sample using the Hedges and Olkin (1985) formula
    to estimate the standard deviation of the effect size.
    """
    standard_error_es = np.sqrt((1 / sample_size) + ((cohens_d**2 / (2 * sample_size))))
    z_critical_value = stats.norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = (
        cohens_d - standard_error_es * z_critical_value,
        cohens_d + standard_error_es * z_critical_value,
    )
    return ci_lower, ci_upper, standard_error_es


def compute_fisher_confidence_interval(
    correlation: float, standard_error: float, z_critical: float
) -> tuple[float, float]:
    """
    Compute the lower and upper confidence intervals for a given correlation using Fisher's z-transformation.

    Args:
        correlation (float): The correlation coefficient (e.g., rank-biserial).
        standard_error (float): The standard error of the correlation.
        z_critical (float): The z-value for the desired confidence level (e.g., 1.96 for 95%).

    Returns:
        tuple[float, float]: The lower and upper confidence interval, bounded to [-1, 1].
    """
    safe_corr = max(min(correlation, 0.999999), -0.999999)
    fisher_z = math.atanh(safe_corr)
    margin = z_critical * standard_error
    lower = max(math.tanh(fisher_z - margin), -1)
    upper = min(math.tanh(fisher_z + margin), 1)
    return lower, upper


def pivotal_ci_t(
    t_score: float, df: float, sample_size: int, confidence_level: float
) -> tuple[float, float]:
    """
    Calculate the pivotal confidence interval for a t-score using bisection.

    Parameters
    ----------
    t_score : float
        The observed t-score.
    df : float
        Degrees of freedom.
    sample_size : int
        The size of the sample.
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%).

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the pivotal confidence interval.
    """
    is_negative = False
    if t_score < 0:
        is_negative = True
        t_score = abs(t_score)
    upper_limit = 1 - (1 - confidence_level) / 2
    lower_limit = (1 - confidence_level) / 2

    lower_criterion = [-t_score, t_score / 2, t_score]
    upper_criterion = [t_score, 2 * t_score, 3 * t_score]

    while stats.nct.cdf(t_score, df, lower_criterion[0]) < upper_limit:
        lower_criterion = [
            lower_criterion[0] - t_score,
            lower_criterion[0],
            lower_criterion[2],
        ]

    while stats.nct.cdf(t_score, df, upper_criterion[0]) < lower_limit:
        if stats.nct.cdf(t_score, df) < lower_limit:
            lower_ci = [0, stats.nct.cdf(t_score, df)]
            upper_criterion = [
                upper_criterion[0] / 4,
                upper_criterion[0],
                upper_criterion[2],
            ]

    while stats.nct.cdf(t_score, df, upper_criterion[2]) > lower_limit:
        upper_criterion = [
            upper_criterion[0],
            upper_criterion[2],
            upper_criterion[2] + t_score,
        ]

    lower_ci = 0.0
    diff_lower = 1
    while diff_lower > 0.00001:
        if stats.nct.cdf(t_score, df, lower_criterion[1]) < upper_limit:
            lower_criterion = [
                lower_criterion[0],
                (lower_criterion[0] + lower_criterion[1]) / 2,
                lower_criterion[1],
            ]
        else:
            lower_criterion = [
                lower_criterion[1],
                (lower_criterion[1] + lower_criterion[2]) / 2,
                lower_criterion[2],
            ]
        diff_lower = abs(stats.nct.cdf(t_score, df, lower_criterion[1]) - upper_limit)
        lower_ci = lower_criterion[1] / (np.sqrt(sample_size))

    upper_ci = 0.0
    diff_upper = 1
    while diff_upper > 0.00001:
        if stats.nct.cdf(t_score, df, upper_criterion[1]) < lower_limit:
            upper_criterion = [
                upper_criterion[0],
                (upper_criterion[0] + upper_criterion[1]) / 2,
                upper_criterion[1],
            ]
        else:
            upper_criterion = [
                upper_criterion[1],
                (upper_criterion[1] + upper_criterion[2]) / 2,
                upper_criterion[2],
            ]
        diff_upper = abs(stats.nct.cdf(t_score, df, upper_criterion[1]) - lower_limit)
        upper_ci = upper_criterion[1] / (np.sqrt(sample_size))
    if is_negative:
        return -upper_ci, -lower_ci
    else:
        return lower_ci, upper_ci


def ci_from_cohens_paired(
    cohens_d: float, sample_size: int, confidence_level: float
) -> tuple[float, float, float]:
    """
    Calculate confidence intervals and standard error for Cohen's d in a paired samples t-test.

    Parameters
    ----------
    cohens_d : float
        The calculated Cohen's d effect size.
    sample_size : int
        The size of the sample.
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%).

    Returns
    -------
    tuple
        A tuple containing:
        - ci_lower (float): Lower bound of the confidence interval.
        - ci_upper (float): Upper bound of the confidence interval.
        - standard_error_es (float): Standard error of the effect size.
    """
    df = sample_size - 1
    if df <= 2:
        raise ValueError("Degrees of freedom must be greater than 2.")
    correction_factor = math.exp(
        math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2)
    )
    standard_error_es = np.sqrt(
        (df / (df - 2)) * (1 / sample_size) * (1 + cohens_d**2 * sample_size)
        - (cohens_d**2 / correction_factor**2)
    )
    z_critical_value = stats.norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = (
        cohens_d - standard_error_es * z_critical_value,
        cohens_d + standard_error_es * z_critical_value,
    )
    return ci_lower, ci_upper, standard_error_es


def ci_from_cohens_d_t_test(
    effect_size: float, sample_size_1: int, sample_size_2: int, confidence_level: float
) -> tuple:
    """
    Calculate confidence intervals and standard errors for Cohen's d in a two-sample t-test.

    Parameters
    ----------
    effect_size : float
        The calculated Cohen's d effect size.
    sample_size_1 : int
        The size of the first sample.
    sample_size_2 : int
        The size of the second sample.
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%).

    Returns
    -------
    tuple
        A tuple containing ci_lower, ci_upper, and seven standard error estimates
        (True, Morris, Hedges, Hedges-Olkin, MLE, Large-N, Small-N).
    """
    sample_size = sample_size_1 + sample_size_2
    df = sample_size - 2
    if df <= 2:
        raise ValueError("Degrees of freedom must be greater than 2.")

    correction_factor = math.exp(
        math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2)
    )
    harmonic_sample_size = 2 / (1 / sample_size_1 + 1 / sample_size_2)
    a = harmonic_sample_size / 2
    standard_error_effect_size_true = np.sqrt(
        (
            (df / (df - 2)) * (1 / a) * (1 + effect_size**2 * a)
            - (effect_size**2 / correction_factor**2)
        )
    )
    standard_error_effect_size_morris = np.sqrt(
        (df / (df - 2)) * (1 / a) * (1 + effect_size**2 * a)
        - (effect_size**2 / (1 - (3 / (4 * (df - 1) - 1))) ** 2)
    )
    standard_error_effect_size_hedges = np.sqrt((1 / a) + effect_size**2 / (2 * df))
    standard_error_effect_size_hedges_olkin = np.sqrt(
        (1 / a) + effect_size**2 / (2 * sample_size)
    )
    standard_error_effect_size_mle = np.sqrt(
        standard_error_effect_size_hedges * ((df + 2) / df)
    )
    standard_error_effect_size_large_n = np.sqrt(1 / a * (1 + effect_size**2 / 8))
    standard_error_effect_size_small_n = np.sqrt(
        standard_error_effect_size_large_n * ((df + 1) / (df - 1))
    )
    z_critical_value = stats.norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = (
        effect_size - standard_error_effect_size_true * z_critical_value,
        effect_size + standard_error_effect_size_true * z_critical_value,
    )
    return (
        ci_lower,
        ci_upper,
        standard_error_effect_size_true,
        standard_error_effect_size_morris,
        standard_error_effect_size_hedges,
        standard_error_effect_size_hedges_olkin,
        standard_error_effect_size_mle,
        standard_error_effect_size_large_n,
        standard_error_effect_size_small_n,
    )


def ci_from_cohens_d_two_samples(
    cohens_d: float, sample_size_1: int, sample_size_2: int, confidence_level: float
) -> tuple[float, float, float]:
    """
    Calculate confidence intervals and standard error for Cohen's d using the
    two-sample formula.

    Parameters
    ----------
    cohens_d : float
        The calculated Cohen's d effect size.
    sample_size_1 : int
        The size of the first sample.
    sample_size_2 : int
        The size of the second sample.
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%).

    Returns
    -------
    tuple[float, float, float]
        Lower bound, upper bound, and standard error of the confidence interval.
    """
    standard_error_es = np.sqrt(
        ((sample_size_1 + sample_size_2) / (sample_size_1 * sample_size_2))
        + ((cohens_d**2 / (2 * (sample_size_1 + sample_size_2))))
    )
    z_critical_value = stats.norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower = cohens_d - standard_error_es * z_critical_value
    ci_upper = cohens_d + standard_error_es * z_critical_value
    return ci_lower, ci_upper, standard_error_es


def central_ci_paired(
    effect_size: float, sample_size: float, confidence_level: float
) -> tuple:
    """
    Calculates the confidence intervals and standard errors for various effect sizes
    Parameters
    ----------
    effect_size : float
        The calculated effect size (Cohen's d).
    sample_size : int
        The size of the sample.
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%).

    Returns
    -------
    tuple
        A tuple containing:
        - ci_lower (float): Lower bound of the confidence interval.
        - ci_upper (float): Upper bound of the confidence interval.
        - Standard_error_effect_size_True (float): Standard error of the effect size (True).
        - Standard_error_effect_size_Morris (float): Standard error of the effect size (Morris).
        - Standard_error_effect_size_Hedges (float): Standard error of the effect size (Hedges).
        - Standard_error_effect_size_Hedges_Olkin (float): Standard error of the effect size (Hedges_Olkin).
        - Standard_error_effect_size_MLE (float): Standard error of the effect size (MLE).
        - Standard_error_effect_size_Large_N (float): Standard error of the effect size (Large N).
        - Standard_error_effect_size_Small_N (float): Standard error of the effect size (Small N).
    """
    df = sample_size - 1
    correction_factor = math.exp(
        math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2)
    )
    standard_error_effect_size_true = np.sqrt(
        (
            (df / (df - 2)) * (1 / sample_size) * (1 + effect_size**2 * sample_size)
            - (effect_size**2 / correction_factor**2)
        )
    )
    standard_error_effect_size_morris = np.sqrt(
        (df / (df - 2)) * (1 / sample_size) * (1 + effect_size**2 * sample_size)
        - (effect_size**2 / (1 - (3 / (4 * (df - 1) - 1))) ** 2)
    )
    standard_error_effect_size_hedges = np.sqrt(
        (1 / sample_size) + effect_size**2 / (2 * df)
    )
    standard_error_effect_size_hedges_olkin = np.sqrt(
        (1 / sample_size) + effect_size**2 / (2 * sample_size)
    )
    standard_error_effect_size_mle = np.sqrt(
        standard_error_effect_size_hedges * ((df + 2) / df)
    )
    standard_error_effect_size_large_n = np.sqrt(
        1 / sample_size * (1 + effect_size**2 / 8)
    )
    standard_error_effect_size_small_n = np.sqrt(
        standard_error_effect_size_large_n * ((df + 1) / (df - 1))
    )
    z_critical_value = stats.norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = (
        effect_size - standard_error_effect_size_true * z_critical_value,
        effect_size + standard_error_effect_size_true * z_critical_value,
    )
    return (
        ci_lower,
        ci_upper,
        standard_error_effect_size_true,
        standard_error_effect_size_morris,
        standard_error_effect_size_hedges,
        standard_error_effect_size_hedges_olkin,
        standard_error_effect_size_mle,
        standard_error_effect_size_large_n,
        standard_error_effect_size_small_n,
    )


def calculate_se_pooled(
    effect_size: float,
    sample_size: float,
    correlation: float,
    confidence_level: float,
) -> tuple:
    """
    Calculates the standard error and confidence intervals for pooled effect sizes.

    Parameters
    ----------
    effect_size : float
        The calculated effect size (Cohen's d).
    sample_size : float
        The size of the sample.
    correlation : float
        The correlation between the two groups.
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%).

    Returns
    -------
    tuple
        A tuple containing ci_lower, ci_upper, and seven standard error estimates
        (True, Morris, Hedges, Hedges-Olkin, MLE, Large-N, Small-N).
    """
    df = sample_size - 1
    correction_factor = math.exp(
        math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2)
    )
    A = sample_size / (2 * (1 - correlation))
    standard_error_effect_size_true = np.sqrt(
        (
            (df / (df - 2)) * (1 / A) * (1 + effect_size**2 * A)
            - (effect_size**2 / correction_factor**2)
        )
    )
    standard_error_effect_size_morris = np.sqrt(
        (df / (df - 2)) * (1 / A) * (1 + effect_size**2 * A)
        - (effect_size**2 / (1 - (3 / (4 * (df - 1) - 1))) ** 2)
    )
    standard_error_effect_size_hedges = np.sqrt((1 / A) + effect_size**2 / (2 * df))
    standard_error_effect_size_hedges_olkin = np.sqrt(
        (1 / A) + effect_size**2 / (2 * sample_size)
    )
    standard_error_effect_size_mle = np.sqrt(
        standard_error_effect_size_hedges * ((df + 2) / df)
    )
    standard_error_effect_size_large_n = np.sqrt(1 / A * (1 + effect_size**2 / 8))
    standard_error_effect_size_small_n = np.sqrt(
        standard_error_effect_size_large_n * ((df + 1) / (df - 1))
    )
    z_critical_value = stats.norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = (
        effect_size - standard_error_effect_size_true * z_critical_value,
        effect_size + standard_error_effect_size_true * z_critical_value,
    )
    return (
        ci_lower,
        ci_upper,
        standard_error_effect_size_true,
        standard_error_effect_size_morris,
        standard_error_effect_size_hedges,
        standard_error_effect_size_hedges_olkin,
        standard_error_effect_size_mle,
        standard_error_effect_size_large_n,
        standard_error_effect_size_small_n,
    )


def ci_t_prime(
    effect_size: float,
    standard_deviation_1: float,
    standard_deviation_2: float,
    sample_size: float,
    correlation: float,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Calculates the confidence interval for the t-prime effect size.

    Parameters
    ----------
    effect_size : float
        The calculated effect size.
    standard_deviation_1 : float
        Standard deviation of the first group.
    standard_deviation_2 : float
        Standard deviation of the second group.
    sample_size : float
        The size of the sample.
    correlation : float
        The correlation between the two groups.
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%).

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the confidence interval.
    """
    corrected_correlation = correlation * (
        stats.gmean([standard_deviation_1**2, standard_deviation_2**2])
        / np.mean((standard_deviation_1**2, standard_deviation_2**2))
    )
    df = sample_size - 1
    df_corrected = 2 / (1 + correlation**2) * df
    correction = math.exp(
        math.lgamma(df_corrected / 2)
        - math.log(math.sqrt(df_corrected / 2))
        - math.lgamma((df_corrected - 1) / 2)
    )
    lambda_function = float(
        effect_size
        * correction
        * np.sqrt(sample_size / (2 * (1 - corrected_correlation)))
    )

    p_lower = 0.5 - confidence_level / 2
    p_upper = 0.5 + confidence_level / 2

    dfn = 1
    dfd = df_corrected

    lower_q = stats.ncf.ppf(p_lower, dfn, dfd, lambda_function)
    upper_q = stats.ncf.ppf(p_upper, dfn, dfd, lambda_function)

    denominator = np.sqrt(sample_size / (2 * (1 - corrected_correlation)))
    lower_ci_adjusted_lambda = lower_q / denominator
    upper_ci_adjusted_lambda = upper_q / denominator

    return lower_ci_adjusted_lambda, upper_ci_adjusted_lambda


def ci_adjusted_lambda_prime(
    effect_size: float,
    standard_deviation_1: float,
    standard_deviation_2: float,
    sample_size: float,
    correlation: float,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Calculates the confidence interval for the adjusted lambda prime effect size.

    Parameters
    ----------
    effect_size : float
        The calculated effect size.
    standard_deviation_1 : float
        Standard deviation of the first group.
    standard_deviation_2 : float
        Standard deviation of the second group.
    sample_size : float
        The size of the sample.
    correlation : float
        The correlation between the two groups.
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%).

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the confidence interval.
    """
    corrected_correlation = correlation * (
        stats.gmean([standard_deviation_1**2, standard_deviation_2**2])
        / np.mean((standard_deviation_1**2, standard_deviation_2**2))
    )
    df = sample_size - 1
    df_corrected = 2 / (1 + correlation**2) * df
    correction1 = math.exp(
        math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2)
    )
    correction2 = math.exp(
        math.lgamma(df_corrected / 2)
        - math.log(math.sqrt(df_corrected / 2))
        - math.lgamma((df_corrected - 1) / 2)
    )
    lambda_function = float(
        effect_size
        * correction1
        * np.sqrt(sample_size / (2 * (1 - corrected_correlation)))
    )

    p_lower = 0.5 - confidence_level / 2
    p_upper = 0.5 + confidence_level / 2

    dfn = 1
    dfd = df_corrected

    lower_q = stats.ncf.ppf(p_lower, dfn, dfd, lambda_function)
    upper_q = stats.ncf.ppf(p_upper, dfn, dfd, lambda_function)

    denominator = 2 * (1 - corrected_correlation) * correction2
    lower_ci_adjusted_lambda = lower_q / denominator
    upper_ci_adjusted_lambda = upper_q / denominator

    return lower_ci_adjusted_lambda, upper_ci_adjusted_lambda


def ci_mag(
    effect_size: float,
    standard_deviation_1: float,
    standard_deviation_2: float,
    sample_size: float,
    correlation: float,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Calculates the confidence interval for the magnitude effect size.

    Parameters
    ----------
    effect_size : float
        The calculated effect size.
    standard_deviation_1 : float
        Standard deviation of the first group.
    standard_deviation_2 : float
        Standard deviation of the second group.
    sample_size : float
        The size of the sample.
    correlation : float
        The correlation between the two groups.
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%).

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the confidence interval.
    """
    corrected_correlation = correlation * (
        stats.gmean([standard_deviation_1**2, standard_deviation_2**2])
        / np.mean((standard_deviation_1**2, standard_deviation_2**2))
    )
    df = sample_size - 1
    correction = math.exp(
        math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2)
    )
    lambda_function = float(
        effect_size
        * correction**2
        * np.sqrt(sample_size / (2 * (1 - corrected_correlation)))
    )
    lower_ci_adjusted_mag = stats.nct.ppf(
        1 / 2 - confidence_level / 2, df=df, nc=lambda_function
    ) / np.sqrt(sample_size / (2 * (1 - corrected_correlation)))
    upper_ci_adjusted_mag = stats.nct.ppf(
        1 / 2 + confidence_level / 2, df=df, nc=lambda_function
    ) / np.sqrt(sample_size / (2 * (1 - corrected_correlation)))
    return lower_ci_adjusted_mag, upper_ci_adjusted_mag


def ci_morris(
    effect_size: float,
    sample_size: float,
    correlation: float,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Calculates the Morris confidence interval for the effect size.

    Parameters
    ----------
    effect_size : float
        The calculated effect size (Cohen's d).
    sample_size : float
        The size of the sample.
    correlation : float
        The correlation between the two groups.
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%).

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the Morris confidence interval.
    """
    df = sample_size - 1
    correction = math.exp(
        math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2)
    )
    cohens_d_variance_corrected = (
        (df / (df - 2))
        * 2
        * (1 - correlation)
        / sample_size
        * (1 + effect_size**2 * sample_size / (2 * (1 - correlation)))
        - effect_size**2 / correction**2
    ) * correction**2
    z_critical_value = stats.norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower_morris, ci_upper_morris = (
        effect_size - np.sqrt(cohens_d_variance_corrected) * z_critical_value,
        effect_size + np.sqrt(cohens_d_variance_corrected) * z_critical_value,
    )
    return ci_lower_morris, ci_upper_morris


def ci_ncp(
    effect_size: float, sample_size: int, confidence_level: float
) -> tuple[float, float]:
    """
    Calculate the Non-Central Parameter (NCP) confidence intervals for a one-sample t-test.

    Parameters
    ----------
    effect_size : float
        The calculated effect size (Cohen's d).
    sample_size : int
        The size of the sample.
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%).

    Returns
    -------
    tuple
        A tuple containing:
        - ci_ncp_low (float): Lower bound of the NCP confidence interval.
        - ci_ncp_high (float): Upper bound of the NCP confidence interval.
    """
    ncp_value = effect_size * math.sqrt(sample_size)

    reduced_size = sample_size - 1
    if reduced_size <= 0:
        raise ValueError("Sample size must be greater than 1 for NCP calculation.")

    def ci_ncp_ppf(is_high: bool) -> float:
        q = 0.5 + confidence_level / 2 if is_high else 0.5 - confidence_level / 2

        return float(
            stats.nct.ppf(
                q,
                reduced_size,
                loc=0,
                scale=1,
                nc=ncp_value,
            )
        )

    ci_ncp_low = ci_ncp_ppf(False) / ncp_value * effect_size
    ci_ncp_high = ci_ncp_ppf(True) / ncp_value * effect_size

    return ci_ncp_low, ci_ncp_high
