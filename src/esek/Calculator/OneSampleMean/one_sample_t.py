"""
This module contains functions and classes for performing one-sample t-tests and
calculating various statistics such as Cohen's d, Hedges' g, t-score, p-value,
confidence intervals, and standard errors.

The module includes the following functions:
- pivotal_ci_t: Calculate the Pivotal confidence intervals for a one-sample t-test.
- calculate_central_ci_one_sample_t_test: Calculate the central confidence intervals
for the effect size in a one-sample t-test.
- CI_NCP_one_Sample: Calculate the Non-Central Parameter (NCP) confidence intervals
for a one-sample t-test.

The module also includes the following class:
- One_Sample_ttest: A class containing static methods for performing one-sample
t-tests from t-score and parameters.
"""

import math
from typing import Optional
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm, nct, t
from ...utils import interfaces
from ...utils import res


@dataclass
class OneSampleTResults:
    """
    A class to store results from one-sample t-tests.
    """

    # Effect sizes
    cohens_d: Optional[res.CohenD] = None
    hedges_g: Optional[res.HedgesG] = None
    # Test statistics
    t_score: Optional[float] = None
    degrees_of_freedom: Optional[int | float] = None
    p_value: Optional[float] = None
    standard_error: Optional[float | int] = None
    sample_mean: Optional[float | int] = None
    population_mean: Optional[float | int] = None
    means_difference: Optional[float | int] = None
    sample_size: Optional[float | int] = None
    sample_sd: Optional[float | int] = None


def pivotal_ci_t(t_score, df, sample_size, confidence_level):
    """
    Calculate the Pivotal confidence intervals for a one-sample t-test.

    Parameters
    ----------
    t_score : float
        The t-score value.
    df : int
        Degrees of freedom.
    sample_size : int
        The size of the sample.
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%).

    Returns
    -------
    tuple
        A tuple containing:
        - lower_ci (float): Lower bound of the confidence interval.
        - upper_ci (float): Upper bound of the confidence interval.
    """
    is_negative = False
    if t_score < 0:
        is_negative = True
        t_score = abs(t_score)
    upper_limit = 1 - (1 - confidence_level) / 2
    lower_limit = (1 - confidence_level) / 2

    lower_criterion = [-t_score, t_score / 2, t_score]
    upper_criterion = [t_score, 2 * t_score, 3 * t_score]

    while nct.cdf(t_score, df, lower_criterion[0]) < upper_limit:
        lower_criterion = [
            lower_criterion[0] - t_score,
            lower_criterion[0],
            lower_criterion[2],
        ]

    while nct.cdf(t_score, df, upper_criterion[0]) < lower_limit:
        if nct.cdf(t_score, df) < lower_limit:
            lower_ci = [0, nct.cdf(t_score, df)]
            upper_criterion = [
                upper_criterion[0] / 4,
                upper_criterion[0],
                upper_criterion[2],
            ]

    while nct.cdf(t_score, df, upper_criterion[2]) > lower_limit:
        upper_criterion = [
            upper_criterion[0],
            upper_criterion[2],
            upper_criterion[2] + t_score,
        ]

    lower_ci = 0.0
    diff_lower = 1
    while diff_lower > 0.00001:
        if nct.cdf(t_score, df, lower_criterion[1]) < upper_limit:
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
        diff_lower = abs(nct.cdf(t_score, df, lower_criterion[1]) - upper_limit)
        lower_ci = lower_criterion[1] / (np.sqrt(sample_size))

    upper_ci = 0.0
    diff_upper = 1
    while diff_upper > 0.00001:
        if nct.cdf(t_score, df, upper_criterion[1]) < lower_limit:
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
        diff_upper = abs(nct.cdf(t_score, df, upper_criterion[1]) - lower_limit)
        upper_ci = upper_criterion[1] / (np.sqrt(sample_size))
    if is_negative:
        return -upper_ci, -lower_ci
    else:
        return lower_ci, upper_ci


def calculate_central_ci_one_sample_t_test(effect_size, sample_size, confidence_level):
    """
    Calculate the central confidence intervals for the effect size in a one-sample t-test.

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
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
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


def ci_ncp_one_sample(effect_size, sample_size, confidence_level):
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
        - CI_NCP_low (float): Lower bound of the NCP confidence interval.
        - CI_NCP_High (float): Upper bound of the NCP confidence interval.
    """
    NCP_value = effect_size * math.sqrt(sample_size)
    CI_NCP_low = (
        (
            nct.ppf(
                1 / 2 - confidence_level / 2,
                (sample_size - 1),
                loc=0,
                scale=1,
                nc=NCP_value,
            )
        )
        / NCP_value
        * effect_size
    )
    CI_NCP_High = (
        (
            nct.ppf(
                1 / 2 + confidence_level / 2,
                (sample_size - 1),
                loc=0,
                scale=1,
                nc=NCP_value,
            )
        )
        / NCP_value
        * effect_size
    )
    return CI_NCP_low, CI_NCP_High


class OneSampleTTest(interfaces.AbstractTest):
    """
    A class to perform one-sample t-tests and calculate various statistics.
    This class provides methods to calculate T-test results from a t-score,
    from sample parameters, and from sample data.
    """

    @staticmethod
    def from_t_score(
        t_score: float, sample_size: float, confidence_level=0.95
    ) -> OneSampleTResults:
        """
        Calculate the one-sample t-test results from a given t-score.
        """

        # Calculation
        df = sample_size - 1
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)
        cohens_d = t_score / np.sqrt(
            df
        )  # This is Cohen's d and it is calculated based on the sample's standard deviation
        correction = math.exp(
            math.lgamma(df / 2)
            - math.log(math.sqrt(df / 2))
            - math.lgamma((df - 1) / 2)
        )
        hedges_g = correction * cohens_d
        (
            ci_lower_cohens_d_central,
            ci_upper_cohens_d_central,
            standard_error_cohens_d_true,
            standard_error_cohens_d_morris,
            standard_error_cohens_d_hedges,
            standard_error_cohens_d_hedges_olkin,
            standard_error_cohens_d_mle,
            standard_error_cohens_d_large_n,
            standard_error_cohens_d_small_n,
        ) = calculate_central_ci_one_sample_t_test(
            cohens_d, sample_size, confidence_level
        )
        (
            ci_lower_hedges_g_central,
            ci_upper_hedges_g_central,
            standard_error_hedges_g_true,
            standard_error_hedges_g_morris,
            standard_error_hedges_g_hedges,
            standard_error_hedges_g_hedges_olkin,
            standard_error_hedges_g_mle,
            standard_error_hedges_g_large_n,
            standard_error_hedges_g_small_n,
        ) = calculate_central_ci_one_sample_t_test(
            hedges_g, sample_size, confidence_level
        )
        ci_lower_cohens_d_pivotal, ci_upper_cohens_d_pivotal = pivotal_ci_t(
            t_score, df, sample_size, confidence_level
        )
        ci_lower_hedges_g_pivotal, ci_upper_hedges_g_pivotal = pivotal_ci_t(
            t_score, df, sample_size, confidence_level
        )
        ci_lower_cohens_d_ncp, ci_upper_cohens_d_ncp = ci_ncp_one_sample(
            cohens_d, sample_size, confidence_level
        )
        ci_lower_hedges_g_ncp, ci_upper_hedges_g_ncp = ci_ncp_one_sample(
            hedges_g, sample_size, confidence_level
        )

        cohens_d = res.CohenD(
            cohens_d,
            ci_lower_cohens_d_central,
            ci_upper_cohens_d_central,
            standard_error_cohens_d_true,
        )

        cohens_d.standardizer = correction
        cohens_d.update_non_central_ci(ci_lower_cohens_d_ncp, ci_upper_cohens_d_ncp)
        cohens_d.update_pivotal_ci(ci_lower_cohens_d_pivotal, ci_upper_cohens_d_pivotal)

        cohens_d_approximated = res.ApproximatedStandardError(
            standard_error_cohens_d_true,
            standard_error_cohens_d_morris,
            standard_error_cohens_d_hedges,
            standard_error_cohens_d_hedges_olkin,
            standard_error_cohens_d_mle,
            standard_error_cohens_d_large_n,
            standard_error_cohens_d_small_n,
        )

        cohens_d.approximated_standard_error = cohens_d_approximated

        hedges_g = res.HedgesG(
            hedges_g,
            ci_lower_hedges_g_central,
            ci_upper_hedges_g_central,
            standard_error_hedges_g_true,
        )

        hedges_g.update_non_central_ci(ci_lower_hedges_g_ncp, ci_upper_hedges_g_ncp)
        hedges_g.update_pivotal_ci(ci_lower_hedges_g_pivotal, ci_upper_hedges_g_pivotal)

        hedges_g_approximated = res.ApproximatedStandardError(
            standard_error_hedges_g_true,
            standard_error_hedges_g_morris,
            standard_error_hedges_g_hedges,
            standard_error_hedges_g_hedges_olkin,
            standard_error_hedges_g_mle,
            standard_error_hedges_g_large_n,
            standard_error_hedges_g_small_n,
        )
        hedges_g.approximated_standard_error = hedges_g_approximated

        results = OneSampleTResults()
        results.t_score = t_score
        results.degrees_of_freedom = df
        results.p_value = p_value
        results.cohens_d = cohens_d
        results.hedges_g = hedges_g

        return results

    @staticmethod
    def from_z_score() -> OneSampleTResults:
        """
        Calculate the one-sample t-test results from a Z-score.
        This method is not implemented yet.
        """
        raise NotImplementedError("This method is not implemented yet.")

    @staticmethod
    def from_parameters(
        population_mean: float,
        sample_mean: float,
        sample_sd: float,
        sample_size: float,
        confidence_level: float = 0.95,
    ) -> OneSampleTResults:
        """
        Calculate the one-sample t-test results from given parameters.
        """
        df = sample_size - 1
        standard_error = sample_sd / np.sqrt(
            df
        )  # This is the standrt error of mean's estimate in o ne samaple t-test
        t_score = (
            sample_mean - population_mean
        ) / standard_error  # This is the t score in the test which is used to calculate the p-value
        cohens_d = (
            sample_mean - population_mean
        ) / sample_sd  # This is the effect size for one sample t-test Cohen's d
        correction = math.exp(
            math.lgamma(df / 2)
            - math.log(math.sqrt(df / 2))
            - math.lgamma((df - 1) / 2)
        )
        hedges_g = cohens_d * correction  # This is the actual corrected effect size
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)
        (
            ci_lower_cohens_d_central,
            ci_upper_cohens_d_central,
            standard_error_cohens_d_true,
            standard_error_cohens_d_morris,
            standard_error_cohens_d_hedges,
            standard_error_cohens_d_hedges_olkin,
            standard_error_cohens_d_mle,
            standard_error_cohens_d_large_n,
            standard_error_cohens_d_small_n,
        ) = calculate_central_ci_one_sample_t_test(
            cohens_d, sample_size, confidence_level
        )
        (
            ci_lower_hedges_g_central,
            ci_upper_hedges_g_central,
            standard_error_hedges_g_true,
            standard_error_hedges_g_morris,
            standard_error_hedges_g_hedges,
            standard_error_hedges_g_hedges_olkin,
            standard_error_hedges_g_mle,
            standard_error_hedges_g_large_n,
            standard_error_hedges_g_small_n,
        ) = calculate_central_ci_one_sample_t_test(
            hedges_g, sample_size, confidence_level
        )
        ci_lower_cohens_d_pivotal, ci_upper_cohens_d_pivotal = pivotal_ci_t(
            t_score, df, sample_size, confidence_level
        )
        ci_lower_hedges_g_pivotal, ci_upper_hedges_g_pivotal = pivotal_ci_t(
            t_score, df, sample_size, confidence_level
        )
        ci_lower_cohens_d_ncp, ci_upper_cohens_d_ncp = ci_ncp_one_sample(
            cohens_d, sample_size, confidence_level
        )
        ci_lower_hedges_g_ncp, ci_upper_hedges_g_ncp = ci_ncp_one_sample(
            hedges_g, sample_size, confidence_level
        )

        cohens_d = res.CohenD(
            cohens_d,
            ci_lower_cohens_d_central,
            ci_upper_cohens_d_central,
            standard_error_cohens_d_true,
        )
        cohens_d.standardizer = correction
        cohens_d.update_non_central_ci(
            float(ci_lower_cohens_d_ncp), float(ci_upper_cohens_d_ncp)
        )
        cohens_d.update_pivotal_ci(ci_lower_cohens_d_pivotal, ci_upper_cohens_d_pivotal)
        cohens_d_approximated = res.ApproximatedStandardError(
            standard_error_cohens_d_true,
            standard_error_cohens_d_morris,
            standard_error_cohens_d_hedges,
            standard_error_cohens_d_hedges_olkin,
            standard_error_cohens_d_mle,
            standard_error_cohens_d_large_n,
            standard_error_cohens_d_small_n,
        )
        cohens_d.approximated_standard_error = cohens_d_approximated

        hedges_g = res.HedgesG(
            hedges_g,
            ci_lower_hedges_g_central,
            ci_upper_hedges_g_central,
            standard_error_hedges_g_true,
        )

        hedges_g.update_non_central_ci(
            float(ci_lower_hedges_g_ncp), float(ci_upper_hedges_g_ncp)
        )
        hedges_g.update_pivotal_ci(ci_lower_hedges_g_pivotal, ci_upper_hedges_g_pivotal)
        hedges_g_approximated = res.ApproximatedStandardError(
            standard_error_hedges_g_true,
            standard_error_hedges_g_morris,
            standard_error_hedges_g_hedges,
            standard_error_hedges_g_hedges_olkin,
            standard_error_hedges_g_mle,
            standard_error_hedges_g_large_n,
            standard_error_hedges_g_small_n,
        )
        hedges_g.approximated_standard_error = hedges_g_approximated

        # Create results object
        results = OneSampleTResults()

        results.t_score = t_score
        results.degrees_of_freedom = df
        results.p_value = p_value
        results.cohens_d = cohens_d
        results.hedges_g = hedges_g

        # Assign values to the results object
        results.sample_size = sample_size
        results.population_mean = population_mean
        results.sample_mean = sample_mean
        results.sample_sd = sample_sd
        results.standard_error = standard_error
        results.means_difference = sample_mean - population_mean

        return results

    @staticmethod
    def from_data() -> OneSampleTResults:
        """
        Calculate the one-sample t-test results from sample data.
        This method is not implemented yet.
        """
        raise NotImplementedError("This method is not implemented yet.")
