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
import numpy as np
from scipy.stats import norm, nct, t


class OneSampleTResults:
    """
    A class to store results from one-sample t-tests.
    """

    def __init__(self):
        # Effect sizes
        self.cohens_d = None
        self.hedges_g = None

        # Test statistics
        self.t_score = None
        self.degrees_of_freedom = None
        self.p_value = None

        # Standard errors for Cohen's d
        self.standard_error_cohens_d_true = None
        self.standard_error_cohens_d_morris = None
        self.standard_error_cohens_d_hedges = None
        self.standard_error_cohens_d_hedges_olkin = None
        self.standard_error_cohens_d_mle = None
        self.standard_error_cohens_d_large_n = None
        self.standard_error_cohens_d_small_n = None

        # Standard errors for Hedges' g
        self.standard_error_hedges_g_true = None
        self.standard_error_hedges_g_morris = None
        self.standard_error_hedges_g_hedges = None
        self.standard_error_hedges_g_hedges_olkin = None
        self.standard_error_hedges_g_mle = None
        self.standard_error_hedges_g_large_n = None
        self.standard_error_hedges_g_small_n = None

        # Confidence intervals for Cohen's d
        self.lower_central_ci_cohens_d = None
        self.upper_central_ci_cohens_d = None
        self.lower_pivotal_ci_cohens_d = None
        self.upper_pivotal_ci_cohens_d = None
        self.lower_ncp_ci_cohens_d = None
        self.upper_ncp_ci_cohens_d = None

        # Confidence intervals for Hedges' g
        self.lower_central_ci_hedges_g = None
        self.upper_central_ci_hedges_g = None
        self.lower_pivotal_ci_hedges_g = None
        self.upper_pivotal_ci_hedges_g = None
        self.lower_ncp_ci_hedges_g = None
        self.upper_ncp_ci_hedges_g = None

        # Additional statistics
        self.correction_factor = None
        self.standard_error_mean = None
        self.standardizer_cohens_d = None
        self.standardizer_hedges_g = None
        self.sample_mean = None
        self.population_mean = None
        self.means_difference = None
        self.sample_size = None
        self.sample_sd = None


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


class OneSampleT:
    """
    A class containing static methods for performing one-sample t-tests from t-score and parameters.

    This class includes the following static methods:
    - one_sample_from_t_score: Calculate the one-sample t-test results from a given t-score.
    - one_sample_from_params: Calculate the one-sample t-test results from given parameters.
    """

    @staticmethod
    def one_sample_from_t_score(params: dict) -> OneSampleTResults:
        """
        Calculate the one-sample t-test results from a given t-score.

        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - "t score" (float): The t-score value.
            - "Sample Size" (int): The size of the sample.
            - "Confidence Level" (float): The confidence level as a percentage (e.g., 95 for 95%).

        Returns
        -------
        OneSampleTResults
            An object containing all calculated statistics and results
        """
        # Get params
        t_score = params["t score"]
        sample_size = params["Sample Size"]
        confidence_level_percentage = params["Confidence Level"]

        # Create results object
        results = OneSampleTResults()

        # Calculation
        confidence_level = confidence_level_percentage / 100
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

        # Set results
        results.cohens_d = round(cohens_d, 4)
        results.hedges_g = round(hedges_g, 4)
        results.t_score = round(t_score, 4)
        results.degrees_of_freedom = round(df, 4)
        results.p_value = round(p_value, 4)
        results.standard_error_cohens_d_true = round(standard_error_cohens_d_true, 4)
        results.standard_error_cohens_d_morris = round(
            standard_error_cohens_d_morris, 4
        )
        results.standard_error_cohens_d_hedges = round(
            standard_error_cohens_d_hedges, 4
        )
        results.standard_error_cohens_d_hedges_olkin = round(
            standard_error_cohens_d_hedges_olkin, 4
        )
        results.standard_error_cohens_d_mle = round(standard_error_cohens_d_mle, 4)
        results.standard_error_cohens_d_large_n = round(
            standard_error_cohens_d_large_n, 4
        )
        results.standard_error_cohens_d_small_n = round(
            standard_error_cohens_d_small_n, 4
        )
        results.standard_error_hedges_g_true = round(standard_error_hedges_g_true, 4)
        results.standard_error_hedges_g_morris = round(
            standard_error_hedges_g_morris, 4
        )
        results.standard_error_hedges_g_hedges = round(
            standard_error_hedges_g_hedges, 4
        )
        results.standard_error_hedges_g_hedges_olkin = round(
            standard_error_hedges_g_hedges_olkin, 4
        )
        results.standard_error_hedges_g_mle = round(standard_error_hedges_g_mle, 4)
        results.standard_error_hedges_g_large_n = round(
            standard_error_hedges_g_large_n, 4
        )
        results.standard_error_hedges_g_small_n = round(
            standard_error_hedges_g_small_n, 4
        )
        results.lower_central_ci_cohens_d = round(ci_lower_cohens_d_central, 4)
        results.upper_central_ci_cohens_d = round(ci_upper_cohens_d_central, 4)
        results.lower_central_ci_hedges_g = round(ci_lower_hedges_g_central, 4)
        results.upper_central_ci_hedges_g = round(ci_upper_hedges_g_central, 4)
        results.lower_pivotal_ci_cohens_d = round(ci_lower_cohens_d_pivotal, 4)
        results.upper_pivotal_ci_cohens_d = round(ci_upper_cohens_d_pivotal, 4)
        results.lower_pivotal_ci_hedges_g = round(
            ci_lower_hedges_g_pivotal * correction, 4
        )
        results.upper_pivotal_ci_hedges_g = round(
            ci_upper_hedges_g_pivotal * correction, 4
        )
        results.lower_ncp_ci_cohens_d = round(ci_lower_cohens_d_ncp, 4)
        results.upper_ncp_ci_cohens_d = round(ci_upper_cohens_d_ncp, 4)
        results.lower_ncp_ci_hedges_g = round(ci_lower_hedges_g_ncp, 4)
        results.upper_ncp_ci_hedges_g = round(ci_upper_hedges_g_ncp, 4)
        results.correction_factor = round(correction, 4)

        return results

    @staticmethod
    def one_sample_from_params(params: dict) -> OneSampleTResults:
        """
        Calculate the one-sample t-test results from given parameters.

        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - "Population Mean" (float): The mean of the population.
            - "Mean Sample" (float): The mean of the sample.
            - "Standard Deviation Sample" (float): The standard deviation of the sample.
            - "Sample Size" (int): The size of the sample.
            - "Confidence Level" (float): The confidence level as a percentage (e.g., 95 for 95%).

        Returns
        -------
        OneSampleTResults
            An object containing all calculated statistics and results
        """
        # Set params
        population_mean = params["Population Mean"]
        sample_mean = params["Mean Sample"]
        sample_sd = params["Standard Deviation Sample"]
        sample_size = params["Sample Size"]
        confidence_level_percentage = params["Confidence Level"]

        # Create results object
        results = OneSampleTResults()

        # Calculation
        confidence_level = confidence_level_percentage / 100
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

        # Set results
        results.cohens_d = round(cohens_d, 4)
        results.hedges_g = round(hedges_g, 4)
        results.t_score = round(t_score, 4)
        results.df = round(df, 4)
        results.p_value = round(p_value, 4)
        results.standardizer_cohens_d = round(sample_sd, 4)
        results.standardizer_hedges_g = round(sample_sd / correction, 4)
        results.standard_error_mean = round(standard_error, 4)
        results.standard_error_cohens_d_true = round(standard_error_cohens_d_true, 4)
        results.standard_error_cohens_d_morris = round(
            standard_error_cohens_d_morris, 4
        )
        results.standard_error_cohens_d_hedges = round(
            standard_error_cohens_d_hedges, 4
        )
        results.standard_error_cohens_d_hedges_olkin = round(
            standard_error_cohens_d_hedges_olkin, 4
        )
        results.standard_error_cohens_d_mle = round(standard_error_cohens_d_mle, 4)
        results.standard_error_cohens_d_large_n = round(
            standard_error_cohens_d_large_n, 4
        )
        results.standard_error_cohens_d_small_n = round(
            standard_error_cohens_d_small_n, 4
        )
        results.standard_error_hedges_g_true = round(standard_error_hedges_g_true, 4)
        results.standard_error_hedges_g_morris = round(
            standard_error_hedges_g_morris, 4
        )
        results.standard_error_hedges_g_hedges = round(
            standard_error_hedges_g_hedges, 4
        )
        results.standard_error_hedges_g_hedges_olkin = round(
            standard_error_hedges_g_hedges_olkin, 4
        )
        results.standard_error_hedges_g_mle = round(standard_error_hedges_g_mle, 4)
        results.standard_error_hedges_g_large_n = round(
            standard_error_hedges_g_large_n, 4
        )
        results.standard_error_hedges_g_small_n = round(
            standard_error_hedges_g_small_n, 4
        )
        results.sample_mean = round(sample_mean, 4)
        results.population_mean = round(population_mean, 4)
        results.means_difference = round(sample_mean - population_mean, 4)
        results.sample_size = round(sample_size, 4)
        results.sample_sd = round(sample_sd, 4)
        results.lower_central_ci_cohens_d = round(ci_lower_cohens_d_central, 4)
        results.upper_central_ci_cohens_d = round(ci_upper_cohens_d_central, 4)
        results.lower_ncp_ci_cohens_d = round(ci_lower_cohens_d_ncp, 4)
        results.upper_ncp_ci_cohens_d = round(ci_upper_cohens_d_ncp, 4)
        results.lower_pivotal_ci_cohens_d = round(ci_lower_cohens_d_pivotal, 4)
        results.upper_pivotal_ci_cohens_d = round(ci_upper_cohens_d_pivotal, 4)
        results.lower_pivotal_ci_hedges_g = round(
            ci_lower_hedges_g_pivotal * correction, 4
        )
        results.upper_pivotal_ci_hedges_g = round(
            ci_upper_hedges_g_pivotal * correction, 4
        )
        results.lower_central_ci_hedges_g = round(ci_lower_hedges_g_central, 4)
        results.upper_central_ci_hedges_g = round(ci_upper_hedges_g_central, 4)
        results.lower_ncp_ci_hedges_g = round(ci_lower_hedges_g_ncp, 4)
        results.upper_ncp_ci_hedges_g = round(ci_upper_hedges_g_ncp, 4)
        results.correction_factor = round(correction, 4)
        return results

    # Things to consider

    # 1. Using a different default for CI - maybe switch to the NCP's one
    # 2. imporve the Pivotal accuracy to match r functions...
    # 3. One Sample from Data
