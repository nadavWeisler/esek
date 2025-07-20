"""
This module provides functionality for calculating the Common Language Effect Size (CLES) for one sample t-tests.

Classes:
    OneSampleCLESResults: A class containing results from CLES calculations
    OneSampleTTest: A class containing static methods for calculating the Common Language Effect Size (CLES) and other related statistics.

Methods:
    pivotal_ci_t: Calculate the pivotal confidence intervals for a t-score.
    calculate_central_ci_from_cohens_d_one_sample_t_test: Calculate the central confidence intervals for Cohen's d in a one-sample t-test.
    CI_NCP_one_Sample: Calculate the Non-Central Parameter (NCP) confidence intervals for a one-sample t-test.
    density: Calculate the density function for a given value.
    area_under_function: Calculate the area under a given function using numerical integration.
    WinsorizedVariance: Calculate the Winsorized variance of a sample.
    WinsorizedCorrelation: Calculate the Winsorized correlation between two samples.
"""

import math
from typing import Optional
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm, nct, t, trim_mean
from ...utils import interfaces
from ...utils import res


@dataclass
class OneSampleCLESResults:
    """Class for storing one-sample CLES test results"""

    sample_size: Optional[int] = None
    population_mean: Optional[float] = None
    sample_mean: Optional[float] = None
    sample_sd: Optional[float] = None
    standard_error: Optional[float] = None
    difference_between_means: Optional[float] = None
    t_score: Optional[float] = None
    degrees_of_freedom: Optional[float] = None
    p_value: Optional[float] = None
    cohens_d = res.CohensDCLES
    hedges_g = res.HedgesGCLES
    robust_t: Optional[res.YuensRobustT] = None


def pivotal_ci_t(t_score, df, sample_size, confidence_level):
    """
    Calculate the pivotal confidence intervals for a t-score.

    Parameters
    ----------
    t_Score : float
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
        - lower_ci (float): Lower bound of the pivotal confidence interval.
        - upper_ci (float): Upper bound of the pivotal confidence interval.
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


def calculate_central_ci_from_cohens_d_one_sample_t_test(
    cohens_d, sample_size, confidence_level
):
    """
    Calculate the central confidence intervals for Cohen's d in a one-sample t-test.

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
        - ci_lower (float): Lower bound of the central confidence interval.
        - ci_upper (float): Upper bound of the central confidence interval.
        - standard_error_es (float): Standard error of the effect size.
    """
    df = sample_size - 1
    correction_factor = math.exp(
        math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2)
    )
    standard_error_es = np.sqrt(
        (df / (df - 2)) * (1 / sample_size) * (1 + cohens_d**2 * sample_size)
        - (cohens_d**2 / correction_factor**2)
    )
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = (
        cohens_d - standard_error_es * z_critical_value,
        cohens_d + standard_error_es * z_critical_value,
    )
    return ci_lower, ci_upper, standard_error_es


def ci_ncp_one_sample(effect_size, sample_size, confidence_level):
    """
    Calculate the Non-Central Parameter (NCP) confidence intervals for a one-sample t-test.

    Parameters
    ----------
    Effect_Size : float
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


def density(x):
    """
    Calculate the density function for a given value.

    Parameters
    ----------
    x : float
        The input value.

    Returns
    -------
    float
        The density value.
    """
    x = np.array(x)
    return x**2 * norm.pdf(x)


def area_under_function(
    f, a, b, *args, function_a=None, function_b=None, limit=10, eps=1e-5
):
    """
    Calculate the area under a given function using numerical integration.

    Parameters
    ----------
    f : function
        The function to integrate.
    a : float
        The lower bound of the integration.
    b : float
        The upper bound of the integration.
    *args : tuple
        Additional arguments to pass to the function.
    function_a : float, optional
        The function value at the lower bound.
    function_b : float, optional
        The function value at the upper bound.
    limit : int, optional
        The maximum number of recursive calls (default is 10).
    eps : float, optional
        The tolerance for stopping the recursion (default is 1e-5).

    Returns
    -------
    float
        The area under the function.
    """
    if function_a is None:
        function_a = f(a, *args)
    if function_b is None:
        function_b = f(b, *args)
    midpoint = (a + b) / 2
    f_midpoint = f(midpoint, *args)
    area_trapezoidal = ((function_a + function_b) * (b - a)) / 2
    area_simpson = ((function_a + 4 * f_midpoint + function_b) * (b - a)) / 6
    if abs(area_trapezoidal - area_simpson) < eps or limit == 0:
        return area_simpson
    return area_under_function(
        f,
        a,
        midpoint,
        *args,
        function_a=function_a,
        function_b=f_midpoint,
        limit=limit - 1,
        eps=eps,
    ) + area_under_function(
        f,
        midpoint,
        b,
        *args,
        function_a=f_midpoint,
        function_b=function_b,
        limit=limit - 1,
        eps=eps,
    )


def calculate_winsorized_variance(x, trimming_level=0.2):
    """
    Calculate the Winsorized variance of a sample.

    Parameters
    ----------
    x : array-like
        The input sample.
    trimming_level : float, optional
        The trimming level (default is 0.2).

    Returns
    -------
    float
        The Winsorized variance.
    """
    y = np.sort(x)
    n = len(x)
    ibot = int(np.floor(trimming_level * n)) + 1
    itop = n - ibot + 1
    xbot = y[ibot - 1]
    xtop = y[itop - 1]
    y = np.where(y <= xbot, xbot, y)
    y = np.where(y >= xtop, xtop, y)
    winvar = np.std(y, ddof=1) ** 2
    return winvar


def calculate_winsorized_correlation(x, y, trimming_level=0.2):
    """
    Calculate the Winsorized correlation between two samples.

    Parameters
    ----------
    x : array-like
        The first input sample.
    y : array-like
        The second input sample.
    trimming_level : float, optional
        The trimming level (default is 0.2).

    Returns
    -------
    dict
        A dictionary containing the Winsorized correlation, covariance, p-value, sample size, and test statistic.
    """
    sample_size = len(x)
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    trimming_size = int(np.floor(trimming_level * sample_size)) + 1
    x_lower = x_sorted[trimming_size - 1]
    x_upper = x_sorted[sample_size - trimming_size]
    y_lower = y_sorted[trimming_size - 1]
    y_upper = y_sorted[sample_size - trimming_size]
    x_winzsorized = np.clip(x, x_lower, x_upper)
    y_winzsorized = np.clip(y, y_lower, y_upper)
    winsorized_correlation = np.corrcoef(x_winzsorized, y_winzsorized)[0, 1]
    winsorized_covariance = np.cov(x_winzsorized, y_winzsorized)[0, 1]
    test_statistic = winsorized_correlation * np.sqrt(
        (sample_size - 2) / (1 - winsorized_correlation**2)
    )
    Number_of_trimmed_values = int(np.floor(trimming_level * sample_size))
    p_value = 2 * (
        1
        - t.cdf(np.abs(test_statistic), sample_size - 2 * Number_of_trimmed_values - 2)
    )
    return {
        "cor": winsorized_correlation,
        "cov": winsorized_covariance,
        "p.value": p_value,
        "n": sample_size,
        "test_statistic": test_statistic,
    }


class OneSampleClesTest(interfaces.AbstractTest):
    """
    A class containing static methods for calculating the Common Language Effect Size (CLES) and other related statistics.

    This class includes the following static methods:
    - one_sample_from_t_score: Calculate the one-sample t-test results from a given t-score.
    - one_sample_from_params: Calculate the one-sample t-test results from given parameters.
    - one_sample_from_data: Calculate the one-sample t-test results from given sample data.
    - Robust_One_Sample: Calculate the robust one-sample t-test results.
    """

    @staticmethod
    def from_score(t_score: float, sample_size: int, confidence_level: float = 0.95):
        """
        Calculate the one-sample CLES test results from a given t-score.
        This method computes the Common Language Effect Size (CLES) for a one-sample t-test,
        including Cohen's d, Hedges' g, and the t-score.
        Parameters
        ----------
        t_score : float
            The t-score value.
        sample_size : int
            The size of the sample.
        confidence_level : float, optional
            The confidence level for the confidence intervals (default is 0.95).
        Returns
        -------
        OneSampleCLESResults
            An instance of OneSampleCLESResults containing the results of the one-sample CLES test.
        """
        df = sample_size - 1
        cohens_d = t_score / np.sqrt(
            df
        )  # This is Cohen's d and it is calculated based on the sample's standard deviation
        correction = math.exp(
            math.lgamma(df / 2)
            - math.log(math.sqrt(df / 2))
            - math.lgamma((df - 1) / 2)
        )
        hedges_g = correction * cohens_d
        cles_d = norm.cdf(cohens_d) * 100
        cles_g = norm.cdf(hedges_g) * 100

        (
            ci_lower_cohens_d_central,
            ci_upper_cohens_d_central,
            standrat_error_cohens_d,
        ) = calculate_central_ci_from_cohens_d_one_sample_t_test(
            cohens_d, sample_size, confidence_level
        )
        (
            ci_lower_hedges_g_central,
            ci_upper_hedges_g_central,
            standrat_error_hedges_g,
        ) = calculate_central_ci_from_cohens_d_one_sample_t_test(
            hedges_g, sample_size, confidence_level
        )
        ci_lower_cohens_d_ncp, ci_upper_cohens_d_ncp = ci_ncp_one_sample(
            cohens_d, sample_size, confidence_level
        )
        ci_lower_hedges_g_ncp, ci_upper_hedges_g_ncp = ci_ncp_one_sample(
            hedges_g, sample_size, confidence_level
        )

        results = OneSampleCLESResults()
        results.cohens_d = res.CohensDCLES(
            value=cles_d,
            lower_central_ci_lower=norm.cdf(ci_lower_cohens_d_central) * 100,
            central_ci_upper=norm.cdf(ci_upper_cohens_d_central) * 100,
            non_central_ci_lower=norm.cdf(ci_lower_cohens_d_ncp) * 100,
            non_central_ci_upper=norm.cdf(ci_upper_cohens_d_ncp) * 100,
            standard_error=standrat_error_cohens_d,
        )
        results.hedges_g = res.HedgesGCLES(
            value=cles_g,
            central_ci_lower=norm.cdf(ci_lower_hedges_g_central) * 100,
            central_ci_upper=norm.cdf(ci_upper_hedges_g_central) * 100,
            non_central_ci_lower=norm.cdf(ci_lower_hedges_g_ncp) * 100,
            non_central_ci_upper=norm.cdf(ci_upper_hedges_g_ncp) * 100,
            standard_error=standrat_error_hedges_g,
        )

        return results

    @staticmethod
    def from_params(
        population_mean: float,
        sample_size: int,
        sample_sd: float,
        sample_mean: float,
        confidence_level: float = 0.95,
    ) -> OneSampleCLESResults:
        """
        Calculate the one-sample CLES test results from given population and sample parameters.
        This method computes the Common Language Effect Size (CLES) for a one-sample t-test,
        including Cohen's d, Hedges' g, and the t-score.
        It also calculates confidence intervals for the effect sizes.
        Parameters
        ----------
        population_mean : float
            The mean of the population.
        sample_size : int
            The size of the sample.
        sample_sd : float
            The standard deviation of the sample.
        sample_mean : float
            The mean of the sample.
        confidence_level : float, optional
            The confidence level for the confidence intervals (default is 0.95).
        Returns
        -------
        OneSampleCLESResults
            An instance of OneSampleCLESResults containing the results of the one-sample CLES test.
        """
        # Calculation
        df = sample_size - 1
        standard_error = sample_sd / np.sqrt(
            df
        )  # This is the standard error of mean's estimate in one sample t-test
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
        cles_d = norm.cdf(cohens_d) * 100
        cles_g = norm.cdf(hedges_g) * 100
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)
        (
            ci_lower_cohens_d_central,
            ci_upper_cohens_d_central,
            standrat_error_cohens_d,
        ) = calculate_central_ci_from_cohens_d_one_sample_t_test(
            cohens_d, sample_size, confidence_level
        )
        (
            ci_lower_hedges_g_central,
            ci_upper_hedges_g_central,
            standrat_error_hedges_g,
        ) = calculate_central_ci_from_cohens_d_one_sample_t_test(
            hedges_g, sample_size, confidence_level
        )
        ci_lower_cohens_d_ncp, ci_upper_cohens_d_ncp = ci_ncp_one_sample(
            cohens_d, sample_size, confidence_level
        )
        ci_lower_hedges_g_ncp, ci_upper_hedges_g_ncp = ci_ncp_one_sample(
            hedges_g, sample_size, confidence_level
        )
        # Create results object
        results = OneSampleCLESResults()
        results.sample_size = sample_size
        results.population_mean = population_mean
        results.sample_mean = sample_mean
        results.sample_sd = sample_sd
        results.standard_error = standard_error
        results.difference_between_means = sample_mean - population_mean
        results.t_score = t_score
        results.degrees_of_freedom = df
        results.p_value = p_value
        results.cohens_d = res.CohensDCLES(
            value=cles_d,
            lower_central_ci_lower=norm.cdf(ci_lower_cohens_d_central) * 100,
            central_ci_upper=norm.cdf(ci_upper_cohens_d_central) * 100,
            non_central_ci_lower=norm.cdf(ci_lower_cohens_d_ncp) * 100,
            non_central_ci_upper=norm.cdf(ci_upper_cohens_d_ncp) * 100,
            standard_error=standrat_error_cohens_d,
        )
        results.hedges_g = res.HedgesGCLES(
            value=cles_g,
            central_ci_lower=norm.cdf(ci_lower_hedges_g_central) * 100,
            central_ci_upper=norm.cdf(ci_upper_hedges_g_central) * 100,
            non_central_ci_lower=norm.cdf(ci_lower_hedges_g_ncp) * 100,
            non_central_ci_upper=norm.cdf(ci_upper_hedges_g_ncp) * 100,
            standard_error=standrat_error_hedges_g,
        )

        return results

    @staticmethod
    def from_data(
        column_1: list,
        population_mean: float,
        confidence_level: float = 0.95,
        trimming_level: float = 0.2,
        reps=1000
    ) -> OneSampleCLESResults:
        """
        Calculate the one-sample CLES test results from given sample data.
        This method computes the Common Language Effect Size (CLES) for a one-sample t-test,
        including Cohen's d, Hedges' g, and the t-score.
        It also calculates confidence intervals for the effect sizes and performs robust calculations.
        Parameters
        ----------
        column_1 : list
            The sample data as a list of numerical values.
        population_mean : float
            The mean of the population.
        confidence_level : float, optional
            The confidence level for the confidence intervals (default is 0.95).
        trimming_level : float, optional
            The trimming level for robust calculations (default is 0.2).
        reps : int, optional
            The number of bootstrap repetitions for robust effect size calculations (default is 1000).
        Returns
        -------
        OneSampleCLESResults
            An instance of OneSampleCLESResults containing the results of the one-sample CLES test.
        """
        sample_mean = np.mean(column_1)
        sample_sd = np.std(column_1, ddof=1)
        sample_size = len(column_1)
        df = sample_size - 1  # Degrees of freedom one sample t-test
        standard_error = sample_sd / np.sqrt(
            df
        )  # This is the standard error of mean's estimate in one sample t-test
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
            standrat_error_cohens_d,
        ) = calculate_central_ci_from_cohens_d_one_sample_t_test(
            cohens_d, sample_size, confidence_level
        )
        (
            ci_lower_hedges_g_central,
            ci_upper_hedges_g_central,
            standrat_error_hedges_g,
        ) = calculate_central_ci_from_cohens_d_one_sample_t_test(
            hedges_g, sample_size, confidence_level
        )
        ci_lower_cohens_d_ncp, ci_upper_cohens_d_ncp = ci_ncp_one_sample(
            cohens_d, sample_size, confidence_level
        )
        ci_lower_hedges_g_ncp, ci_upper_hedges_g_ncp = ci_ncp_one_sample(
            hedges_g, sample_size, confidence_level
        )
        cles_d = norm.cdf(cohens_d) * 100
        cles_g = norm.cdf(hedges_g) * 100

        difference = np.array(column_1) - population_mean
        correction = np.sqrt(
            area_under_function(
                density, norm.ppf(trimming_level), norm.ppf(1 - trimming_level)
            )
            + 2 * (norm.ppf(trimming_level) ** 2) * trimming_level
        )
        trimmed_mean_1 = trim_mean(column_1, trimming_level)
        winsorized_standard_deviation_1 = np.sqrt(calculate_winsorized_variance(column_1))

        # Algina, Penfield, Kesselman robust effect size (AKP)
        standardizer = np.sqrt(calculate_winsorized_variance(difference, trimming_level))
        trimmed_mean = trim_mean(difference, trimming_level)
        akp_effect_size = correction * (trimmed_mean - population_mean) / standardizer

        # Confidence Intervals for AKP effect size using Bootstrapping
        bootstrap_difference = []
        for _ in range(reps):
            # Generate bootstrap samples
            difference_bootstrap = np.random.choice(
                difference, len(difference), replace=True
            )
            bootstrap_difference.append(difference_bootstrap)

        trimmed_means_of_bootstrap = trim_mean(
            bootstrap_difference, trimming_level, axis=1
        )
        standardizers_of_bootstrap = np.sqrt(
            [
                calculate_winsorized_variance(array, trimming_level)
                for array in bootstrap_difference
            ]
        )
        akp_effect_size_bootstrap = (
            correction
            * (trimmed_means_of_bootstrap - population_mean)
            / standardizers_of_bootstrap
        )
        lower_ci_akp_boot = np.percentile(
            akp_effect_size_bootstrap,
            ((1 - confidence_level) - ((1 - confidence_level) / 2)) * 100,
        )
        upper_ci_akp_boot = np.percentile(
            akp_effect_size_bootstrap,
            ((confidence_level) + ((1 - confidence_level) / 2)) * 100,
        )

        # Yuen Test Statistics
        non_winsorized_sample_size = len(column_1) - 2 * np.floor(
            trimming_level * len(column_1)
        )
        df = non_winsorized_sample_size - 1
        yuen_standard_error = winsorized_standard_deviation_1 / (
            (1 - 2 * trimming_level) * np.sqrt(len(column_1))
        )
        difference_trimmed_means = trimmed_mean_1 - population_mean
        yuen_statistic = difference_trimmed_means / yuen_standard_error
        yuen_p_value = 2 * (1 - t.cdf(np.abs(yuen_statistic), df))

        # Create results object
        results = OneSampleCLESResults()

        results.sample_size = sample_size
        results.population_mean = population_mean
        results.sample_mean = sample_mean
        results.sample_sd = sample_sd
        results.standard_error = standard_error
        results.difference_between_means = sample_mean - population_mean
        results.t_score = t_score
        results.degrees_of_freedom = df
        results.p_value = p_value
        results.cohens_d = res.CohensDCLES(
            value=cles_d,
            lower_central_ci_lower=norm.cdf(ci_lower_cohens_d_central) * 100,
            central_ci_upper=norm.cdf(ci_upper_cohens_d_central) * 100,
            non_central_ci_lower=norm.cdf(ci_lower_cohens_d_ncp) * 100,
            non_central_ci_upper=norm.cdf(ci_upper_cohens_d_ncp) * 100,
            standard_error=standrat_error_cohens_d,
        )
        results.hedges_g = res.HedgesGCLES(
            value=cles_g,
            central_ci_lower=norm.cdf(ci_lower_hedges_g_central) * 100,
            central_ci_upper=norm.cdf(ci_upper_hedges_g_central) * 100,
            non_central_ci_lower=norm.cdf(ci_lower_hedges_g_ncp) * 100,
            non_central_ci_upper=norm.cdf(ci_upper_hedges_g_ncp) * 100,
            standard_error=standrat_error_hedges_g,
        )
        results.robust_t = res.YuensRobustT(
            value=akp_effect_size,
            lower_ci_robust_akp=lower_ci_akp_boot,
            upper_ci_robust_akp=upper_ci_akp_boot,
            trimmed_mean=trimmed_mean_1,
            winsorized_mean=winsorized_standard_deviation_1,
            robust_apk_value=correction,
            degrees_of_freedom=df,
            p_value=yuen_p_value,
            standard_error=yuen_standard_error,
        )

        return results