"""
This module contains functions for performing one-sample Z-tests and calculating
various statistics such as Cohen's d, Z-score, p-value, confidence intervals, and
standard errors.

The module includes the following functions:
- calculate_central_ci_from_cohens_d_one_sample: Calculate the confidence intervals
and standard error for Cohen's d effect size in a one-sample Z-test.
- One_Sample_ZTests: A class containing static methods for performing one-sample
Z-tests from Z-score, parameters, and data.
"""

import numpy as np
from scipy.stats import norm


def calculate_central_ci_from_cohens_d_one_sample(
    cohens_d, sample_size, confidence_level
):
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
    sample_size : int
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
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = (
        cohens_d - standard_error_es * z_critical_value,
        cohens_d + standard_error_es * z_critical_value,
    )
    return ci_lower, ci_upper, standard_error_es

class OneSampleZResults:
    """
    A class to store results from one-sample Z statistical tests.

    This class contains attributes to store various statistical measures including:
    - Effect size (Cohen's d)
    - Z-score and p-value
    - Standard error of the mean
    - Confidence intervals for Cohen's d
    - Standard error of the effect size
    """
    def __init__(self) -> None:
        self.cohens_d: float | None = None
        self.z_score: float | None = None 
        self.p_value: float | None = None
        self.standard_error_of_the_mean: float | None = None
        self.cohens_d_ci_lower: float | None = None
        self.cohens_d_ci_upper: float | None = None
        self.standard_error_of_the_effect_size: float | None = None
        self.sample_mean: float | None = None
        self.sample_sd: float | None = None
        self.difference_between_means: float | None = None
        self.sample_size: float | None = None

def one_sample_from_z_score(params: dict) -> dict:
    """
    Calculate the one-sample Z-test results from a given Z-score.

    Parameters
    ----------
    params : dict
        A dictionary containing the following keys:
        - "Z-score" (float): The Z-score value
        - "Sample Size" (int): The size of the sample
        - "Confidence Level" (float): The confidence level as a percentage (e.g., 95 for 95%)

    Returns
    -------
    dict
        A dictionary containing the calculated results:
        - "Cohen's d" (float): The calculated Cohen's d effect size
        - "Z-score" (float): The Z-score value
        - "p-value" (float): The p-value
        - "Cohen's d CI Lower" (float): Lower bound of the confidence interval for Cohen's d
        - "Cohen's d CI Upper" (float): Upper bound of the confidence interval for Cohen's d
        - "Standard Error of the Effect Size" (float): Standard error of the effect size
        - "Statistical Line" (str): A formatted string with the statistical results
    """

    # Set params
    z_score = params["Z-score"]
    sample_size = params["Sample Size"]
    confidence_level_percentages = params["Confidence Level"]

    # Calculation
    confidence_level = confidence_level_percentages / 100
    p_value = min(float(norm.sf((abs(z_score))) * 2), 0.99999)
    cohens_d = z_score / np.sqrt(sample_size)
    ci_lower, ci_upper, standard_error_es = (
        calculate_central_ci_from_cohens_d_one_sample(
            cohens_d, sample_size, confidence_level
        )
    )

    results = OneSampleZResults()
    results.cohens_d = round(cohens_d, 4)
    results.z_score = round(z_score, 4)
    results.p_value = round(p_value, 4)
    results.cohens_d_ci_lower = round(ci_lower, 4)
    results.cohens_d_ci_upper = round(ci_upper, 4)
    results.standard_error_of_the_effect_size = round(standard_error_es, 4)

    return results

def one_sample_from_parameters(params: dict) -> dict:
    """
    Calculate the one-sample Z-test results from given population and sample parameters.

    Parameters
    ----------
    params : dict
        A dictionary containing the following keys:
        - "Population Mean" (float): The mean of the population
        - "Population Standard Deviation" (float): The standard deviation of the population
        - "Sample's Mean" (float): The mean of the sample
        - "Sample Size" (int): The size of the sample
        - "Confidence Level" (float): The confidence level as a percentage (e.g., 95 for 95%)

    Returns
    -------
    dict
        A dictionary containing the calculated results:
        - "Cohen's d" (float): The calculated Cohen's d effect size
        - "Z-score" (float): The Z-score value
        - "p-value" (float): The p-value
        - "Standard Error of the Mean" (float): The standard error of the mean
        - "Cohen's d CI Lower" (float): Lower bound of the confidence interval for Cohen's d
        - "Cohen's d CI Upper" (float): Upper bound of the confidence interval for Cohen's d
        - "Standard Error of the Effect Size" (float): Standard error of the effect size
        - "Statistical Line" (str): A formatted string with the statistical results
    """

    # Set params
    population_mean = params["Population Mean"]
    population_sd = params["Population Standard Deviation"]
    sample_mean = params["Sample's Mean"]
    sample_size = params["Sample Size"]
    confidence_level_percentages = params["Confidence Level"]

    # Calculation
    confidence_level = confidence_level_percentages / 100
    mean_standard_error = population_sd / np.sqrt(sample_size)
    confidence_level = confidence_level_percentages / 100
    z_score = (population_mean - sample_mean) / mean_standard_error
    cohens_d = (population_mean - sample_mean) / population_sd
    p_value = min(float(norm.sf((abs(z_score))) * 2), 0.99999)
    ci_lower, ci_upper, standard_error_es = (
        calculate_central_ci_from_cohens_d_one_sample(
            cohens_d, sample_size, confidence_level
        )
    )


    # Create results object
    results = OneSampleZResults()
    results.cohens_d = round(cohens_d, 4)
    results.z_score = round(z_score, 4)
    results.p_value = round(p_value, 4)
    results.standard_error_of_the_mean = round(mean_standard_error, 4)
    results.cohens_d_ci_lower = round(ci_lower, 4)
    results.cohens_d_ci_upper = round(ci_upper, 4)
    results.standard_error_of_the_effect_size = round(standard_error_es, 4)
    return results

def one_sample_from_data(params: dict) -> dict:
    """
    Calculate the one-sample Z-test results from given sample data.

    Parameters
    ----------
    params : dict
        A dictionary containing the following keys:
        - "column_1" (list): The sample data
        - "Population Mean" (float): The mean of the population
        - "Population Standard Deviation" (float): The standard deviation of the population
        - "Confidence Level" (float): The confidence level as a percentage (e.g., 95 for 95%)

    Returns
    -------
    OneSampleZResults
        An object containing the calculated results:
        - cohens_d (float): The calculated Cohen's d effect size
        - z_score (float): The Z-score value 
        - p_value (float): The p-value
        - standard_error_of_the_mean (float): The standard error of the mean
        - cohens_d_ci_lower (float): Lower bound of the confidence interval for Cohen's d
        - cohens_d_ci_upper (float): Upper bound of the confidence interval for Cohen's d
        - standard_error_of_the_effect_size (float): Standard error of the effect size
        - sample_mean (float): The mean of the sample
        - sample_sd (float): The standard deviation of the sample
        - difference_between_means (float): The difference between the population mean and the
            sample mean
        - sample_size (int): The size of the sample
    """

    # Set params
    column_1 = params["column_1"]
    population_mean = params["Population Mean"]
    population_sd = params["Popoulation Standard Deviation"]
    confidence_level_percentages = params["Confidence Level"]

    # Calculation
    confidence_level = confidence_level_percentages / 100
    sample_mean = np.mean(column_1)
    sample_sd = np.std(column_1, ddof=1)
    diff_mean = population_mean - sample_mean
    sample_size = len(column_1)
    standard_error = population_sd / (np.sqrt(sample_size))
    z_score = diff_mean / standard_error
    cohens_d = (diff_mean) / population_sd  # This is the effect size for one sample z-test Cohen's d
    p_value = min(float(norm.sf((abs(z_score))) * 2), 0.99999)
    ci_lower, ci_upper, standard_error_es = (
        calculate_central_ci_from_cohens_d_one_sample(
            cohens_d, sample_size, confidence_level
        )
    )

    # Create results object
    results = OneSampleZResults()
    results.cohens_d = round(cohens_d, 4)
    results.z_score = round(z_score, 4)
    results.p_value = round(p_value, 4)
    results.standard_error_of_the_mean = round(standard_error, 4)
    results.cohens_d_ci_lower = round(ci_lower, 4)
    results.cohens_d_ci_upper = round(ci_upper, 4)
    results.standard_error_of_the_effect_size = round(standard_error_es, 4)
    results.sample_mean = round(sample_mean, 4)
    results.sample_sd = round(sample_sd, 4)
    results.difference_between_means = round(diff_mean, 4)
    results.sample_size = round(sample_size, 4)

    return results