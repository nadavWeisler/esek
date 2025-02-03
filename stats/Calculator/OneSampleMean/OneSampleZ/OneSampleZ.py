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


class OneSampleZ:
    """
    A class containing static methods for performing one-sample Z-tests from Z-score,
    parameters, and data.

    This class includes the following static methods:
    - one_sample_from_z_score: Calculate the one-sample Z-test results from a given
    Z-score.
    - one_sample_from_parameters: Calculate the one-sample Z-test results from given
    population and sample parameters.
    - one_sample_from_data: Calculate the one-sample Z-test results from given sample
    data.
    """

    @staticmethod
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

        # Set results
        results = {}
        results["Cohen's d"] = round(cohens_d, 4)
        results["Z-score"] = round(z_score, 4)
        results["p-value"] = round(p_value, 4)
        results["Cohen's d CI Lower"] = round(ci_lower, 4)
        results["Cohen's d CI Upper"] = round(ci_upper, 4)
        results["Standard Error of the Effect Size"] = round(standard_error_es, 4)
        results["Statistical Line"] = (
            f" \033[3mz\033[0m = {z_score:.3f}, \033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}, Cohen's d = {cohens_d:.3f}, {confidence_level_percentages}% CI [{ci_lower:.3f},{ci_upper:.3f}]"
        )
        results["Statistical Line"] = (
            results["Statistical Line"]
            .replace(
                f"\033[3mp\033[0m {'= {:.3f}'.format(p_value)}",
                f"\033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}",
            )
            .replace(".000", ".")
            .replace("= 0.", "= .")
            .replace("< 0.", "< .")
        )
        return results

    @staticmethod
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

        # Set Results
        results = {}
        results["Cohen's d"] = round(cohens_d, 4)
        results["Z-score"] = round(z_score, 4)
        results["p-value"] = round(p_value, 4)
        results["Standard Error of the Mean"] = round(mean_standard_error, 4)
        results["Cohen's d CI Lower"] = round(ci_lower, 4)
        results["Cohen's d CI Upper"] = round(ci_upper, 4)
        results["Standard Error of the Effect Size"] = round(standard_error_es, 4)
        results["Statistical Line"] = (
            f" \033[3mz\033[0m = {z_score:.3f}, \033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}, Cohen's d = {cohens_d:.3f}, {confidence_level_percentages}% CI [{ci_lower:.3f},{ci_upper:.3f}]"
        )
        results["Statistical Line"] = (
            results["Statistical Line"]
            .replace(
                f"\033[3mp\033[0m {'= {:.3f}'.format(p_value)}",
                f"\033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}",
            )
            .replace(".000", ".")
            .replace("= 0.", "= .")
            .replace("< 0.", "< .")
        )
        return results

    @staticmethod
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
        dict
            A dictionary containing the calculated results:
            - "Cohen's d" (float): The calculated Cohen's d effect size
            - "Z-score" (float): The Z-score value
            - "p-value" (float): The p-value
            - "Standard Error of the Mean" (float): The standard error of the mean
            - "Cohen's d CI Lower" (float): Lower bound of the confidence interval for Cohen's d
            - "Cohen's d CI Upper" (float): Upper bound of the confidence interval for Cohen's d
            - "Standard Error of the Effect Size" (float): Standard error of the effect size
            - "Sample's Mean" (float): The mean of the sample
            - "Sample's Standard Deviation" (float): The standard deviation of the sample
            - "Difference Between Means" (float): The difference between the population mean and the
            sample mean
            - "Sample Size" (int): The size of the sample
            - "Statistical Line" (str): A formatted string with the statistical results
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

        results = {}
        results["Cohen's d"] = round(cohens_d, 4)
        results["Z-score"] = round(z_score, 4)
        results["p-value"] = round(p_value, 20)
        results["Standard Error of the Mean"] = round(standard_error, 4)
        results["Cohen's d CI Lower"] = round(ci_lower, 4)
        results["Cohen's d CI Upper"] = round(ci_upper, 4)
        results["Standard Error of the Effect Size"] = round(standard_error_es, 4)
        results["Sample's Mean"] = round(sample_mean, 4)
        results["Sample's Standard Deviation"] = round(sample_sd, 4)
        results["Difference Between Means"] = round(diff_mean, 4)
        results["Sample Size"] = round(sample_size, 4)
        results["Statistical Line"] = (
            f" \033[3mz\033[0m = {z_score:.3f}, \033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}, Cohen's d = {cohens_d:.3f}, {confidence_level_percentages}% CI [{ci_lower:.3f},{ci_upper:.3f}]"
        )
        results["Statistical Line"] = (
            results["Statistical Line"]
            .replace(
                f"\033[3mp\033[0m {'= {:.3f}'.format(p_value)}",
                f"\033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}",
            )
            .replace(".000", ".")
            .replace("= 0.", "= .")
            .replace("< 0.", "< .")
        )
        return results

