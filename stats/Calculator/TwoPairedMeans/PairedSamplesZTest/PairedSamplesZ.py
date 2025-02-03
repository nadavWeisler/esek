"""
This module provides functionality for calculating the effect size for paired samples Z-test.

Functions:
    calculate_central_ci_from_cohens_d_one_sample: Calculate the central confidence intervals for Cohen's d in a one-sample Z-test.

Classes:
    TwoPairedSamplesZ: A class containing static methods for calculating the effect size for paired samples Z-test.
"""

import numpy as np
from scipy.stats import norm


def calculate_central_ci_from_cohens_d_one_sample(
    cohens_d, sample_size, confidence_level
):
    """
    Calculate the central confidence intervals for Cohen's d in a one-sample Z-test.

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
        - Standard_error_es (float): Standard error of the effect size.
    """
    Standard_error_es = np.sqrt((1 / sample_size) + ((cohens_d**2 / (2 * sample_size))))
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = (
        cohens_d - Standard_error_es * z_critical_value,
        cohens_d + Standard_error_es * z_critical_value,
    )
    return ci_lower, ci_upper, Standard_error_es


class TwoPairedSamplesZ:
    """
    A class containing static methods for calculating the effect size for paired samples Z-test.

    This class includes the following static methods:
    - two_dependent_samples_from_z_score: Calculate the two dependent samples Z-test results from a given Z-score.
    - two_dependent_samples_from_params: Calculate the two dependent samples Z-test results from given parameters.
    - two_dependent_samples_from_data: Calculate the two dependent samples Z-test results from given data.
    """

    @staticmethod
    def two_dependent_samples_from_z_score(params: dict) -> dict:
        """
        Calculate the two dependent samples Z-test results from a given Z-score.

        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - "Z-score" (float): The Z-score value.
            - "Number of Pairs" (int): The number of paired samples.
            - "Confidence Level" (float): The confidence level as a percentage (e.g., 95 for 95%).

        Returns
        -------
        dict
            A dictionary containing the calculated results:
            - "Cohen's d" (float): The calculated Cohen's d effect size.
            - "Z-score" (float): The Z-score value.
            - "p-value" (float): The p-value.
            - "Cohen's d CI Lower" (float): Lower bound of the confidence interval for Cohen's d.
            - "Cohen's d CI Upper" (float): Upper bound of the confidence interval for Cohen's d.
            - "Standard Error of Cohen's d" (float): Standard error of Cohen's d.
            - "Statistical Line" (str): Formatted statistical line.
        """
        # Get params
        z_score = params["Z-score"]
        sample_size = params["Number of Pairs"]
        confidence_level_percentages = params["Confidence Level"]

        # Calculate
        confidence_level = confidence_level_percentages / 100
        p_value = min(float(norm.sf((abs(z_score))) * 2), 0.99999)
        cohens_d = z_score / np.sqrt(sample_size)
        ci_lower, ci_upper, Standard_error_es = (
            calculate_central_ci_from_cohens_d_one_sample(
                cohens_d, sample_size, confidence_level
            )
        )

        # Create results
        results = {}
        results["Cohen's d"] = round(cohens_d, 4)
        results["Z-score"] = round(z_score, 4)
        results["p-value"] = round(p_value, 4)
        results["Cohen's d CI Lower"] = round(ci_lower, 4)
        results["Cohen's d CI Upper"] = round(ci_upper, 4)
        results["Standard Error of Cohen's d"] = round(Standard_error_es, 4)
        results["Statistical Line"] = (
            f" \033[3mz\033[0m = {z_score:.3f}, \033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}, Cohen's d = {cohens_d:.3f}, {confidence_level_percentages}% CI [{round(ci_lower,3)}, {round(ci_upper,3)}]"
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
    def two_dependent_samples_from_params(params: dict) -> dict:
        """
        Calculate the two dependent samples Z-test results from given parameters.

        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - "Difference in the Population" (float): The mean difference in the population.
            - "Standard Deviation of the Difference in the Population" (float): The standard deviation of the difference in the population.
            - "Sample 1 Mean" (float): The mean of sample 1.
            - "Sample 2 Mean" (float): The mean of sample 2.
            - "Number of Pairs" (int): The number of paired samples.
            - "Confidence Level" (float): The confidence level as a percentage (e.g., 95 for 95%).

        Returns
        -------
        dict
            A dictionary containing the calculated results:
            - "Cohen's d" (float): The calculated Cohen's d effect size.
            - "Z-score" (float): The Z-score value.
            - "p-value" (float): The p-value.
            - "Standard Error of the Mean" (float): Standard error of the mean.
            - "Cohen's d CI Lower" (float): Lower bound of the confidence interval for Cohen's d.
            - "Cohen's d CI Upper" (float): Upper bound of the confidence interval for Cohen's d.
            - "Standard Error of Cohen's d" (float): Standard error of Cohen's d.
            - "Statistical Line" (str): Formatted statistical line.
        """
        # Set params
        population_mean_difference = params["Difference in the Population"]
        population_sd_diff = params[
            "Standard Deviation of the Difference in the Population"
        ]
        sample_mean_1 = params["Sample 1 Mean"]
        sample_mean_2 = params["Sample 2 Mean"]
        sample_size = params["Number of Pairs"]
        confidence_level_percentages = params["Confidence Level"]

        # Calculate
        confidence_level = confidence_level_percentages / 100
        mean_Standard_error = population_sd_diff / np.sqrt(sample_size)
        Sample_Mean_Diff = sample_mean_1 - sample_mean_2
        z_score = (Sample_Mean_Diff - population_mean_difference) / mean_Standard_error
        cohens_d = (Sample_Mean_Diff - population_mean_difference) / population_sd_diff
        p_value = min(float(norm.sf((abs(z_score))) * 2), 0.99999)
        ci_lower, ci_upper, Standard_error_es = (
            calculate_central_ci_from_cohens_d_one_sample(
                cohens_d, sample_size, confidence_level
            )
        )

        # Set Results
        results = {}
        results["Cohen's d"] = round(cohens_d, 4)
        results["Z-score"] = round(z_score, 4)
        results["p-value"] = round(p_value, 15)
        results["Standard Error of the Mean"] = round(mean_Standard_error, 4)
        results["Cohen's d CI Lower"] = round(ci_lower, 4)
        results["Cohen's d CI Upper"] = round(ci_upper, 4)
        results["Standard Error of Cohen's d"] = round(Standard_error_es, 4)
        results["Statistical Line"] = (
            f" \033[3mz\033[0m = {z_score:.3f}, \033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}, Cohen's d = {cohens_d:.3f}, {confidence_level_percentages}% CI [{round(ci_lower,3)}, {round(ci_upper,3)}]"
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
    def two_dependent_samples_from_data(params: dict) -> dict:
        """
        Calculate the two dependent samples Z-test results from given data.

        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - "column_1" (array-like): The data for sample 1.
            - "column_2" (array-like): The data for sample 2.
            - "Difference in the Population" (float): The mean difference in the population.
            - "Standard Deviation of the Difference in the Population" (float): The standard deviation of the difference in the population.
            - "Confidence Level" (float): The confidence level as a percentage (e.g., 95 for 95%).

        Returns
        -------
        dict
            A dictionary containing the calculated results:
            - "Cohen's d" (float): The calculated Cohen's d effect size.
            - "Z-score" (float): The Z-score value.
            - "p-value" (float): The p-value.
            - "Standard Error of the Mean" (float): Standard error of the mean.
            - "Cohen's d CI Lower" (float): Lower bound of the confidence interval for Cohen's d.
            - "Cohen's d CI Upper" (float): Upper bound of the confidence interval for Cohen's d.
            - "Standard Error of Cohen's d" (float): Standard error of Cohen's d.
            - "Mean Sample 1" (float): Mean of sample 1.
            - "Mean Sample 2" (float): Mean of sample 2.
            - "Standard Deviation Sample 1" (float): Standard deviation of sample 1.
            - "Standard Deviation Sample 2" (float): Standard deviation of sample 2.
            - "Sample Size 1" (int): Size of sample 1.
            - "Sample Size 2" (int): Size of sample 2.
            - "Statistical Line" (str): Formatted statistical line.
        """
        # Get params
        column_1 = params["column_1"]
        column_2 = params["column_2"]
        population_diff = params["Difference in the Population"]
        population_diff_sd = params[
            "Standard Deviation of the Difference in the Population"
        ]
        confidence_level_percentages = params["Confidence Level"]

        # Calculate
        confidence_level = confidence_level_percentages / 100
        sample_mean_1 = np.mean(column_1)
        sample_mean_2 = np.mean(column_2)
        sample_sd_1 = np.std(column_1, ddof=1)
        sample_sd_2 = np.std(column_2, ddof=1)

        diff_mean = (sample_mean_1 - sample_mean_2) - population_diff

        sample_size = len(column_1)
        Standard_error = population_diff_sd / (np.sqrt(sample_size))

        # Calculate the z-statistic
        z_score = diff_mean / Standard_error
        cohens_d = (
            diff_mean
        ) / population_diff_sd  # This is the effect size for one sample z-test Cohen's d
        p_value = min(float(norm.sf((abs(z_score))) * 2), 0.99999)
        ci_lower, ci_upper, Standard_error_es = (
            calculate_central_ci_from_cohens_d_one_sample(
                cohens_d, sample_size, confidence_level
            )
        )

        # Set results
        results = {}
        results["Cohen's d"] = round(cohens_d, 4)
        results["Z-score"] = round(z_score, 4)
        results["p-value"] = round(p_value, 4)
        results["Standard Error of the Mean"] = round(Standard_error, 4)
        results["Cohen's d CI Lower"] = round(ci_lower, 4)
        results["Cohen's d CI Upper"] = round(ci_upper, 4)
        results["Standard Error of Cohen's d"] = round(Standard_error_es, 4)
        results["Mean Sample 1"] = round(sample_mean_1, 4)
        results["Mean Sample 2"] = round(sample_mean_2, 4)
        results["Standard Deviation Sample 1"] = round(sample_sd_1, 4)
        results["Standard Deviation Sample 2"] = round(sample_sd_2, 4)
        results["Sample Size 1"] = round(sample_size, 4)
        results["Sample Size 2"] = round(sample_size, 4)
        results["Statistical Line"] = (
            f" \033[3mz\033[0m = {z_score:.3f}, \033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}, Cohen's d = {cohens_d:.3f}, {confidence_level_percentages}% CI [{round(ci_lower,3)}, {round(ci_upper,3)}]"
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
