"""
This module implements the Two Paired Z-test for comparing two paired samples.
It provides methods to calculate the test results from a Z-score, sample parameters,
or sample data. The results include Cohen's d, z-score, p-value, and confidence intervals for the effect size.
The TwoPairedZResults class encapsulates the results of the test and includes attributes such as
sample means, standard deviations, and sample sizes for both groups.
It is designed to be used in statistical analysis where paired samples are compared,
such as in pre-test and post-test scenarios.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.stats import norm
from ...interfaces import AbstractTest
from ...results import CohenD
from ...utility import central_ci_from_cohens_d


@dataclass
class TwoPairedZResults:
    """
    a dataclass for the results of a two paired Z-test.
    """

    cohens_d: Optional[CohenD] = None
    z_score: Optional[float] = None
    p_value: Optional[float] = None
    difference_standard_error: Optional[float] = None
    sample_mean_1: Optional[float] = None
    sample_mean_2: Optional[float] = None
    difference_population_sd: Optional[float] = None
    sample_sd_1: Optional[float] = None
    sample_sd_2: Optional[float] = None
    difference_between_means: Optional[float] = None
    difference_sd: Optional[float] = None
    sample_size_1: Optional[float] = None
    sample_size_2: Optional[float] = None


class TwoPairedZTests(AbstractTest):
    """
    A class for performing two paired Z-tests.
    This class provides methods to calculate Z-test results from a Z-score,
    from sample parameters, and from sample data.
    It encapsulates the results in a TwoPairedZResults object, which includes
    Cohen's d, z-score, p-value, and confidence intervals for the effect size.

    Methods:
        - from_z_score: Calculate the two paired Z-test results from a given Z-score.
        - from_parameters: Calculate the two paired Z-test results from sample parameters.
        - from_data: Calculate the two paired Z-test results from given sample data.
    """

    @staticmethod
    def from_z_score(
        z_score: float, sample_size: float, confidence_level: float
    ) -> TwoPairedZResults:
        """
        Calculate the two paired Z-test results from a given Z-score.
        This method computes the effect size (Cohen's d), p-value, and confidence intervals
        for the effect size based on the provided Z-score and sample size.
        It returns an instance of TwoPairedZResults containing these values.
        """

        p_value = min(float(norm.sf((abs(z_score))) * 2), 0.99999)
        cohens_d = z_score / np.sqrt(sample_size)
        ci_lower, ci_upper, standard_error_es = central_ci_from_cohens_d(
            cohens_d, sample_size, confidence_level
        )
        cohens_d = CohenD(
            value=round(cohens_d, 4),
            ci_lower=round(ci_lower, 4),
            ci_upper=round(ci_upper, 4),
            standard_error=round(standard_error_es, 4),
        )

        results = TwoPairedZResults()
        results.cohens_d = cohens_d
        results.z_score = round(z_score, 4)
        results.p_value = round(p_value, 4)

        return results

    @staticmethod
    def from_parameters(
        sample_mean_1: float,
        sample_mean_2: float,
        sample_size: float,
        population_diff: float,
        population_diff_sd: float,
        confidence_level: float,
    ) -> TwoPairedZResults:
        """
        Calculate the two paired Z-test results from sample parameters.
        This method computes the effect size (Cohen's d), z-score, p-value,
        and confidence intervals for the effect size based on the provided sample means,
        sample size, population difference, and population standard deviation of the difference.
        It returns an instance of TwoPairedZResults containing these values.
        """
        mean_Standard_error = population_diff_sd / np.sqrt(sample_size)
        sample_mean_diff = sample_mean_1 - sample_mean_2
        z_score = (sample_mean_diff - population_diff) / mean_Standard_error
        cohens_d = (sample_mean_diff - population_diff) / population_diff_sd
        p_value = min(float(norm.sf((abs(z_score))) * 2), 0.99999)
        ci_lower, ci_upper, standard_error_es = central_ci_from_cohens_d(
            cohens_d, sample_size, confidence_level
        )

        cohens_d = CohenD(
            value=round(cohens_d, 4),
            ci_lower=round(ci_lower, 4),
            ci_upper=round(ci_upper, 4),
            standard_error=round(standard_error_es, 4),
        )

        results = TwoPairedZResults()
        results.cohens_d = cohens_d
        results.z_score = round(z_score, 4)
        results.p_value = round(p_value, 15)
        results.difference_standard_error = round(mean_Standard_error, 4)

        return results

    @staticmethod
    def from_data(
        columns: list,
        population_diff: float,
        population_diff_sd: float,
        confidence_level: float,
    ) -> TwoPairedZResults:
        """
        Calculate the two paired Z-test results from given sample data.
        This method computes the effect size (Cohen's d), z-score, p-value,
        and confidence intervals for the effect size based on the provided sample data,
        population difference, and population standard deviation of the difference.
        It returns an instance of TwoPairedZResults containing these values.
        """
        sample_mean_1 = np.mean(columns[0])
        sample_mean_2 = np.mean(columns[1])
        sample_sd_1 = np.std(columns[0], ddof=1)
        sample_sd_2 = np.std(columns[1], ddof=1)

        diff_mean = (sample_mean_1 - sample_mean_2) - population_diff

        sample_size = len(columns[0])
        standard_error = population_diff_sd / (np.sqrt(sample_size))

        z_score = diff_mean / standard_error
        cohens_d = (diff_mean) / population_diff_sd
        p_value = min(float(norm.sf((abs(z_score))) * 2), 0.99999)
        ci_lower, ci_upper, standard_error_es = central_ci_from_cohens_d(
            cohens_d, sample_size, confidence_level
        )

        cohens_d = CohenD(
            value=round(cohens_d, 4),
            ci_lower=round(ci_lower, 4),
            ci_upper=round(ci_upper, 4),
            standard_error=round(standard_error_es, 4),
        )

        results = TwoPairedZResults()
        results.cohens_d = cohens_d
        results.z_score = round(z_score, 4)
        results.p_value = round(p_value, 4)
        results.difference_standard_error = round(standard_error, 4)
        results.sample_mean_1 = round(sample_mean_1, 4)
        results.sample_mean_2 = round(sample_mean_2, 4)
        results.sample_sd_1 = round(float(sample_sd_1), 4)
        results.sample_sd_2 = round(float(sample_sd_2), 4)
        results.sample_size_1 = round(sample_size, 4)
        results.sample_size_2 = round(sample_size, 4)
        results.difference_between_means = round(diff_mean, 4)
        results.difference_population_sd = round(population_diff_sd, 4)
        results.difference_sd = None  # TODO: add

        return results
