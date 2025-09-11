"""
This module provides functionality for calculating the Aparametric effect size using the
sign test for one sample.

Classes:
    AparametricOneSample: A class containing static methods for calculating
        the Aparametric effect size.

Methods:
    ApermetricEffectSizeOneSample: Calculate the Aparametric effect size
        using the sign test for one sample.
"""

import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.stats import norm, rankdata, median_abs_deviation
from ...utils import interfaces, res


# Create results class
@dataclass
class OneSampleAparametricResults:
    """
    A class to store results from one-sample aparametric statistical tests.

    This class contains attributes to store various statistical measures including:
    - General summary statistics (sample size, means, medians etc.)
    - Wilcoxon test statistics (ignoring ties)
    - Rank biserial correlations
    - Confidence intervals
    - Statistical lines in formatted output
    - Pratt test statistics (considering ties)
    """

    sample_size: Optional[int] = None
    median: Optional[int | float] = None
    median_of_absolute_deviation: Optional[int | float] = None
    mean: Optional[int | float] = None
    standard_deviation: Optional[int | float] = None

    how_many_times_sample_is_larger: Optional[int | float] = None
    how_many_times_sample_is_smaller: Optional[int | float] = None
    how_many_ties: Optional[int | float] = None
    sample_size: Optional[int | float] = None
    sample_size_of_non_ties: Optional[int | float] = None

    wilcoxon_matched_pairs_rank_biserial_z_score: Optional[int | float] = None
    wilcoxon_matched_pairs_rank_biserial_p_value: Optional[int | float] = None
    wilcoxon_matched_pairs_rank_biserial_effect_size: Optional[int | float] = None
    wilcoxon_matched_pairs_rank_biserial_confidence_interval: Optional[
        tuple[int | float, int | float]
    ] = None

    wilcoxon_z_based_rank_biserial_correlation_z_score: Optional[int | float] = None
    wilcoxon_z_based_rank_biserial_correlation_p_value: Optional[int | float] = None
    wilcoxon_z_based_rank_biserial_correlation_effect_size: Optional[int | float] = None
    wilcoxon_z_based_rank_biserial_correlation_confidence_interval: Optional[
        tuple[int | float, int | float]
    ] = None

    wilcoxon_z_based_corrected_rank_biserial_correlation_z_score: Optional[
        int | float
    ] = None
    wilcoxon_z_based_corrected_rank_biserial_correlation_p_value: Optional[
        int | float
    ] = None
    wilcoxon_z_based_corrected_rank_biserial_correlation_effect_size: Optional[
        int | float
    ] = None
    wilcoxon_z_based_corrected_rank_biserial_correlation_confidence_interval: Optional[res.ConfidenceInterval] = None

    pratt_matched_pairs_rank_biserial_z_score: Optional[int | float] = None
    pratt_matched_pairs_rank_biserial_p_value: Optional[int | float] = None
    pratt_matched_pairs_rank_biserial_effect_size: Optional[int | float] = None
    pratt_matched_pairs_rank_biserial_confidence_interval: Optional[res.ConfidenceInterval] = None

    pratt_z_based_rank_biserial_correlation_z_score: Optional[int | float] = None
    pratt_z_based_rank_biserial_correlation_p_value: Optional[int | float] = None
    pratt_z_based_rank_biserial_correlation_effect_size: Optional[int | float] = None
    pratt_z_based_rank_biserial_correlation_confidence_interval: Optional[res.ConfidenceInterval] = None

    pratt_z_based_corrected_rank_biserial_correlation_z_score: Optional[int | float] = (
        None
    )
    pratt_z_based_corrected_rank_biserial_correlation_p_value: Optional[int | float] = (
        None
    )
    pratt_z_based_corrected_rank_biserial_correlation_effect_size: Optional[
        int | float
    ] = None
    pratt_z_based_corrected_rank_biserial_correlation_confidence_interval: Optional[res.ConfidenceInterval] = None


class OneSampleAparametric(interfaces.AbstractTest):
    """
    A class to perform one-sample aparametric tests using the sign test.

    This class contains methods to calculate the Aparametric effect size for one sample
    and returns the results in a structured format.
    """
    @staticmethod
    def from_score() -> OneSampleAparametricResults:
        """
        A static method to create results from a score.
        This method is not implemented and will raise a NotImplementedError.
        Raises:
            NotImplementedError: This method is not implemented for OneSampleAparametric.
        """
        raise NotImplementedError(
            "from_score method is not implemented for OneSampleAparametric."
        )

    @staticmethod
    def from_parameters() -> OneSampleAparametricResults:
        """
        A static method to create results from parameters.
        This method is not implemented and will raise a NotImplementedError.
        Raises:
            NotImplementedError: This method is not implemented for OneSampleAparametric.
        """
        raise NotImplementedError(
            "from_parameters method is not implemented for OneSampleAparametric."
        )

    @staticmethod
    def from_data(
        columns: list, population_mean: float, confidence_level: float = 0.95
    ) -> OneSampleAparametricResults:
        """
        Calculate the Aparametric effect size using the sign test for one sample from data.

        Parameters:
        column (list): A list of sample data.
        population_mean (float): The population mean to compare against.
        confidence_level (float): The confidence level as a decimal (default is 0.95).

        Returns:
        OneSampleAparametricResults: An instance of OneSampleAparametricResults containing the results.
        """
        column_1 = columns[0]
        # General Summary Statistics
        sample_median = np.median(column_1)
        sample_mean = np.mean(population_mean)
        sample_standard_deviation_1 = float(np.std(column_1, ddof=1))
        difference = column_1 - population_mean
        # How many times sample is greater than population value
        positive_n = difference[difference > 0].shape[0]
        # How many times sample is lower than population value
        negative_n = difference[difference < 0].shape[0]
        # Number of ties
        zero_n = difference[difference == 0].shape[0]
        sample_size = len(difference)
        median_absolute_deviation = float(median_abs_deviation(difference))

        # Summary Statistics for the Wilcoxon Sign Rank Test not Considering ties
        difference_no_ties = difference[difference != 0]  # This line removes the ties
        ranked_no_ties = rankdata(abs(difference_no_ties))
        positive_sum_ranks_no_ties = ranked_no_ties[difference_no_ties > 0].sum()
        negative_sum_ranks_no_ties = ranked_no_ties[difference_no_ties < 0].sum()

        # Summary Statistics for the Wilcoxon Sign Rank Considering ties
        ranked_with_ties = rankdata(abs(difference))
        positive_sum_ranks_with_ties = ranked_with_ties[difference > 0].sum()
        negative_sum_ranks_with_ties = ranked_with_ties[difference < 0].sum()

        # Wilcoxon Sign Rank Test Statistics Non Considering Ties (Wilcoxon Method)
        mean_w_not_considering_ties = (
            positive_sum_ranks_no_ties + negative_sum_ranks_no_ties
        ) / 2

        sign_no_ties = np.where(
            difference_no_ties == 0, 0, (np.where(difference_no_ties < 0, -1, 1))
        )

        ranked_signs_no_ties = sign_no_ties * ranked_no_ties
        ranked_signs_no_ties = np.where(
            difference_no_ties == 0, 0, ranked_signs_no_ties
        )

        var_adj_t = (ranked_signs_no_ties * ranked_signs_no_ties).sum()
        adjusted_variance_wilcoxon = (1 / 4) * var_adj_t

        # Calculate The Z score wilcox
        z_numerator_wilcoxon = positive_sum_ranks_no_ties - mean_w_not_considering_ties
        z_numerator_wilcoxon = np.where(
            z_numerator_wilcoxon < 0, z_numerator_wilcoxon + 0.5, z_numerator_wilcoxon
        )

        z_adjusted_wilcoxon = (z_numerator_wilcoxon) / np.sqrt(
            adjusted_variance_wilcoxon
        )
        z_adjusted_normal_approximation_wilcoxon = (
            z_numerator_wilcoxon - 0.5
        ) / np.sqrt(adjusted_variance_wilcoxon)
        p_value_adjusted_wilcoxon = min(
            float(norm.sf((abs(z_adjusted_wilcoxon))) * 2), 0.99999
        )
        p_value_adjusted_normal_approximation_wilcoxon = min(
            float(norm.sf((abs(z_adjusted_normal_approximation_wilcoxon))) * 2),
            0.99999,
        )

        # Wilcoxon Sign Rank Test Statistics Considering Ties (Pratt Method)
        mean_w_considering_ties = (
            positive_sum_ranks_with_ties + negative_sum_ranks_with_ties
        ) / 2
        sign_with_ties = np.where(difference == 0, 0, (np.where(difference < 0, -1, 1)))
        ranked_signs_with_ties = sign_with_ties * ranked_with_ties
        ranked_signs_with_ties = np.where(difference == 0, 0, ranked_signs_with_ties)
        var_adj_t_with_ties = (ranked_signs_with_ties * ranked_signs_with_ties).sum()
        adjusted_variance_pratt = (1 / 4) * var_adj_t_with_ties
        z_numerator_pratt = positive_sum_ranks_with_ties - mean_w_considering_ties
        z_numerator_pratt = np.where(
            z_numerator_pratt < 0, z_numerator_pratt + 0.5, z_numerator_pratt
        )
        z_adjusted_pratt = (z_numerator_pratt) / np.sqrt(adjusted_variance_pratt)
        z_adjusted_normal_approximation_pratt = (z_numerator_pratt - 0.5) / np.sqrt(
            adjusted_variance_pratt
        )
        p_value_adjusted_pratt = min(
            float(norm.sf((abs(z_adjusted_pratt))) * 2), 0.99999
        )
        p_value_adjusted_normal_approximation_pratt = min(
            float(norm.sf((abs(z_adjusted_normal_approximation_pratt))) * 2), 0.99999
        )

        # Matched Pairs Rank Biserial Correlation
        matched_pairs_rank_biserial_correlation_ignoring_ties = min(
            (positive_sum_ranks_no_ties - negative_sum_ranks_no_ties)
            / np.sum(ranked_no_ties),
            0.99999999,
        )  # This is the match paired rank biserial correlation using kerby formula that is not considering ties (Kerby, 2014)
        matched_pairs_rank_biserial_correlation_considering_ties = min(
            (positive_sum_ranks_with_ties - negative_sum_ranks_with_ties)
            / np.sum(ranked_with_ties),
            0.999999999,
        )  # this is the Kerby 2014 Formula - (With ties one can apply either Kerby or King Minium Formulae but not cureton - King's Formula is the most safe)

        # Z-based Rank Biserial Correlation (Note that since the Wilcoxon method is ignoring ties the sample size should actually be the number of the non tied pairs)
        z_based_rank_biserial_correlation_no_ties = z_adjusted_wilcoxon / np.sqrt(
            len(ranked_no_ties)
        )
        z_based_rank_biserial_correlation_corrected_no_ties = (
            z_adjusted_normal_approximation_wilcoxon / np.sqrt(len(ranked_no_ties))
        )
        z_based_rank_biserial_correlation_with_ties = z_adjusted_pratt / np.sqrt(
            sample_size
        )
        z_based_rank_biserial_correlation_corrected_with_ties = (
            z_adjusted_normal_approximation_pratt / np.sqrt(sample_size)
        )

        # Confidence Intervals
        standard_error_match_pairs_rank_biserial_correlation_no_ties = np.sqrt(
            (
                (
                    2 * (len(ranked_no_ties)) ** 3
                    + 3 * (len(ranked_no_ties)) ** 2
                    + (len(ranked_no_ties))
                )
                / 6
            )
            / (((len(ranked_no_ties)) ** 2 + (len(ranked_no_ties)) / 2))
        )
        standard_error_match_pairs_rank_biserial_correlation_with_ties = np.sqrt(
            ((2 * sample_size**3 + 3 * sample_size**2 + sample_size) / 6)
            / ((sample_size**2 + sample_size) / 2)
        )

        # Calculate the critical value for the confidence interval
        z_critical_value = norm.ppf((1 - confidence_level) + ((confidence_level) / 2))
        lower_ci_matched_pairs_wilcoxon = max(
            math.tanh(
                math.atanh(matched_pairs_rank_biserial_correlation_ignoring_ties)
                - z_critical_value
                * standard_error_match_pairs_rank_biserial_correlation_no_ties
            ),
            -1,
        )
        upper_ci_matched_pairs_wilcoxon = min(
            math.tanh(
                math.atanh(matched_pairs_rank_biserial_correlation_ignoring_ties)
                + z_critical_value
                * standard_error_match_pairs_rank_biserial_correlation_no_ties
            ),
            1,
        )
        lower_ci_z_based_wilcoxon = max(
            math.tanh(
                math.atanh(z_based_rank_biserial_correlation_no_ties)
                - z_critical_value
                * standard_error_match_pairs_rank_biserial_correlation_no_ties
            ),
            -1,
        )
        upper_ci_z_based_wilcoxon = min(
            math.tanh(
                math.atanh(z_based_rank_biserial_correlation_no_ties)
                + z_critical_value
                * standard_error_match_pairs_rank_biserial_correlation_no_ties
            ),
            1,
        )
        lower_ci_z_based_corrected_wilcoxon = max(
            math.tanh(
                math.atanh(z_based_rank_biserial_correlation_corrected_no_ties)
                - z_critical_value
                * standard_error_match_pairs_rank_biserial_correlation_no_ties
            ),
            -1,
        )
        upper_ci_z_based_corrected_wilcoxon = min(
            math.tanh(
                math.atanh(z_based_rank_biserial_correlation_corrected_no_ties)
                + z_critical_value
                * standard_error_match_pairs_rank_biserial_correlation_no_ties
            ),
            1,
        )
        lower_ci_matched_pairs_pratt = max(
            math.tanh(
                math.atanh(matched_pairs_rank_biserial_correlation_considering_ties)
                - z_critical_value
                * standard_error_match_pairs_rank_biserial_correlation_with_ties
            ),
            -1,
        )
        upper_ci_matched_pairs_pratt = min(
            math.tanh(
                math.atanh(matched_pairs_rank_biserial_correlation_considering_ties)
                + z_critical_value
                * standard_error_match_pairs_rank_biserial_correlation_with_ties
            ),
            1,
        )
        lower_ci_z_based_pratt = max(
            math.tanh(
                math.atanh(z_based_rank_biserial_correlation_with_ties)
                - z_critical_value
                * standard_error_match_pairs_rank_biserial_correlation_with_ties
            ),
            -1,
        )
        upper_ci_z_based_pratt = min(
            math.tanh(
                math.atanh(z_based_rank_biserial_correlation_with_ties)
                + z_critical_value
                * standard_error_match_pairs_rank_biserial_correlation_with_ties
            ),
            1,
        )
        lower_ci_z_based_corrected_pratt = max(
            math.tanh(
                math.atanh(z_based_rank_biserial_correlation_corrected_with_ties)
                - z_critical_value
                * standard_error_match_pairs_rank_biserial_correlation_with_ties
            ),
            -1,
        )
        upper_ci_z_based_corrected_pratt = min(
            math.tanh(
                math.atanh(z_based_rank_biserial_correlation_corrected_with_ties)
                + z_critical_value
                * standard_error_match_pairs_rank_biserial_correlation_with_ties
            ),
            1,
        )

        results = OneSampleAparametricResults()
        results.sample_size = sample_size
        results.median = sample_median
        results.median_of_absolute_deviation = median_absolute_deviation
        results.mean = sample_mean
        results.standard_deviation = sample_standard_deviation_1

        results.how_many_times_sample_is_larger = positive_n
        results.how_many_times_sample_is_smaller = negative_n
        results.how_many_ties = zero_n
        results.sample_size_of_non_ties = len(difference_no_ties)

        results.wilcoxon_matched_pairs_rank_biserial_z_score = z_adjusted_wilcoxon
        results.wilcoxon_matched_pairs_rank_biserial_p_value = p_value_adjusted_wilcoxon
        results.wilcoxon_matched_pairs_rank_biserial_effect_size = (
            matched_pairs_rank_biserial_correlation_ignoring_ties
        )

        results.wilcoxon_matched_pairs_rank_biserial_confidence_interval = res.ConfidenceInterval(
            lower_ci_matched_pairs_wilcoxon,
            upper_ci_matched_pairs_wilcoxon,
        )
        results.wilcoxon_z_based_rank_biserial_correlation_z_score = (
            z_based_rank_biserial_correlation_no_ties
        )
        results.wilcoxon_z_based_rank_biserial_correlation_p_value = (
            p_value_adjusted_wilcoxon
        )
        results.wilcoxon_z_based_rank_biserial_correlation_effect_size = (
            z_based_rank_biserial_correlation_no_ties
        )
        results.wilcoxon_z_based_rank_biserial_correlation_confidence_interval = res.ConfidenceInterval(
            lower_ci_z_based_wilcoxon,
            upper_ci_z_based_wilcoxon
        )
        results.wilcoxon_z_based_corrected_rank_biserial_correlation_z_score = (
            z_based_rank_biserial_correlation_corrected_no_ties
        )
        results.wilcoxon_z_based_corrected_rank_biserial_correlation_p_value = (
            p_value_adjusted_normal_approximation_wilcoxon
        )
        results.wilcoxon_z_based_corrected_rank_biserial_correlation_effect_size = (
            z_based_rank_biserial_correlation_corrected_no_ties
        )
        results.wilcoxon_z_based_corrected_rank_biserial_correlation_confidence_interval = res.ConfidenceInterval(
            lower_ci_z_based_corrected_wilcoxon,
            upper_ci_z_based_corrected_wilcoxon,
        )
        results.pratt_matched_pairs_rank_biserial_z_score = z_adjusted_pratt
        results.pratt_matched_pairs_rank_biserial_p_value = p_value_adjusted_pratt
        results.pratt_matched_pairs_rank_biserial_effect_size = (
            matched_pairs_rank_biserial_correlation_considering_ties
        )
        results.pratt_matched_pairs_rank_biserial_confidence_interval = res.ConfidenceInterval(
            lower_ci_matched_pairs_pratt,
            upper_ci_matched_pairs_pratt,
        )
        results.pratt_z_based_rank_biserial_correlation_z_score = (
            z_based_rank_biserial_correlation_with_ties
        )

        results.pratt_z_based_rank_biserial_correlation_p_value = p_value_adjusted_pratt
        results.pratt_z_based_rank_biserial_correlation_effect_size = (
            z_based_rank_biserial_correlation_with_ties
        )
        results.pratt_z_based_rank_biserial_correlation_confidence_interval = (
            lower_ci_z_based_pratt,
            upper_ci_z_based_pratt,
        )
        results.pratt_z_based_corrected_rank_biserial_correlation_z_score = (
            z_based_rank_biserial_correlation_corrected_with_ties
        )
        results.pratt_z_based_corrected_rank_biserial_correlation_p_value = (
            p_value_adjusted_normal_approximation_pratt
        )
        results.pratt_z_based_corrected_rank_biserial_correlation_effect_size = (
            z_based_rank_biserial_correlation_corrected_with_ties
        )
        results.pratt_z_based_corrected_rank_biserial_correlation_confidence_interval = res.ConfidenceInterval(
            lower_ci_z_based_corrected_pratt,
            upper_ci_z_based_corrected_pratt,
        )

        return results