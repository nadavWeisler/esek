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
from ...utils import interfaces, res, es


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

    group: Optional[res.Group] = None
    wilcoxon_signed_rank: Optional[res.WilcoxonSignedRank] = None
    wilcoxon_matched_pairs_rank_biserial: Optional[es.Biserial] = None
    wilcoxon_z_based_rank_biserial_correlation: Optional[es.Biserial] = None
    wilcoxon_z_based_corrected_rank_biserial_correlation: Optional[es.Biserial] = None
    pratt_matched_pairs_rank_biserial_z_score: Optional[es.Biserial] = None
    pratt_z_based_rank_biserial_correlation: Optional[es.Biserial] = None
    pratt_z_based_corrected_rank_biserial_correlation: Optional[es.Biserial] = None


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
        group = res.Group(
            sample_size=sample_size,
            median=sample_median,
            median_absolute_deviation=median_absolute_deviation,
            mean=sample_mean,
            standard_deviation=sample_standard_deviation_1,
        )
        wilcoxon_signed_rank = res.WilcoxonSignedRank(
            times_group1_larger=positive_n,
            times_group2_larger=negative_n,
            ties=zero_n,
            num_of_pairs=len(difference_no_ties),
            num_of_non_tied_pairs=len(difference_no_ties),
        )

        wilcoxon_matched_pairs_rank_biserial = es.Biserial(
            name="Matched Pairs Rank Biserial Correlation (Ignoring Ties)",
            value=matched_pairs_rank_biserial_correlation_ignoring_ties,
            ci_lower=lower_ci_matched_pairs_wilcoxon,
            ci_upper=upper_ci_matched_pairs_wilcoxon,
            standard_error=standard_error_match_pairs_rank_biserial_correlation_no_ties,
        )
        wilcoxon_matched_pairs_rank_biserial.z_score = z_adjusted_wilcoxon
        wilcoxon_matched_pairs_rank_biserial.p_value = p_value_adjusted_wilcoxon

        wilcoxon_z_based_rank_biserial_correlation = es.Biserial(
            name="Z Based Rank Biserial Correlation (Ignoring Ties)",
            value=z_based_rank_biserial_correlation_no_ties,
            ci_lower=lower_ci_z_based_wilcoxon,
            ci_upper=upper_ci_z_based_wilcoxon,
            standard_error=standard_error_match_pairs_rank_biserial_correlation_no_ties,
        )
        wilcoxon_z_based_rank_biserial_correlation.z_score = z_adjusted_wilcoxon
        wilcoxon_z_based_rank_biserial_correlation.p_value = p_value_adjusted_wilcoxon

        wilcoxon_z_based_corrected_rank_biserial_correlation = es.Biserial(
            name="Z Based Corrected Rank Biserial Correlation (Ignoring Ties)",
            value=z_based_rank_biserial_correlation_corrected_no_ties,
            ci_lower=lower_ci_z_based_corrected_wilcoxon,
            ci_upper=upper_ci_z_based_corrected_wilcoxon,
            standard_error=standard_error_match_pairs_rank_biserial_correlation_no_ties,
        )
        wilcoxon_z_based_corrected_rank_biserial_correlation.z_score = (
            z_adjusted_normal_approximation_wilcoxon
        )
        wilcoxon_z_based_corrected_rank_biserial_correlation.p_value = (
            p_value_adjusted_normal_approximation_wilcoxon
        )

        pratt_matched_pairs_rank_biserial = es.Biserial(
            name="Matched Pairs Rank Biserial Correlation (Considering Ties)",
            value=matched_pairs_rank_biserial_correlation_considering_ties,
            ci_lower=lower_ci_matched_pairs_pratt,
            ci_upper=upper_ci_matched_pairs_pratt,
            standard_error=standard_error_match_pairs_rank_biserial_correlation_with_ties,
        )
        pratt_matched_pairs_rank_biserial.z_score = z_adjusted_pratt
        pratt_matched_pairs_rank_biserial.p_value = p_value_adjusted_pratt

        pratt_z_based_rank_biserial_correlation = es.Biserial(
            name="Z Based Rank Biserial Correlation (Considering Ties)",
            value=z_based_rank_biserial_correlation_with_ties,
            ci_lower=lower_ci_z_based_pratt,
            ci_upper=upper_ci_z_based_pratt,
            standard_error=standard_error_match_pairs_rank_biserial_correlation_with_ties,
        )

        pratt_z_based_rank_biserial_correlation.z_score = z_adjusted_pratt
        pratt_z_based_rank_biserial_correlation.p_value = p_value_adjusted_pratt

        pratt_z_based_corrected_rank_biserial_correlation = es.Biserial(
            name="Z Based Corrected Rank Biserial Correlation (Considering Ties)",
            value=z_based_rank_biserial_correlation_corrected_with_ties,
            ci_lower=lower_ci_z_based_corrected_pratt,
            ci_upper=upper_ci_z_based_corrected_pratt,
            standard_error=standard_error_match_pairs_rank_biserial_correlation_with_ties,
        )
        pratt_z_based_corrected_rank_biserial_correlation.z_score = (
            z_adjusted_normal_approximation_pratt
        )
        pratt_z_based_corrected_rank_biserial_correlation.p_value = (
            p_value_adjusted_normal_approximation_pratt
        )


        results.group = group
        results.wilcoxon_signed_rank = wilcoxon_signed_rank
        results.wilcoxon_matched_pairs_rank_biserial = (
            wilcoxon_matched_pairs_rank_biserial
        )
        results.wilcoxon_z_based_rank_biserial_correlation = (
            wilcoxon_z_based_rank_biserial_correlation
        )
        results.wilcoxon_z_based_corrected_rank_biserial_correlation = (
            wilcoxon_z_based_corrected_rank_biserial_correlation
        )
        results.pratt_matched_pairs_rank_biserial_z_score = (
            pratt_matched_pairs_rank_biserial
        )
        results.pratt_z_based_rank_biserial_correlation = (
            pratt_z_based_rank_biserial_correlation
        )
        results.pratt_z_based_corrected_rank_biserial_correlation = (
            pratt_z_based_corrected_rank_biserial_correlation
        )

        return results
