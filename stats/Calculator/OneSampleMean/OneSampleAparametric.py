"""
This module provides functionality for calculating the Aparametric effect size using the sign test for one sample.

Classes:
    AparametricOneSample: A class containing static methods for calculating the Aparametric effect size.

Methods:
    ApermetricEffectSizeOneSample: Calculate the Aparametric effect size using the sign test for one sample.
"""
import math
import numpy as np
from scipy.stats import norm, rankdata, median_abs_deviation

# Create results class
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
    def __init__(self):
        # General Summary Statistics
        self.sample = None
        self.sample_median = None
        self.median_of_the_differnece = None
        self.median_of_absoult_deviation = None
        self.sample_mean = None
        self.sample_standard_deviation = None
        self.number_of_pairs = None
        self.number_of_pairs_with_a_sign = None
        self.number_of_times_sample_is_larger = None
        self.number_of_times_sample_is_smaller = None
        self.number_of_ties = None

        # Wilcoxon Statistics (Wilcoxon Method that Ignores ties)
        self.wilcoxon_method = ""
        self._______________ = ""
        self.sum_of_the_positive_ranks_without_ties = None
        self.sum_of_the_negative_ranks_without_ties = None

        # Wilcoxon Sign Rank Test Statistics (Wilcoxon)
        self.wilcoxon_mean_w_without_ties = None
        self.wilcoxon_standard_deviation = None
        self.wilcoxon_z = None
        self.wilcoxon_z_with_normal_approximation_continuity_correction = None
        self.wilcoxon_p_value = None
        self.wilcoxon_p_value_with_normal_approximation_continuity_correction = None

        # Rank Biserial Correlation
        self.matched_pairs_rank_biserial_correlation_ignoring_ties = None
        self.z_based_rank_biserial_correlation_wilcoxon_method = None
        self.z_based_corrected_rank_biserial_correlation_wilcoxon_method = None

        # Confidence Intervals
        self.standard_error_of_the_matched_pairs_rank_biserial_correlation_wilcoxon_method = None
        self.lower_ci_matched_pairs_rank_biserial_wilcoxon = None
        self.upper_ci_matched_pairs_rank_biserial_wilcoxon = None
        self.lower_ci_z_based_rank_biserial_wilcoxon = None
        self.upper_ci_z_based_rank_biserial_wilcoxon = None
        self.lower_ci_z_based_corrected_rank_biserial_wilcoxon = None
        self.upper_ci_z_based_corrected_rank_biserial_wilcoxon = None

        # Statistical Lines Wilcoxon Method
        self.statistical_line_wilcoxon = None
        self.statistical_line_wilcoxon_corrected = None
        self.statistical_line_wilcoxon_matched_pairs = None

        self.pratt_method = ""
        self.sum_of_the_positive_ranks_with_ties = None
        self.sum_of_the_negative_ranks_with_ties = None

        self.pratt_meanw_considering_ties = None
        self.pratt_standard_deviation = None
        self.pratt_z = None
        self.pratt_z_with_normal_approximation_continuity_correction = None
        self.pratt_p_value = None
        self.pratt_p_value_with_normal_approximation_continuity_correction = None

        # Rank Biserial Correlation
        self.matched_pairs_rank_biserial_correlation_considering_ties = None
        self.z_based_rank_biserial_correlation_pratt_method = None
        self.z_based_corrected_rank_biserial_correlation_pratt_method = None

        # Confidence Intervals
        self.standard_error_of_the_matched_pairs_rank_biserial_correlation_pratt_method = None
        self.lower_ci_matched_pairs_rank_biserial_pratt = None
        self.upper_ci_matched_pairs_rank_biserial_pratt = None
        self.lower_ci_z_based_rank_biserial_pratt = None
        self.upper_ci_z_based_rank_biserial_pratt = None
        self.lower_ci_z_based_corrected_rank_biserial_pratt = None
        self.upper_ci_z_based_corrected_rank_biserial_pratt = None

        # Statistical Lines
        self.statistical_line_pratt = None
        self.statistical_line_pratt_corrected = None
        self.statistical_line_pratt_matched_pairs = None



class AparametricOneSample:
    """
    A class containing static methods for calculating the Aparametric effect size using the sign test for one sample.

    This class includes the following static methods:
    - Apermetric_Effect_Size_onesample: Calculate the Aparametric effect size using the sign test for one sample.
    """

    @staticmethod
    def apermetric_effect_size_one_sample(params: dict) -> dict:
        """
        Calculate the Aparametric effect size using the sign test for one sample.

        Parameters:
        params (dict): A dictionary containing the following keys:
            - "Column 1": A numpy array of sample data.
            - "Population's Value": The population value to compare against.
            - "Confidence Level": The confidence level as a percentage.

        Returns:
        dict: A dictionary containing the results of the Aparametric effect size calculations.
        """

        # Set Parameters
        column_1 = params["Column 1"]
        population_value = params["Population's Value"]
        confidence_level_percentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentages / 100

        # General Summary Statistics
        sample_median_1 = np.median(column_1)
        sample_mean_1 = np.mean(column_1)
        sample_standard_deviation_1 = np.std(column_1, ddof=1)
        difference = column_1 - population_value
        positive_n = difference[difference > 0].shape[
            0
        ]  # How many times sample is greater than population value
        negative_n = difference[difference < 0].shape[
            0
        ]  # How many times sample is lower than population value
        zero_n = difference[difference == 0].shape[0]  # Number of ties
        sample_size = len(difference)
        median_difference = np.median(difference)
        median_absulute_deviation = median_abs_deviation(difference)

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
        unadjusted_variance_wilcoxon = (
            len(difference_no_ties)
            * (len(difference_no_ties) + 1)
            * (2 * (len(difference_no_ties)) + 1)
        ) / 24
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
        z_unadjusted_wilcoxon = (z_numerator_wilcoxon) / np.sqrt(
            unadjusted_variance_wilcoxon
        )
        z_unadjusted_normal_approximation_wilcoxon = (
            z_numerator_wilcoxon - 0.5
        ) / np.sqrt(unadjusted_variance_wilcoxon)
        p_value_adjusted_wilcoxon = min(
            float(norm.sf((abs(z_adjusted_wilcoxon))) * 2), 0.99999
        )
        p_value_adjusted_normal_approximation_wilcoxon = min(
            float(norm.sf((abs(z_adjusted_normal_approximation_wilcoxon))) * 2),
            0.99999,
        )
        p_value_unadjusted_wilcoxon = min(
            float(norm.sf((abs(z_unadjusted_wilcoxon))) * 2), 0.99999
        )
        p_value_unadjusted_normal_approximation_wilcoxon = min(
            float(norm.sf((abs(z_unadjusted_normal_approximation_wilcoxon))) * 2),
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

        
        def CreateStatisticalLine(test_statistic, z_statistic, p_value, effect_size, confidence_level_percentages, lower_ci, upper_ci, test_type="w", z_type="Z"):
            """Creates a formatted statistical line string with test statistic, z-score, p-value, effect size and confidence intervals.
            
            Args:
                test_statistic (float): The test statistic value (T)
                z_statistic (float): The z-score statistic
                p_value (float): The p-value
                effect_size (float): The effect size value (rank biserial correlation)
                confidence_level_percentages (int): Confidence level percentage
                lower_ci (float): Lower confidence interval bound
                upper_ci (float): Upper confidence interval bound
                test_type (str): Type of test ('w' for Wilcoxon or 'p' for Pratt)
                z_type (str): Type of z-statistic ('Z' or 'Zcorrected')
                
            Returns:
                str: Formatted statistical line
            """
            # Format test statistic
            formatted_test_stat = (
                int(test_statistic) 
                if float(test_statistic).is_integer()
                else test_statistic
            )
            
            # Format p-value
            formatted_p_value = (
                "{:.3f}".format(p_value).lstrip("0")
                if p_value >= 0.001
                else "\033[3mp\033[0m < .001"
            )
            
            # Format effect size
            formatted_effect_size = (
                ("-" if str(effect_size).startswith("-") else "") +
                str(round(effect_size, 3))
                .lstrip("-")
                .lstrip("0")
                or "0"
            )
            
            # Format confidence intervals
            formatted_lower_ci = (
                ("-" if str(lower_ci).startswith("-") else "") +
                str(round(lower_ci, 3))
                .lstrip("-")
                .lstrip("0")
                or "0"
            )
            
            formatted_upper_ci = (
                ("-" if str(upper_ci).startswith("-") else "") +
                str(round(upper_ci, 3))
                .lstrip("-")
                .lstrip("0")
                or "0"
            )
            
            # Create statistical line
            return " \033[3mT{}\033[0m = {}, \033[3m{}\033[0m = {:.3f}, {}{}, \033[3mRBz\033[0m = {}, {}% CI(Fisher) [{}, {}]".format(
                test_type,
                formatted_test_stat,
                z_type,
                z_statistic,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                formatted_effect_size,
                confidence_level_percentages,
                formatted_lower_ci,
                formatted_upper_ci
            )

        results = OneSampleAparametricResults()

        # General Summary Statistics
        results.sample = positive_sum_ranks_with_ties
        results.sample_median = round(sample_median_1, 4)
        results.median_of_the_differnece = median_difference
        results.median_of_absoult_deviation = median_absulute_deviation
        results.sample_mean = round(sample_mean_1, 4)
        results.sample_standard_deviation = round(sample_standard_deviation_1, 4)
        results.number_of_pairs = sample_size
        results.number_of_pairs_with_a_sign = len(ranked_no_ties)
        results.number_of_times_sample_is_larger = positive_n
        results.number_of_times_sample_is_smaller = negative_n
        results.number_of_ties = zero_n

        # Wilcoxon Statistics (Wilcoxon Method that Ignores ties)
        results.sum_of_the_positive_ranks_without_ties = round(positive_sum_ranks_no_ties, 4)
        results.sum_of_the_negative_ranks_without_ties = round(negative_sum_ranks_no_ties, 4)

        # Wilcoxon Sign Rank Test Statistics (Wilcoxon)
        results.wilcoxon_mean_w_without_ties = mean_w_not_considering_ties
        results.wilcoxon_standard_deviation = np.sqrt(adjusted_variance_wilcoxon)
        results.wilcoxon_z = z_adjusted_wilcoxon
        results.wilcoxon_z_with_normal_approximation_continuity_correction = z_adjusted_normal_approximation_wilcoxon
        results.wilcoxon_p_value = p_value_adjusted_wilcoxon
        results.wilcoxon_p_value_with_normal_approximation_continuity_correction = p_value_adjusted_normal_approximation_wilcoxon

        # Rank Biserial Correlation
        results.matched_pairs_rank_biserial_correlation_ignoring_ties = round(matched_pairs_rank_biserial_correlation_ignoring_ties, 5)
        results.z_based_rank_biserial_correlation_wilcoxon_method = round(z_based_rank_biserial_correlation_no_ties, 5)
        results.z_based_corrected_rank_biserial_correlation_wilcoxon_method = round(z_based_rank_biserial_correlation_corrected_no_ties, 5)

        # Confidence Intervals
        results.standard_error_of_the_matched_pairs_rank_biserial_correlation_wilcoxon_method = round(standard_error_match_pairs_rank_biserial_correlation_no_ties, 4)
        results.lower_ci_matched_pairs_rank_biserial_wilcoxon = round(lower_ci_matched_pairs_wilcoxon, 5)
        results.upper_ci_matched_pairs_rank_biserial_wilcoxon = round(upper_ci_matched_pairs_wilcoxon, 5)
        results.lower_ci_z_based_rank_biserial_wilcoxon = round(lower_ci_z_based_wilcoxon, 5)
        results.upper_ci_z_based_rank_biserial_wilcoxon = round(upper_ci_z_based_wilcoxon, 5)
        results.lower_ci_z_based_corrected_rank_biserial_wilcoxon = round(lower_ci_z_based_corrected_wilcoxon, 5)
        results.upper_ci_z_based_corrected_rank_biserial_wilcoxon = round(upper_ci_z_based_corrected_wilcoxon, 5)

        # Statistical Lines Wilcoxon Method
        results.statistical_line_wilcoxon = CreateStatisticalLine(
            positive_sum_ranks_no_ties,
            z_adjusted_wilcoxon,
            p_value_adjusted_wilcoxon,
            z_based_rank_biserial_correlation_no_ties,
            confidence_level_percentages,
            lower_ci_z_based_wilcoxon,
            upper_ci_z_based_wilcoxon,
            "w",
            "Z"
        )

        results.statistical_line_wilcoxon_corrected = CreateStatisticalLine(
            positive_sum_ranks_no_ties,
            z_adjusted_normal_approximation_wilcoxon,
            p_value_adjusted_normal_approximation_wilcoxon,
            z_based_rank_biserial_correlation_corrected_no_ties,
            confidence_level_percentages,
            lower_ci_z_based_corrected_wilcoxon,
            upper_ci_z_based_corrected_wilcoxon,
            "w",
            "Zcorrected"
        )

        results.statistical_line_wilcoxon_matched_pairs = CreateStatisticalLine(
            positive_sum_ranks_no_ties,
            z_adjusted_normal_approximation_wilcoxon,
            p_value_adjusted_normal_approximation_wilcoxon,
            matched_pairs_rank_biserial_correlation_ignoring_ties,
            confidence_level_percentages,
            lower_ci_matched_pairs_wilcoxon,
            upper_ci_matched_pairs_wilcoxon,
            "w",
            "Zcorrected"
        )

        results.sum_of_the_positive_ranks_with_ties = round(positive_sum_ranks_with_ties, 4)
        results.sum_of_the_negative_ranks_with_ties = round(negative_sum_ranks_with_ties, 4)

        results.pratt_meanw_considering_ties = mean_w_considering_ties
        results.pratt_standard_deviation = np.sqrt(adjusted_variance_pratt)
        results.pratt_z = z_adjusted_pratt
        results.pratt_z_with_normal_approximation_continuity_correction = z_adjusted_normal_approximation_pratt
        results.pratt_p_value = p_value_adjusted_pratt
        results.pratt_p_value_with_normal_approximation_continuity_correction = p_value_adjusted_normal_approximation_pratt

        # Rank Biserial Correlation
        results.matched_pairs_rank_biserial_correlation_considering_ties = round(matched_pairs_rank_biserial_correlation_considering_ties, 5)
        results.z_based_rank_biserial_correlation_pratt_method = round(z_based_rank_biserial_correlation_with_ties, 5)
        results.z_based_corrected_rank_biserial_correlation_pratt_method = round(z_based_rank_biserial_correlation_corrected_with_ties, 5)

        # Confidence Intervals
        results.standard_error_of_the_matched_pairs_rank_biserial_correlation_pratt_method = round(standard_error_match_pairs_rank_biserial_correlation_with_ties, 4)
        results.lower_ci_matched_pairs_rank_biserial_pratt = round(lower_ci_matched_pairs_pratt, 5)
        results.upper_ci_matched_pairs_rank_biserial_pratt = round(upper_ci_matched_pairs_pratt, 5)
        results.lower_ci_z_based_rank_biserial_pratt = round(lower_ci_z_based_pratt, 5)
        results.upper_ci_z_based_rank_biserial_pratt = round(upper_ci_z_based_pratt, 5)
        results.lower_ci_z_based_corrected_rank_biserial_pratt = round(lower_ci_z_based_corrected_pratt, 5)
        results.upper_ci_z_based_corrected_rank_biserial_pratt = round(upper_ci_z_based_corrected_pratt, 5)

        # Statistical Lines Pratt Method
        results.statistical_line_pratt = CreateStatisticalLine(
            positive_sum_ranks_with_ties,
            z_adjusted_pratt,
            p_value_adjusted_pratt,
            z_based_rank_biserial_correlation_with_ties,
            confidence_level_percentages,
            lower_ci_z_based_pratt,
            upper_ci_z_based_pratt,
            "p",
            "Z"
        )

        results.statistical_line_pratt_corrected = CreateStatisticalLine(
            positive_sum_ranks_with_ties,
            z_adjusted_normal_approximation_pratt,
            p_value_adjusted_normal_approximation_pratt,
            z_based_rank_biserial_correlation_corrected_with_ties,
            confidence_level_percentages,
            lower_ci_z_based_corrected_pratt,
            upper_ci_z_based_corrected_pratt,
            "p",
            "Zcorrected"
        )

        results.statistical_line_pratt_matched_pairs = CreateStatisticalLine(
            positive_sum_ranks_with_ties,
            z_adjusted_normal_approximation_pratt,
            p_value_adjusted_normal_approximation_pratt,
            matched_pairs_rank_biserial_correlation_considering_ties,
            confidence_level_percentages,
            lower_ci_matched_pairs_pratt,
            upper_ci_matched_pairs_pratt,
            "p",
            "Zcorrected"
        )

        return results

    # Things to Consider
    # 1. Consider adding other CI's for example metsamuuronen method for sommers delta (which in the case of two groups equals the rank biserial correlation)
    # 2. Test if the matched pairs version is also equal to Sommers delta and cliffs delta (dependent version)
    # 3. For convenience change the confidence levels to percentages and not decimals
