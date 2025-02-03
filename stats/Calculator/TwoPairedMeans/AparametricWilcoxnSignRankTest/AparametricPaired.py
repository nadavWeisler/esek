"""
This module provides functionality for calculating the Aparametric effect size using the Wilcoxon signed-rank test for paired samples.

Classes:
    Aparametric_Paired_Samples: A class containing static methods for calculating the Aparametric effect size.

Methods:
    Apermetric_Effect_Size_Dependent: Calculate the Aparametric effect size using the Wilcoxon signed-rank test for paired samples.
"""

import numpy as np
from scipy.stats import pearsonr, norm, rankdata
import math


class Aparametric_Paired_Samples:
    @staticmethod
    def Apermetric_Effect_Size_Dependent(
        params: dict,
    ) -> dict:  # Aparametric effect size with sign test

        # Set Parameters
        column_1 = params["Column 1"]
        column_2 = params["Column 2"]
        confidence_level = params["Confidence Level"]

        # Calculation
        sample_median_1 = np.median(column_1)
        sample_median_2 = np.median(column_2)
        sample_mean_1 = np.mean(column_1)
        sample_mean_2 = np.mean(column_2)
        sample_standard_deviation_1 = np.std(column_1, ddof=1)
        sample_standard_deviation_2 = np.std(column_2, ddof=1)
        sample_size = len(column_1)
        correlation = pearsonr(column_1, column_2)
        pearson_correlation = correlation[0]
        pearson_correlation_pvalue = correlation[1]

        def sign_test_wilcoxon_method(
            x, y, population_median=0
        ):  # this is the original method by wilcoxon that ignores ties
            if y is None:
                difference = x - population_median
            else:
                difference = x - y
            sample_size = len(x)
            sign = np.where(difference < 0, -1, 1)
            difference = difference[difference != 0]
            difference_abs = np.abs(difference)
            total_n = difference.shape[0]
            positive_n = difference[difference > 0].shape[0]
            negative_n = difference[difference < 0].shape[0]
            zero_n = difference[difference == 0].shape[0]
            ranked = rankdata(difference_abs)
            sign = np.where(difference < 0, -1, 1)
            ranked_sign = sign * ranked
            total_sum_ranks = ranked.sum()
            positive_sum_ranks = ranked[difference > 0].sum()
            negative_sum_ranks = ranked[difference < 0].sum()
            mean_positive_ranks = positive_sum_ranks / positive_n
            mean_negative_ranks = negative_sum_ranks / negative_n
            zero_sum_ranks = ranked[difference == 0].sum()
            sign2 = np.where(difference == 0, 0, sign)
            ranked2 = sign2 * ranked
            ranked2 = np.where(difference == 0, 0, ranked2)
            var_adj_T = (ranked2 * ranked2).sum()
            Adjusted_Variance = (1 / 4) * var_adj_T
            Unadjusted_Variance = ((total_n * (total_n + 1)) * (2 * total_n + 1)) / 24
            var_zero_adj_T_pos = -1 * ((zero_n * (zero_n + 1)) * (2 * zero_n + 1)) / 24
            var_ties_adj = Adjusted_Variance - Unadjusted_Variance - var_zero_adj_T_pos
            MeanW = ((positive_n + negative_n) * (positive_n + negative_n + 1)) / 4
            RBCwilcoxon = (positive_sum_ranks - negative_sum_ranks) / (
                positive_sum_ranks + negative_sum_ranks
            )  # This is the match paired rank biserial correlation (Kerby, 2014)

            # Calculate The Z score wilcox
            Z_adjusted = (positive_sum_ranks - MeanW) / np.sqrt(Adjusted_Variance)
            Z_Unadjusted = (positive_sum_ranks - MeanW) / np.sqrt(Unadjusted_Variance)
            Z_adjusted_Corrected = (
                positive_sum_ranks - MeanW - 0.5 * np.sign(positive_sum_ranks - MeanW)
            ) / np.sqrt(Adjusted_Variance)
            Z_Unadjusted_Corrected = (
                positive_sum_ranks - MeanW - 0.5 * np.sign(positive_sum_ranks - MeanW)
            ) / np.sqrt(Unadjusted_Variance)
            p_value_adjusted = min(float(norm.sf((abs(Z_adjusted))) * 2), 0.99999)
            p_value_unadjusted = min(float(norm.sf((abs(Z_Unadjusted))) * 2), 0.99999)
            p_value_adjusted_corrected = min(
                float(norm.sf((abs(Z_adjusted_Corrected))) * 2), 0.99999
            )
            p_value_unadjusted_corrected = min(
                float(norm.sf((abs(Z_Unadjusted_Corrected))) * 2), 0.99999
            )
            RBC_adjusted = Z_adjusted / np.sqrt(sample_size)
            RBC_unadjusted = Z_Unadjusted / np.sqrt(sample_size)
            RBC_unajusted_corrected = Z_adjusted_Corrected / np.sqrt(sample_size)
            RBC_adjusted_corrected = Z_Unadjusted_Corrected / np.sqrt(sample_size)

            return np.array(
                [
                    positive_n,
                    negative_n,
                    total_n,
                    positive_sum_ranks,
                    negative_sum_ranks,
                    total_sum_ranks,
                    mean_positive_ranks,
                    mean_negative_ranks,
                    MeanW,
                    Adjusted_Variance,
                    Unadjusted_Variance,
                    RBCwilcoxon,
                    Z_adjusted,
                    Z_Unadjusted,
                    Z_adjusted_Corrected,
                    Z_Unadjusted_Corrected,
                    p_value_adjusted,
                    p_value_unadjusted,
                    p_value_adjusted_corrected,
                    p_value_unadjusted_corrected,
                    RBC_adjusted,
                    RBC_unadjusted,
                    RBC_unajusted_corrected,
                    RBC_adjusted_corrected,
                ]
            )

        def sign_test_pratt_method(
            x, y, population_median=0
        ):  # this is the original method by pratt that consider ties in the data
            if y is None:
                difference = x - population_median
            else:
                difference = x - y
            sample_size = len(difference)
            sign = np.where(difference < 0, -1, 1)
            difference_abs = np.abs(difference)
            total_n = difference.shape[0]
            positive_n = difference[difference > 0].shape[0]
            negative_n = difference[difference < 0].shape[0]
            zero_n = difference[difference == 0].shape[0]
            ranked = rankdata(difference_abs)
            sign = np.where(difference < 0, -1, 1)
            ranked_sign = sign * ranked
            total_sum_ranks = ranked.sum()
            positive_sum_ranks = ranked[difference > 0].sum()
            negative_sum_ranks = ranked[difference < 0].sum()
            mean_positive_ranks = positive_sum_ranks / positive_n
            mean_negative_ranks = negative_sum_ranks / negative_n
            zero_sum_ranks = ranked[difference == 0].sum()
            sign2 = np.where(difference == 0, 0, sign)
            ranked2 = sign2 * ranked
            ranked2 = np.where(difference == 0, 0, ranked2)
            var_adj_T = (ranked2 * ranked2).sum()
            Adjusted_Var = (1 / 4) * var_adj_T
            Unadjusted_Var = ((total_n * (total_n + 1)) * (2 * total_n + 1)) / 24
            var_zero_adj_T_pos = -1 * ((zero_n * (zero_n + 1)) * (2 * zero_n + 1)) / 24
            var_ties_adj = Adjusted_Var - Unadjusted_Var - var_zero_adj_T_pos
            MeanW = (positive_sum_ranks + negative_sum_ranks) / 2
            RBCpratt = (positive_sum_ranks - negative_sum_ranks) / (
                positive_sum_ranks + negative_sum_ranks + zero_sum_ranks
            )  # This is the matched pairs rank biserial correlation

            # Calculate The Z score pratt
            Z_adjusted = (positive_sum_ranks - MeanW) / np.sqrt(Adjusted_Var)
            Z_Unadjusted = (positive_sum_ranks - MeanW) / np.sqrt(Unadjusted_Var)
            Z_adjusted_Corrected = (
                positive_sum_ranks - MeanW - 0.5 * np.sign(positive_sum_ranks - MeanW)
            ) / np.sqrt(Adjusted_Var)
            Z_Unadjusted_Corrected = (
                positive_sum_ranks - MeanW - 0.5 * np.sign(positive_sum_ranks - MeanW)
            ) / np.sqrt(Unadjusted_Var)
            p_value_adjusted = min(float(norm.sf((abs(Z_adjusted))) * 2), 0.99999)
            p_value_unadjusted = min(float(norm.sf((abs(Z_Unadjusted))) * 2), 0.99999)
            p_value_adjusted_corrected = min(
                float(norm.sf((abs(Z_adjusted_Corrected))) * 2), 0.99999
            )
            p_value_unadjusted_corrected = min(
                float(norm.sf((abs(Z_Unadjusted_Corrected))) * 2), 0.99999
            )
            RBC_adjusted = Z_adjusted / np.sqrt(sample_size)
            RBC_unadjusted = Z_Unadjusted / np.sqrt(sample_size)
            RBC_unajusted_corrected = Z_Unadjusted_Corrected / np.sqrt(sample_size)
            RBC_adjusted_corrected = Z_adjusted_Corrected / np.sqrt(sample_size)

            return np.array(
                [
                    positive_n,
                    negative_n,
                    total_n,
                    positive_sum_ranks,
                    negative_sum_ranks,
                    total_sum_ranks,
                    mean_positive_ranks,
                    mean_negative_ranks,
                    MeanW,
                    Adjusted_Var,
                    Unadjusted_Var,
                    RBCpratt,
                    Z_adjusted,
                    Z_Unadjusted,
                    Z_adjusted_Corrected,
                    Z_Unadjusted_Corrected,
                    p_value_adjusted,
                    p_value_unadjusted,
                    p_value_adjusted_corrected,
                    p_value_unadjusted_corrected,
                    RBC_adjusted,
                    RBC_unadjusted,
                    RBC_unajusted_corrected,
                    RBC_adjusted_corrected,
                ]
            )

        # First method: sign_test_wilcoxon_method
        (
            wilcoxon_positive_n,
            wilcoxon_negative_n,
            wilcoxon_total_n,
            wilcoxon_positive_sum_ranks,
            wilcoxon_negative_sum_ranks,
            wilcoxon_total_sum_ranks,
            wilcoxon_mean_positive_ranks,
            wilcoxon_mean_negative_ranks,
            wilcoxon_MeanW,
            wilcoxon_Adjusted_Var,
            wilcoxon_Unadjusted_Var,
            wilcoxon_RBCwilcoxon,
            wilcoxon_Z_adjusted,
            wilcoxon_Z_Unadjusted,
            wilcoxon_Z_adjusted_Corrected,
            wilcoxon_Z_Unadjusted_Corrected,
            wilcoxon_p_value_adjusted,
            wilcoxon_p_value_unadjusted,
            wilcoxon_p_value_adjusted_corrected,
            wilcoxon_p_value_unadjusted_corrected,
            wilcoxon_RBC_adjusted,
            wilcoxon_RBC_unadjusted,
            wilcoxon_RBC_unajusted_corrected,
            wilcoxon_RBC_adjusted_corrected,
        ) = sign_test_wilcoxon_method(column_1, column_2)

        # Second method: sign_test_pratt_method
        (
            pratt_positive_n,
            pratt_negative_n,
            pratt_total_n,
            pratt_positive_sum_ranks,
            pratt_negative_sum_ranks,
            pratt_total_sum_ranks,
            pratt_mean_positive_ranks,
            pratt_mean_negative_ranks,
            pratt_MeanW,
            pratt_Adjusted_Var,
            pratt_Unadjusted_Var,
            RBCpratt,
            pratt_Z_adjusted,
            pratt_Z_Unadjusted,
            pratt_Z_adjustedCorrected,
            pratt_Z_Unadjusted_Corrected,
            pratt_p_value_adjusted,
            pratt_p_value_unadjusted,
            pratt_p_value_adjusted_corrected,
            pratt_p_value_unadjusted_corrected,
            pratt_RBC_adjusted,
            pratt_RBC_unadjusted,
            pratt_RBC_unajusted_corrected,
            pratt_RBC_adjusted_corrected,
        ) = sign_test_pratt_method(column_1, column_2)

        # Confidence Intervals for the Rank Biserial Correlation
        Standrd_Error_RBR = np.sqrt(
            ((2 * sample_size**3 + 3 * sample_size**2 + sample_size) / 6)
            / ((sample_size**2 + sample_size) / 2)
        )
        Z_Critical_Value = norm.ppf((1 - confidence_level) + ((confidence_level) / 2))

        Lower_CI_Matched_Wilcoxon = math.tanh(
            math.atanh(wilcoxon_RBCwilcoxon) - Z_Critical_Value * Standrd_Error_RBR
        )
        Upper_CI_Matched_Wilcoxon = math.tanh(
            math.atanh(wilcoxon_RBCwilcoxon) + Z_Critical_Value * Standrd_Error_RBR
        )
        Lower_CI_Adjusted_Wilcoxon = math.tanh(
            math.atanh(wilcoxon_RBC_adjusted) - Z_Critical_Value * Standrd_Error_RBR
        )
        Upper_CI_Adjusted_Wilcoxon = math.tanh(
            math.atanh(wilcoxon_RBC_adjusted) + Z_Critical_Value * Standrd_Error_RBR
        )
        Lower_CI_Unadjusted_Wilcoxon = math.tanh(
            math.atanh(wilcoxon_RBC_unadjusted) - Z_Critical_Value * Standrd_Error_RBR
        )
        Upper_CI_Unadjusted_Wilcoxon = math.tanh(
            math.atanh(wilcoxon_RBC_unadjusted) + Z_Critical_Value * Standrd_Error_RBR
        )
        Lower_CI_Adjusted_Corrected_Wilcoxon = math.tanh(
            math.atanh(wilcoxon_RBC_unajusted_corrected)
            - Z_Critical_Value * Standrd_Error_RBR
        )
        Upper_CI_Adjusted_Corrected_Wilcoxon = math.tanh(
            math.atanh(wilcoxon_RBC_unajusted_corrected)
            + Z_Critical_Value * Standrd_Error_RBR
        )
        Lower_CI_Unadjusted_Corrected_Wilcoxon = math.tanh(
            math.atanh(wilcoxon_RBC_adjusted_corrected)
            - Z_Critical_Value * Standrd_Error_RBR
        )
        Upper_CI_Unadjusted_Corrected_Wilcoxon = math.tanh(
            math.atanh(wilcoxon_RBC_adjusted_corrected)
            + Z_Critical_Value * Standrd_Error_RBR
        )

        Lower_CI_Matched_Pratt = math.tanh(
            math.atanh(RBCpratt) - Z_Critical_Value * Standrd_Error_RBR
        )
        Upper_CI_Matched_Pratt = math.tanh(
            math.atanh(RBCpratt) + Z_Critical_Value * Standrd_Error_RBR
        )
        Lower_CI_Adjusted_Pratt = math.tanh(
            math.atanh(pratt_RBC_adjusted) - Z_Critical_Value * Standrd_Error_RBR
        )
        Upper_CI_Adjusted_Pratt = math.tanh(
            math.atanh(pratt_RBC_adjusted) + Z_Critical_Value * Standrd_Error_RBR
        )
        Lower_CI_Unadjusted_Pratt = math.tanh(
            math.atanh(pratt_RBC_unadjusted) - Z_Critical_Value * Standrd_Error_RBR
        )
        Upper_CI_Unadjusted_Pratt = math.tanh(
            math.atanh(pratt_RBC_unadjusted) + Z_Critical_Value * Standrd_Error_RBR
        )
        Lower_CI_Adjusted_Corrected_Pratt = math.tanh(
            math.atanh(pratt_RBC_unajusted_corrected)
            - Z_Critical_Value * Standrd_Error_RBR
        )
        Upper_CI_Adjusted_Corrected_Pratt = math.tanh(
            math.atanh(pratt_RBC_unajusted_corrected)
            + Z_Critical_Value * Standrd_Error_RBR
        )
        Lower_CI_Unadjusted_Corrected_Pratt = math.tanh(
            math.atanh(pratt_RBC_adjusted_corrected)
            - Z_Critical_Value * Standrd_Error_RBR
        )
        Upper_CI_Unadjusted_Corrected_Pratt = math.tanh(
            math.atanh(pratt_RBC_adjusted_corrected)
            + Z_Critical_Value * Standrd_Error_RBR
        )

        # Set results
        results = {}

        results["Sample Median 1"] = round(sample_median_1, 4)
        results["Sample Median 2"] = round(sample_median_2, 4)
        results["Sample Mean 1"] = round(sample_mean_1, 4)
        results["Sample Mean 2"] = round(sample_mean_2, 4)
        results["Sample Standard Deviation 1"] = round(sample_standard_deviation_1, 4)
        results["Sample Standard Deviation 2"] = round(sample_standard_deviation_2, 4)
        results["Sample Size"] = sample_size
        results["Pearson Correlation"] = round(pearson_correlation, 4)
        results["Pearson Correlation p-value"] = round(pearson_correlation_pvalue, 4)

        results["Wilcoxon Positive Count"] = wilcoxon_positive_n
        results["Wilcoxon Negative Count"] = wilcoxon_negative_n
        results["Wilcoxon Total Count"] = wilcoxon_total_n
        results["Wilcoxon Positive Sum Ranks"] = wilcoxon_positive_sum_ranks
        results["Wilcoxon Negative Sum Ranks"] = wilcoxon_negative_sum_ranks
        results["Wilcoxon Total Sum Ranks"] = wilcoxon_total_sum_ranks
        results["Wilcoxon Mean Positive Ranks"] = wilcoxon_mean_positive_ranks
        results["Wilcoxon Mean Negative Ranks"] = wilcoxon_mean_negative_ranks
        results["Wilcoxon MeanW"] = wilcoxon_MeanW
        results["Wilcoxon Adjusted Std"] = np.sqrt(wilcoxon_Adjusted_Var)
        results["Wilcoxon Unadjusted Std"] = np.sqrt(wilcoxon_Unadjusted_Var)
        results["Matched Pairs Rank Biserial Correlation (Kerby, 2014)"] = (
            wilcoxon_RBCwilcoxon
        )
        results["Wilcoxon Z Adjusted"] = wilcoxon_Z_adjusted
        results["Wilcoxon Z Unadjusted"] = wilcoxon_Z_Unadjusted
        results["Wilcoxon Z Adjusted Corrected"] = wilcoxon_Z_adjusted_Corrected
        results["Wilcoxon Z Unadjusted Corrected"] = wilcoxon_Z_Unadjusted_Corrected
        results["Wilcoxon p-value Adjusted"] = wilcoxon_p_value_adjusted
        results["Wilcoxon p-value Unadjusted"] = wilcoxon_p_value_unadjusted
        results["Wilcoxon p-value Adjusted Corrected"] = (
            wilcoxon_p_value_adjusted_corrected
        )
        results["Wilcoxon p-value Unadjusted Corrected"] = (
            wilcoxon_p_value_unadjusted_corrected
        )
        results["Wilcoxon RBC Adjusted"] = wilcoxon_RBC_adjusted
        results["Wilcoxon RBC Unadjusted"] = wilcoxon_RBC_unadjusted
        results["Wilcoxon RBC Unadjusted Corrected"] = wilcoxon_RBC_unajusted_corrected
        results["Wilcoxon RBC Adjusted Corrected"] = wilcoxon_RBC_adjusted_corrected

        results["Lower CI RBC Matched Pairs Wilcoxon"] = Lower_CI_Matched_Wilcoxon
        results["Upper CI RBC Matched Pairs Wilcoxon"] = Upper_CI_Matched_Wilcoxon
        results["Lower CI RBC Adjusted Wilcoxon"] = Lower_CI_Adjusted_Wilcoxon
        results["Upper CI RBC Adjusted Wilcoxon"] = Upper_CI_Adjusted_Wilcoxon
        results["Lower CI RBC Unadjusted Wilcoxon"] = Lower_CI_Unadjusted_Wilcoxon
        results["Upper CI RBC Unadjusted Wilcoxon"] = Upper_CI_Unadjusted_Wilcoxon
        results["Lower CI RBC Adjusted Corrected Wilcoxon"] = (
            Lower_CI_Adjusted_Corrected_Wilcoxon
        )
        results["Upper CI RBC Adjusted Corrected Wilcoxon"] = (
            Upper_CI_Adjusted_Corrected_Wilcoxon
        )
        results["Lower CI RBC Unadjusted Corrected Wilcoxon"] = (
            Lower_CI_Unadjusted_Corrected_Wilcoxon
        )
        results["Upper CI RBC Unadjusted Corrected Wilcoxon"] = (
            Upper_CI_Unadjusted_Corrected_Wilcoxon
        )

        results["Pratt Positive Count"] = pratt_positive_n
        results["Pratt Negative Count"] = pratt_negative_n
        results["Pratt Total Count"] = pratt_total_n
        results["Pratt Positive Sum Ranks"] = pratt_positive_sum_ranks
        results["Pratt Negative Sum Ranks"] = pratt_negative_sum_ranks
        results["Pratt Total Sum Ranks"] = pratt_total_sum_ranks
        results["Pratt Mean Positive Ranks"] = pratt_mean_positive_ranks
        results["Pratt Mean Negative Ranks"] = pratt_mean_negative_ranks
        results["Pratt MeanW"] = pratt_MeanW
        results["Pratt Adjusted Std"] = np.sqrt(pratt_Adjusted_Var)
        results["Pratt Unadjusted Std"] = np.sqrt(pratt_Unadjusted_Var)
        results["Matched Pairs Rank Biserial Correlation (Kerby, 2014)"] = RBCpratt
        results["Pratt Z Adjusted"] = pratt_Z_adjusted
        results["Pratt Z Unadjusted"] = pratt_Z_Unadjusted
        results["Pratt Z Adjusted Corrected"] = pratt_Z_adjustedCorrected
        results["Pratt Z Unadjusted Corrected"] = pratt_Z_Unadjusted_Corrected
        results["Pratt p-value Adjusted"] = pratt_p_value_adjusted
        results["Pratt p-value Unadjusted"] = pratt_p_value_unadjusted
        results["Pratt p-value Adjusted Corrected"] = pratt_p_value_adjusted_corrected
        results["Pratt p-value Unadjusted Corrected"] = (
            pratt_p_value_unadjusted_corrected
        )
        results["Pratt RBC Adjusted"] = pratt_RBC_adjusted
        results["Pratt RBC Unadjusted"] = pratt_RBC_unadjusted
        results["Pratt RBC Unadjusted Corrected"] = pratt_RBC_unajusted_corrected
        results["Pratt RBC Adjusted Corrected"] = pratt_RBC_adjusted_corrected

        results["Lower CI RBC Matched Pairs Pratt"] = Lower_CI_Matched_Pratt
        results["Upper CI RBC Matched Pairs Pratt"] = Upper_CI_Matched_Pratt
        results["Lower CI RBC Adjusted Pratt"] = Lower_CI_Adjusted_Pratt
        results["Upper CI RBC Adjusted Pratt"] = Upper_CI_Adjusted_Pratt
        results["Lower CI RBC Unadjusted Pratt"] = Lower_CI_Unadjusted_Pratt
        results["Upper CI RBC Unadjusted Pratt"] = Upper_CI_Unadjusted_Pratt
        results["Lower CI RBC Adjusted Corrected Pratt"] = (
            Lower_CI_Adjusted_Corrected_Pratt
        )
        results["Upper CI RBC Adjusted Corrected Pratt"] = (
            Upper_CI_Adjusted_Corrected_Pratt
        )
        results["Lower CI RBC Unadjusted Corrected Pratt"] = (
            Lower_CI_Unadjusted_Corrected_Pratt
        )
        results["Upper CI RBC Unadjusted Corrected Pratt"] = (
            Upper_CI_Unadjusted_Corrected_Pratt
        )
        formatted_p_value = (
            "{:.3f}".format(pratt_p_value_adjusted_corrected).lstrip("0")
            if pratt_p_value_adjusted_corrected >= 0.001
            else "\033[3mp\033[0m < .001"
        )
        results["Statistical Line"] = (
            " \033[3mz\033[0m = {:.3f}, {}{}, \033[3mr\033[0m\u2098\u209a = {}, {}% CI(fisher) [{},{}]".format(
                pratt_Z_adjustedCorrected,
                (
                    "\033[3mp = \033[0m"
                    if pratt_p_value_adjusted_corrected >= 0.001
                    else ""
                ),
                formatted_p_value,
                str(round(pratt_RBC_adjusted_corrected, 9)).lstrip("0"),
                (confidence_level * 100),
                str(round(Lower_CI_Matched_Pratt, 3)).lstrip("0"),
                str(round(Upper_CI_Matched_Pratt, 4)).lstrip("0"),
            )
        )

        return results

    # Things to Consider
    # 1. Consider adding other CI's for example metsamuuronen method for sommers delta (which in the case of two groups equals the rank biserial correlation)
