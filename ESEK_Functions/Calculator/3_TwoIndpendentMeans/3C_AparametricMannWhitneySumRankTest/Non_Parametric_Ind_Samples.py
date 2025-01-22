
#####################################################
## Effect Size for Aparametric Independent Samples ##
#####################################################

import numpy as np
from scipy.stats import norm, rankdata, mannwhitneyu
import math 
from collections import Counter


class Aparametric_Paired_Samples():
    @staticmethod
    def Apermetric_Effect_Size_Independent(params: dict) -> dict: #Aparametric effect size with sign test

        # Set Parameters
        column_1 = params["Column 1"]
        column_2 = params["Column 2"]
        confidence_level_percentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentages / 100
        sample_median_1 = np.median(column_1)
        sample_median_2 = np.median(column_2)
        sample_mean_1 = np.mean(column_1)
        sample_mean_2 = np.mean(column_2)    
        sample_size_1 = len(column_1)
        sample_size_2 = len(column_2)
        sample_size = sample_size_1 + sample_size_2 

        # Mean and sum of the ranks
        merged_samples = np.append(column_1,column_2)
        ranks = rankdata(merged_samples, method = "average")
        sum_ranks_1 = np.sum(ranks[:sample_size_1]) # W1 statistic
        sum_ranks_2 = np.sum(ranks[sample_size_1:]) # W2 statistic
        mean_ranks_1 = np.mean(ranks[:sample_size_1])
        mean_ranks_2 = np.mean(ranks[sample_size_1:])

        #Handle Ties
        freq = Counter(merged_samples)
        frequencies = list(freq.values()) # Counting the frequqncies in an array
        tiescorrection = [(f ** 3 - f) for f in frequencies]; 
        multiplicity_factor = sum(tiescorrection) #This is a correction factor to compensate on ties in the data

        # Wilcoxon Mann Whitney - Test statistics
        Mean_W_1 = (sample_size_1*(sample_size+1))/2
        Mean_W_2 = (sample_size_2*(sample_size+1))/2
        U_statistic_1 = sum_ranks_1 - (sample_size_1*(sample_size_1+1)) / 2
        U_statistic_2 = sum_ranks_2 - (sample_size_2*(sample_size_2+1)) / 2
        Variance = ((sample_size_1*sample_size_2*(sample_size_1+sample_size_2+1))/12) - ((sample_size_1*sample_size_2*(multiplicity_factor)))/ (12*sample_size*(sample_size-1))

        z_score = abs(sum_ranks_1 - Mean_W_1 )  / np.sqrt(Variance)
        z_score_corrected = (abs(sum_ranks_1 - Mean_W_1) - 0.5) /np.sqrt(Variance)

        p_value = min(float(norm.sf((abs(z_score))) * 2), 0.99999)
        p_value_corrected = min(float(norm.sf((abs(z_score_corrected))) * 2), 0.99999)

        #Effect Sizes Rank Biserial Correlation Family
        rank_biserial_correlation = (U_statistic_1 - U_statistic_2) / (U_statistic_1 + U_statistic_2) # This is just one way to get this rank biserial correlation - there are at least 6 ways to get to these same results (Cliffs delta, Sommers Delta, Wundt's formula, Cureton formula, U-stat based formula and Glass Formula) 
        Rosenthal_rank_biserial_correlation_z = z_score/np.sqrt(sample_size)
        Rosenthal_rank_biserial_correlation_z_corrected = z_score_corrected/ np.sqrt(sample_size)

        exact_p_value = mannwhitneyu(column_1,column_2, method = "exact")[1] # This is one version of the exact p-value

        # Fisher Based Confidence Intervals
        Standrd_Error_RBC = np.sqrt((sample_size_1+sample_size_2+1)/(3*sample_size_1+sample_size_2)) #see totser package formulae for paired data as well
        Z_Critical_Value = norm.ppf((1-confidence_level) + ((confidence_level) / 2))

        Lower_CI_Rank_Biserial_Correlation = max(math.tanh(math.atanh(rank_biserial_correlation) - Z_Critical_Value * Standrd_Error_RBC),-1)
        Upper_CI_Rank_Biserial_Correlation = min(math.tanh(math.atanh(rank_biserial_correlation) + Z_Critical_Value * Standrd_Error_RBC),1)
        Lower_CI_Z_basesd = max(math.tanh(math.atanh(Rosenthal_rank_biserial_correlation_z) - Z_Critical_Value * Standrd_Error_RBC),-1)
        Upper_CI_Z_basesd = min(math.tanh(math.atanh(Rosenthal_rank_biserial_correlation_z) + Z_Critical_Value * Standrd_Error_RBC),1)
        Lower_CI_Z_basesd_corrected = max(math.tanh(math.atanh(Rosenthal_rank_biserial_correlation_z_corrected) - Z_Critical_Value * Standrd_Error_RBC),-1)
        Upper_CI_Z_basesd_corrected = min(math.tanh(math.atanh(Rosenthal_rank_biserial_correlation_z_corrected) + Z_Critical_Value * Standrd_Error_RBC),1)

        results = {}
        results["Sample Mean 1"] = round(sample_mean_1, 4)
        results["Sample Mean 2"] = round(sample_mean_2, 4)
        results["Sample Median 1"] = round(sample_median_1, 4)
        results["Sample Median 2"] = round(sample_median_2, 4)
        results["Sample Size 1"] = sample_size_1
        results["Sample Size 2"] = sample_size_2

        results["Sum of the Ranks 1 (W1)"] = round(sum_ranks_1, 4)
        results["Sum of the Ranks 2 (W2))"] = round(sum_ranks_2, 4)
        results["Mean Ranks 1"] = round(mean_ranks_1, 4)
        results["Mean Ranks 2"] = round(mean_ranks_2, 4)
        results["U1 Statistic"] = round(U_statistic_1, 4)
        results["U2 Statistic"] = round(U_statistic_2, 4)
        
        results["Mean W 1"] = round(Mean_W_1, 4)
        results["Mean W 2"] = round(Mean_W_2, 4)
        results["Variance"] = round(Variance, 4)
        results["Standard Deviation"] = round(np.sqrt(Variance), 4)

        results["z"] = round(z_score, 4)
        results["zcorrected"] = round(z_score_corrected, 4)
        results["p-value"] = round(p_value, 6)
        results["p-value corrected"] = round(p_value_corrected, 6)
        results["Exact p-value"] = round(exact_p_value, 6)

        #Rank Biserial Correlation Effect Sizes
        results["Rank Biserial Correlation (Cureton, Glass, Kerby)"] = round(rank_biserial_correlation, 4)
        results["Rank Biserial Correlation Based on Z Score (Rosenthal)"] = round(Rosenthal_rank_biserial_correlation_z, 4)
        results["Rank Biserial Correlation Based on Corrected Z Score (Rosenthal)"] = round(Rosenthal_rank_biserial_correlation_z_corrected, 6)

        # Statistical Lines
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        formatted_p_value_corrected = "{:.3f}".format(p_value_corrected).lstrip('0') if p_value_corrected >= 0.001 else "\033[3mp\033[0m < .001"

        results["Statistical Line RBC"] = " \033[3mU\033[0m = {}, \033[3mZcorrcted\033[0m = {:.3f}, {}{}, \033[3mRBip\033[0m = {}, {}% CI(Fisher) [{}, {}]".format(((int(U_statistic_1) if float(U_statistic_1).is_integer() else U_statistic_1)), z_score_corrected, '\033[3mp = \033[0m' if p_value_corrected >= 0.001 else '', formatted_p_value_corrected, (('-' if str(rank_biserial_correlation).startswith('-') else '') + str(round(rank_biserial_correlation,3)).lstrip('-').lstrip('0') or '0'), (confidence_level_percentages), (('-' if str(Lower_CI_Rank_Biserial_Correlation).startswith('-') else '') + str(round(Lower_CI_Rank_Biserial_Correlation,3)).lstrip('-').lstrip('0') or '0'), (('-' if str(Upper_CI_Rank_Biserial_Correlation).startswith('-') else '') + str(round(Upper_CI_Rank_Biserial_Correlation,3)).lstrip('-').lstrip('0') or '0'))
        results["Statistical Line Z-based RBC"] = " \033[3mU\033[0m = {}, \033[3mZ\033[0m = {:.3f}, {}{}, \033[3mRBz\033[0m = {}, {}% CI(Fisher) [{}, {}]".format(((int(U_statistic_1) if float(U_statistic_1).is_integer() else U_statistic_1)), z_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, (('-' if str(Rosenthal_rank_biserial_correlation_z).startswith('-') else '') + str(round(Rosenthal_rank_biserial_correlation_z,3)).lstrip('-').lstrip('0') or '0'), (confidence_level_percentages), (('-' if str(Lower_CI_Z_basesd).startswith('-') else '') + str(round(Lower_CI_Z_basesd,3)).lstrip('-').lstrip('0') or '0'), (('-' if str(Upper_CI_Z_basesd).startswith('-') else '') + str(round(Upper_CI_Z_basesd,3)).lstrip('-').lstrip('0') or '0'))
        results["Statistical Line Corrected Z-based RBC"] = " \033[3mU\033[0m = {}, \033[3mZcorrected\033[0m = {:.3f}, {}{}, \033[3mRBz\033[0m = {}, {}% CI(Fisher) [{}, {}]".format(((int(U_statistic_1) if float(U_statistic_1).is_integer() else U_statistic_1)), z_score_corrected, '\033[3mp = \033[0m' if p_value_corrected >= 0.001 else '', formatted_p_value_corrected, (('-' if str(Rosenthal_rank_biserial_correlation_z_corrected).startswith('-') else '') + str(round(Rosenthal_rank_biserial_correlation_z_corrected,3)).lstrip('-').lstrip('0') or '0'), (confidence_level_percentages), (('-' if str(Lower_CI_Z_basesd_corrected).startswith('-') else '') + str(round(Lower_CI_Z_basesd_corrected,3)).lstrip('-').lstrip('0') or '0'), (('-' if str(Upper_CI_Z_basesd_corrected).startswith('-') else '') + str(round(Upper_CI_Z_basesd_corrected,3)).lstrip('-').lstrip('0') or '0'))


 
    
        return results
    
    # Things to Consider
    # 1. Consider adding other CI's for example metsamuuronen method for sommers delta (which in the case of two groups equals the rank biserial correlation)
    # 2. Consider adding the gamma rank correlation found in metsamuuronen
    # 3. Consider Adding the calculation from U-score
    # 4. Add a comment that one can also use the cles family in CLES options




