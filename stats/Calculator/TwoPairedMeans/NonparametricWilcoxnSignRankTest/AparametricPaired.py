#####################################################
## Effect Size for Aparametric Paired Samples Test ##
#####################################################

import numpy as np
from scipy.stats import norm, rankdata, median_abs_deviation, mannwhitneyu
import math 

class Aparametric_Paired_Samples():
    @staticmethod
    def Apermetric_Effect_Size_Dependent(params: dict) -> dict:
        """
        Calculate the effect size for a nonparametric paired samples test.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            - "Column 1": First sample measurements
            - "Column 2": Second sample measurements
            - "Difference in the Population": Expected difference between populations
            - "Confidence Level": Confidence level in percentages
            
        Returns
        -------
        dict
            Dictionary containing:
            - Various rank statistics and effect sizes
            - Test statistics and p-values
            - Confidence intervals
            - Statistical lines in APA format
        """
        
        # Set Parameters
        column_1 = params["Column 1"]
        column_2 = params["Column 2"]
        population_difference = params["Difference in the Population"]
        confidence_level_percentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentages / 100

        # General Summary Statistics        
        sample_median_1 = np.median(column_1)
        sample_median_2 = np.median(column_2)
        sample_mean_1 = np.mean(column_1)
        sample_mean_2 = np.mean(column_2)
        sample_standard_deviation_1 = np.std(column_1, ddof=1)
        sample_standard_deviation_2 = np.std(column_2, ddof=1)
        difference = (column_1 - column_2) - population_difference
        positive_n = difference[difference > 0].shape[0] # How many times group1 is larger than group 2
        negative_n = difference[difference < 0].shape[0] # How many times group2 is ilarger than group 1
        zero_n = difference[difference == 0].shape[0] # Number of ties
        sample_size = len(difference)
        Median_Differnece = np.median(difference)
        Median_Abs_Deviation = median_abs_deviation(difference)

        # Summary Statistics for the Wilcoxon Sign Rank Test not Considering ties         
        difference_no_ties = difference[difference != 0] # This line removes the ties
        ranked_no_ties =  rankdata(abs(difference_no_ties))
        positive_sum_ranks_no_ties = ranked_no_ties[difference_no_ties > 0].sum()
        negative_sum_ranks_no_ties = ranked_no_ties[difference_no_ties < 0].sum()
 
        # Summary Statistics for the Wilcoxon Sign Rank Considering ties         
        ranked_with_ties =  rankdata(abs(difference))
        positive_sum_ranks_with_ties = ranked_with_ties[difference > 0].sum()
        negative_sum_ranks_with_ties = ranked_with_ties[difference < 0].sum()

        # Wilcoxon Sign Rank Test Statistics Non Considering Ties (Wilcoxon Method)
        MeanW_not_considering_ties = (positive_sum_ranks_no_ties + negative_sum_ranks_no_ties) / 2
        sign_no_ties = np.where(difference_no_ties == 0, 0, (np.where(difference_no_ties < 0, -1, 1)))
        ranked_signs_no_ties = (sign_no_ties * ranked_no_ties)
        ranked_signs_no_ties = np.where(difference_no_ties == 0, 0, ranked_signs_no_ties)
        unadjusted_variance_wilcoxon = (len(difference_no_ties) * (len(difference_no_ties) +1) *(2*(len(difference_no_ties))+1)) / 24
        var_adj_T = (ranked_signs_no_ties * ranked_signs_no_ties).sum()
        Adjusted_Variance_wilcoxon = (1/4) * var_adj_T

        #Calculate The Z score wilcox # Just for debbuging and comparing between softwares i also return the unadjusted versions (most softwares adjust the variance)
        Z_Numerator_wilcoxon = (positive_sum_ranks_no_ties - MeanW_not_considering_ties)
        Z_Numerator_wilcoxon = np.where(Z_Numerator_wilcoxon < 0, Z_Numerator_wilcoxon + 0.5, Z_Numerator_wilcoxon)
        
        Z_adjusted_wilcoxon = (Z_Numerator_wilcoxon) /  np.sqrt(Adjusted_Variance_wilcoxon)
        Z_adjusted_Normal_Approxinmation_wilcoxon = (Z_Numerator_wilcoxon - 0.5) /  np.sqrt(Adjusted_Variance_wilcoxon)
        Z_unadjusted_wilcoxon = (Z_Numerator_wilcoxon) /  np.sqrt(unadjusted_variance_wilcoxon)
        Z_unadjusted_Normal_Approxinmation_wilcoxon = (Z_Numerator_wilcoxon - 0.5) /  np.sqrt(unadjusted_variance_wilcoxon)
        p_value_adjusted_wilcoxon = min(float(norm.sf((abs(Z_adjusted_wilcoxon))) * 2), 0.99999)
        p_value_adjusted_Normal_Approxinmation_wilcoxon = min(float(norm.sf((abs(Z_adjusted_Normal_Approxinmation_wilcoxon))) * 2), 0.99999)
        p_value_unadjusted_wilcoxon = min(float(norm.sf((abs(Z_unadjusted_wilcoxon))) * 2), 0.99999)
        p_value_Unadjusted_Normal_Approxinmation_wilcoxon = min(float(norm.sf((abs(Z_unadjusted_Normal_Approxinmation_wilcoxon))) * 2), 0.99999)
  
        # Wilcoxon Sign Rank Test Statistics Considering Ties (Pratt Method)
        MeanW_considering_ties = (positive_sum_ranks_with_ties + negative_sum_ranks_with_ties) / 2
        sign_with_ties = np.where(difference == 0, 0, (np.where(difference < 0, -1, 1)))
        ranked_signs_with_ties = (sign_with_ties * ranked_with_ties)
        ranked_signs_with_ties = np.where(difference == 0, 0, ranked_signs_with_ties)
        var_adj_T_with_ties = (ranked_signs_with_ties * ranked_signs_with_ties).sum()
        Adjusted_Variance_pratt = (1/4) * var_adj_T_with_ties

        Z_Numerator_pratt = (positive_sum_ranks_with_ties - MeanW_considering_ties)
        Z_Numerator_pratt = np.where(Z_Numerator_pratt < 0, Z_Numerator_pratt + 0.5, Z_Numerator_pratt)     
        
        Z_adjusted_pratt = (Z_Numerator_pratt) /  np.sqrt(Adjusted_Variance_pratt)
        Z_adjusted_Normal_Approxinmation_pratt = (Z_Numerator_pratt - 0.5) /  np.sqrt(Adjusted_Variance_pratt)
        p_value_adjusted_pratt = min(float(norm.sf((abs(Z_adjusted_pratt))) * 2), 0.99999) 
        p_value_adjusted_Normal_Approximation_pratt= min(float(norm.sf((abs(Z_adjusted_Normal_Approxinmation_pratt))) * 2), 0.99999) 

        # Matched Pairs Rank Biserial Correlation
        Matched_Pairs_Rank_Biserial_Corelation_ignoring_ties = (positive_sum_ranks_no_ties-negative_sum_ranks_no_ties)/np.sum(ranked_no_ties) #This is the match paired rank biserial correlation using kerby formula that is not considerign ties (Kerby, 2014)
        Matched_Pairs_Rank_Biserial_Corelation_Considering_ties = (positive_sum_ranks_with_ties-negative_sum_ranks_with_ties)/np.sum(ranked_with_ties) # this is the Kerby 2014 Formula - (With ties one can apply either Kerby or King Minium Formulae but not cureton - King's Formula is the most safe)

        # Z-based Rank Biserial Correlation (Note that since the Wilcoxon method is ignoring ties the sample size should actually be the number of the non tied paires)
        Z_based_Rank_Biserial_Correlation_no_ties = Z_adjusted_wilcoxon / np.sqrt(len(ranked_no_ties))
        Z_based_Rank_Biserial_Correlation_corrected_no_ties = Z_adjusted_Normal_Approxinmation_wilcoxon / np.sqrt(len(ranked_no_ties))
        Z_based_Rank_Biserial_Correlation_with_ties = Z_adjusted_pratt / np.sqrt(sample_size)
        Z_based_Rank_Biserial_Correlation_corrected_with_ties = Z_adjusted_Normal_Approxinmation_pratt / np.sqrt(sample_size)

        # Confidence Intervals 
        Standrd_Error_Match_Pairs_Rank_Biserial_Corelation_no_ties = np.sqrt(( (2*(len(ranked_no_ties))**3 + 3* (len(ranked_no_ties))**2 + (len(ranked_no_ties)))/6) / (((len(ranked_no_ties))**2 + (len(ranked_no_ties)))/2))
        Standrd_Error_Match_Pairs_Rank_Biserial_Corelation_with_ties = np.sqrt(( (2*sample_size**3 + 3* sample_size**2 + sample_size)/6) / ((sample_size**2 + sample_size)/2))
        Z_Critical_Value = norm.ppf((1-confidence_level) + ((confidence_level) / 2))

        Lower_CI_Matched_Pairs_Wilcoxon = max(math.tanh(math.atanh(Matched_Pairs_Rank_Biserial_Corelation_ignoring_ties) - Z_Critical_Value * Standrd_Error_Match_Pairs_Rank_Biserial_Corelation_no_ties),-1)
        Upper_CI_Matched_Pairs_Wilcoxon = min(math.tanh(math.atanh(Matched_Pairs_Rank_Biserial_Corelation_ignoring_ties) + Z_Critical_Value * Standrd_Error_Match_Pairs_Rank_Biserial_Corelation_no_ties),1)
        Lower_CI_Z_basesd_Wilcoxon = max(math.tanh(math.atanh(Z_based_Rank_Biserial_Correlation_no_ties) - Z_Critical_Value * Standrd_Error_Match_Pairs_Rank_Biserial_Corelation_no_ties),-1)
        Upper_CI_Z_basesd_Wilcoxon = min(math.tanh(math.atanh(Z_based_Rank_Biserial_Correlation_no_ties) + Z_Critical_Value * Standrd_Error_Match_Pairs_Rank_Biserial_Corelation_no_ties),1)
        Lower_CI_Z_basesd_corrected_Wilcoxon = max(math.tanh(math.atanh(Z_based_Rank_Biserial_Correlation_corrected_no_ties) - Z_Critical_Value * Standrd_Error_Match_Pairs_Rank_Biserial_Corelation_no_ties),-1)
        Upper_CI_Z_basesd_corrected_Wilcoxon = min(math.tanh(math.atanh(Z_based_Rank_Biserial_Correlation_corrected_no_ties) + Z_Critical_Value * Standrd_Error_Match_Pairs_Rank_Biserial_Corelation_no_ties),1)
        
        Lower_CI_Matched_Pairs_Pratt = max(math.tanh(math.atanh(Matched_Pairs_Rank_Biserial_Corelation_Considering_ties) - Z_Critical_Value * Standrd_Error_Match_Pairs_Rank_Biserial_Corelation_with_ties),-1)
        Upper_CI_Matched_Pairs_Pratt = min(math.tanh(math.atanh(Matched_Pairs_Rank_Biserial_Corelation_Considering_ties) + Z_Critical_Value * Standrd_Error_Match_Pairs_Rank_Biserial_Corelation_with_ties),1)
        Lower_CI_Z_basesd_Pratt = max(math.tanh(math.atanh(Z_based_Rank_Biserial_Correlation_with_ties) - Z_Critical_Value * Standrd_Error_Match_Pairs_Rank_Biserial_Corelation_with_ties),-1)
        Upper_CI_Z_basesd_Pratt = min(math.tanh(math.atanh(Z_based_Rank_Biserial_Correlation_with_ties) + Z_Critical_Value * Standrd_Error_Match_Pairs_Rank_Biserial_Corelation_with_ties),1)
        Lower_CI_Z_basesd_corrected_Pratt = max(math.tanh(math.atanh(Z_based_Rank_Biserial_Correlation_corrected_with_ties) - Z_Critical_Value * Standrd_Error_Match_Pairs_Rank_Biserial_Corelation_with_ties),-1)
        Upper_CI_Z_basesd_corrected_Pratt = min(math.tanh(math.atanh(Z_based_Rank_Biserial_Correlation_corrected_with_ties) + Z_Critical_Value * Standrd_Error_Match_Pairs_Rank_Biserial_Corelation_with_ties),1)


        # Set results
        results = {}

        # General Summary Statistics 
        results["Sample Median 1"] = round(sample_median_1, 4)
        results["Sample Median 2"] = round(sample_median_2, 4)
        results["Median of the Differnece"] = Median_Differnece
        results["Median of Absoult Deviation"] = Median_Abs_Deviation
        results["Sample Mean 1"] = round(sample_mean_1, 4)
        results["Sample Mean 2"] = round(sample_mean_2, 4)
        results["Sample Standard Deviation 1"] = round(sample_standard_deviation_1, 4)
        results["Sample Standard Deviation 2"] = round(sample_standard_deviation_2, 4)
        results["Number of Pairs"] = sample_size
        results["Number of Pairs with a Sign"] = len(ranked_no_ties)
        results["Number of times Group 1 is Larger"] = positive_n
        results["Number of times Group 2 is larger"] = round(negative_n, 4)
        results["Number of times Group 1 and 2 are Equal"] = zero_n

        # Wilcoxon Statistics (Wilcoxon Method that Ignores ties) 
        results["Wilcoxon Method"] = ''
        results["_______________"] = ''
        results["Sum of the Positive Ranks (without ties)"] = round(positive_sum_ranks_no_ties, 4)
        results["Sum of the Negative Ranks (without ties)"] = round(negative_sum_ranks_no_ties, 4)
 
        # Wilcoxon Sign Rank Test Statistics (WIlcoxon)
        results["Wilcoxon Mean W (Without ties)"] = MeanW_not_considering_ties
        results["Wilcoxon Standard Deviation"] = np.sqrt(Adjusted_Variance_wilcoxon)
        results["Wilcoxon Z"] = Z_adjusted_wilcoxon
        results["Wilcoxon Z With Normal Approximation (Continuiety Correction)"] = Z_adjusted_Normal_Approxinmation_wilcoxon
        results["Wilcoxon p-value"] = p_value_adjusted_wilcoxon
        results["Wilcoxon p-value with Normal Approximation (Continuiety Correction)"] = p_value_adjusted_Normal_Approxinmation_wilcoxon
        results["Wilcoxon p-value2)"] = p_value_unadjusted_wilcoxon
        results["Wilcoxon p-value3)"] = p_value_Unadjusted_Normal_Approxinmation_wilcoxon

        # Rank Biserial Correlation
        results["Matched Pairs Rank Biserial Correlation (Ignoring Ties)"] = round(Matched_Pairs_Rank_Biserial_Corelation_ignoring_ties, 5)
        results["Z-based Rank Biserial Correlation (Wilcoxon Method) "] = round(Z_based_Rank_Biserial_Correlation_no_ties, 5)
        results["Z-based Corrected Rank Biserial Correlation (Wilcoxon Method)"] = round(Z_based_Rank_Biserial_Correlation_corrected_no_ties, 5)

        # Confidence Intervals
        results["Standard Error of the Matched Pairs Rank Biserial Correlation (Wilcoxon Method)"] = round(Standrd_Error_Match_Pairs_Rank_Biserial_Corelation_no_ties, 4)
        results["Lower CI Matched Pairs Rank Biserial Wilcoxon"] = round(Lower_CI_Matched_Pairs_Wilcoxon,5)
        results["Upper CI Matched Pairs Rank Biserial Wilcoxon"] = round(Upper_CI_Matched_Pairs_Wilcoxon,5)
        results["Lower CI Z-based Rank Biserial Wilcoxon"] = round(Lower_CI_Z_basesd_Wilcoxon,5)
        results["Upper CI Z-based Rank Biserial Wilcoxon"] = round(Upper_CI_Z_basesd_Wilcoxon,5)
        results["Lower CI Z-based Corrected Rank Biserial Wilcoxon"] = round(Lower_CI_Z_basesd_corrected_Wilcoxon,5)
        results["Upper CI Z-based Corrected Rank Biserial Wilcoxon"] = round(Upper_CI_Z_basesd_corrected_Wilcoxon,5)  
        
        # Statistical Lines
    
        formatted_p_value_wilcoxon = "{:.3f}".format(p_value_adjusted_wilcoxon).lstrip('0') if p_value_adjusted_wilcoxon >= 0.001 else "\033[3mp\033[0m < .001"
        formatted_p_value_wilcoxon_corrected = "{:.3f}".format(p_value_adjusted_Normal_Approxinmation_wilcoxon).lstrip('0') if p_value_adjusted_Normal_Approxinmation_wilcoxon >= 0.001 else "\033[3mp\033[0m < .001"

        results["Statistical Line Wilcoxon"] = " \033[3mTw\033[0m = {}, \033[3mZ\033[0m = {:.3f}, {}{}, \033[3mRBz\033[0m = {}, {}% CI(Fisher) [{}, {}]".format((int(positive_sum_ranks_no_ties) if float(positive_sum_ranks_no_ties).is_integer() else positive_sum_ranks_no_ties), Z_adjusted_wilcoxon, '\033[3mp = \033[0m' if p_value_adjusted_wilcoxon >= 0.001 else '', formatted_p_value_wilcoxon, (('-' if str(Z_based_Rank_Biserial_Correlation_no_ties).startswith('-') else '') + str(round(Z_based_Rank_Biserial_Correlation_no_ties,3)).lstrip('-').lstrip('0') or '0'), (confidence_level_percentages), (('-' if str(Lower_CI_Z_basesd_Wilcoxon).startswith('-') else '') + str(round(Lower_CI_Z_basesd_Wilcoxon,3)).lstrip('-').lstrip('0') or '0'), (('-' if str(Upper_CI_Z_basesd_Wilcoxon).startswith('-') else '') + str(round(Upper_CI_Z_basesd_Wilcoxon,3)).lstrip('-').lstrip('0') or '0'))
        results["Statistical Line Wilcoxon Corrected"] = " \033[3mTw\033[0m = {}, \033[3mZcorrected\033[0m = {:.3f}, {}{}, \033[3mRBz\033[0m = {}, {}% CI(Fisher) [{}, {}]".format((int(positive_sum_ranks_no_ties) if float(positive_sum_ranks_no_ties).is_integer() else positive_sum_ranks_no_ties), Z_adjusted_Normal_Approxinmation_wilcoxon, '\033[3mp = \033[0m' if confidence_level >= 0.001 else '', formatted_p_value_wilcoxon_corrected, (('-' if str(Z_based_Rank_Biserial_Correlation_corrected_no_ties).startswith('-') else '') + str(round(Z_based_Rank_Biserial_Correlation_corrected_no_ties,3)).lstrip('-').lstrip('0') or '0'), (confidence_level_percentages),(('-' if str(Lower_CI_Z_basesd_corrected_Wilcoxon).startswith('-') else '') + str(round(Lower_CI_Z_basesd_corrected_Wilcoxon,3)).lstrip('-').lstrip('0') or '0'), (('-' if str(Upper_CI_Z_basesd_corrected_Wilcoxon).startswith('-') else '') + str(round(Upper_CI_Z_basesd_corrected_Wilcoxon,3)).lstrip('-').lstrip('0') or '0'))
        results["Statistical Line Wilcoxon Matched Pairs"] = " \033[3mTw\033[0m = {}, \033[3mZcorrected\033[0m = {:.3f}, {}{}, \033[3mRBmp\033[0m = {}, {}% CI(Fisher) [{}, {}]".format(((int(positive_sum_ranks_no_ties) if float(positive_sum_ranks_no_ties).is_integer() else positive_sum_ranks_no_ties)), Z_adjusted_Normal_Approxinmation_wilcoxon, '\033[3mp = \033[0m' if confidence_level >= 0.001 else '', formatted_p_value_wilcoxon_corrected, (('-' if str(Matched_Pairs_Rank_Biserial_Corelation_ignoring_ties).startswith('-') else '') + str(round(Matched_Pairs_Rank_Biserial_Corelation_ignoring_ties,3)).lstrip('-').lstrip('0') or '0'), (confidence_level_percentages), (('-' if str(Lower_CI_Matched_Pairs_Wilcoxon).startswith('-') else '') + str(round(Lower_CI_Matched_Pairs_Wilcoxon,3)).lstrip('-').lstrip('0') or '0'), (('-' if str(Upper_CI_Matched_Pairs_Wilcoxon).startswith('-') else '') + str(round(Upper_CI_Matched_Pairs_Wilcoxon,3)).lstrip('-').lstrip('0') or '0'))

        results["Pratt Method"] = ''
        results["______________"] = ''
        results["Sum of the Positive Ranks (with ties)"] = round(positive_sum_ranks_with_ties, 4)
        results["Sum of the Negative Ranks (with ties)"] = round(negative_sum_ranks_with_ties, 4)

        results["Pratt MeanW (Considering Ties)"] = MeanW_considering_ties
        results["Pratt Standard Deviation"] = np.sqrt(Adjusted_Variance_pratt)
        results["Pratt Z"] = Z_adjusted_pratt
        results["Pratt Z with Normal Approximation (Continuiety Correction)"] = Z_adjusted_Normal_Approxinmation_pratt
        results["Pratt p-value"] = p_value_adjusted_pratt
        results["Pratt p-value with Normal Approximation (Continuiety Correction)"] = p_value_adjusted_Normal_Approximation_pratt
      
        # Rank Biserial Correlation
        results["Matched Pairs Rank Biserial Correlation (Considering Ties) "] = round(Matched_Pairs_Rank_Biserial_Corelation_Considering_ties, 5)
        results["Z-based Rank Biserial Correlation (Pratt Method)"] = round(Z_based_Rank_Biserial_Correlation_with_ties, 5)
        results["Z-based Corrected Rank Biserial Correlation (Pratt Method)"] = round(Z_based_Rank_Biserial_Correlation_corrected_with_ties, 5)

        # Confidence Intervals
        results["Standard Error of the Matched Pairs Rank Biserial Correlation (Pratt Method)"] = round(Standrd_Error_Match_Pairs_Rank_Biserial_Corelation_with_ties, 4)
        results["Lower CI Matched Pairs Rank Biserial Pratt"] = round(Lower_CI_Matched_Pairs_Pratt,5)
        results["Upper CI Matched Pairs Rank Biserial Pratt"] = round(Upper_CI_Matched_Pairs_Pratt,5)
        results["Lower CI Z-based Rank Biserial Pratt"] = round(Lower_CI_Z_basesd_Pratt,5)
        results["Upper CI Z-based Rank Biserial Pratt"] = round(Upper_CI_Z_basesd_Pratt,5)
        results["Lower CI Z-based Corrected Rank Biserial Pratt"] = round(Lower_CI_Z_basesd_corrected_Pratt,5)
        results["Upper CI Z-based Corrected Rank Biserial  Pratt"] = round(Upper_CI_Z_basesd_corrected_Pratt,5)

        # Statistical Lines In APA Style - everything here is Tailored for the exact APA rules
        formatted_p_value_pratt = "{:.3f}".format(p_value_adjusted_pratt).lstrip('0') if p_value_adjusted_pratt >= 0.001 else "\033[3mp\033[0m < .001"
        formatted_p_value_pratt_corrected = "{:.3f}".format(p_value_adjusted_Normal_Approximation_pratt).lstrip('0') if p_value_adjusted_Normal_Approximation_pratt >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Pratt"] = " \033[3mTp\033[0m = {}, \033[3mZ\033[0m = {:.3f}, {}{}, \033[3mRBz\033[0m = {}, {}% CI(Fisher) [{}, {}]".format(((int(positive_sum_ranks_with_ties) if float(positive_sum_ranks_with_ties).is_integer() else positive_sum_ranks_with_ties)), Z_adjusted_pratt, '\033[3mp = \033[0m' if p_value_adjusted_pratt >= 0.001 else '', formatted_p_value_pratt, (('-' if str(Z_based_Rank_Biserial_Correlation_with_ties).startswith('-') else '') + str(round(Z_based_Rank_Biserial_Correlation_with_ties,3)).lstrip('-').lstrip('0') or '0'), (confidence_level_percentages), (('-' if str(Lower_CI_Z_basesd_Pratt).startswith('-') else '') + str(round(Lower_CI_Z_basesd_Pratt,3)).lstrip('-').lstrip('0') or '0'), (('-' if str(Upper_CI_Z_basesd_Pratt).startswith('-') else '') + str(round(Upper_CI_Z_basesd_Pratt,3)).lstrip('-').lstrip('0') or '0'))
        results["Statistical Line Pratt Corrected"] = " \033[3mTp\033[0m = {}, \033[3mZcorrected\033[0m = {:.3f}, {}{}, \033[3mRBz\033[0m = {}, {}% CI(Fisher) [{}, {}]".format(((int(positive_sum_ranks_with_ties) if float(positive_sum_ranks_with_ties).is_integer() else positive_sum_ranks_with_ties)), Z_adjusted_Normal_Approxinmation_pratt, '\033[3mp = \033[0m' if p_value_adjusted_Normal_Approximation_pratt >= 0.001 else '', formatted_p_value_pratt_corrected, (('-' if str(Z_based_Rank_Biserial_Correlation_corrected_with_ties).startswith('-') else '') + str(round(Z_based_Rank_Biserial_Correlation_corrected_with_ties,3)).lstrip('-').lstrip('0') or '0'), (confidence_level_percentages), (('-' if str(Lower_CI_Z_basesd_corrected_Pratt).startswith('-') else '') + str(round(Lower_CI_Z_basesd_corrected_Pratt,3)).lstrip('-').lstrip('0') or '0'), (('-' if str(Upper_CI_Z_basesd_corrected_Pratt).startswith('-') else '') + str(round(Upper_CI_Z_basesd_corrected_Pratt,3)).lstrip('-').lstrip('0') or '0'))
        results["Statistical Line Pratt Matched Pairs"] = " \033[3mTp\033[0m = {}, \033[3mZcorrected\033[0m = {:.3f}, {}{}, \033[3mRBmp\033[0m = {}, {}% CI(Fisher) [{}, {}]".format(((int(positive_sum_ranks_with_ties) if float(positive_sum_ranks_with_ties).is_integer() else positive_sum_ranks_with_ties)), Z_adjusted_Normal_Approxinmation_pratt, '\033[3mp = \033[0m' if p_value_adjusted_Normal_Approximation_pratt >= 0.001 else '', formatted_p_value_pratt_corrected, (('-' if str(Matched_Pairs_Rank_Biserial_Corelation_Considering_ties).startswith('-') else '') + str(round(Matched_Pairs_Rank_Biserial_Corelation_Considering_ties,3)).lstrip('-').lstrip('0') or '0'), (confidence_level_percentages), (('-' if str(Lower_CI_Matched_Pairs_Pratt).startswith('-') else '') + str(round(Lower_CI_Matched_Pairs_Pratt,3)).lstrip('-').lstrip('0') or '0'), (('-' if str(Upper_CI_Matched_Pairs_Pratt).startswith('-') else '') + str(round(Upper_CI_Matched_Pairs_Pratt,3)).lstrip('-').lstrip('0') or '0'))


        return results
    
    # Things to Consider
    # 1. Consider adding other CI's for example metsamuuronen method for sommers delta (which in the case of two groups equals the rank biserial correlation)
    # 2. Test if the matched pairs version is also equal to Sommers delta and cliffs delta (dependent version)
    # 3. For convinence change the confidence levels to percentages and not decim

    def wilcoxon_test(self, x, y):
        """
        Perform Wilcoxon signed-rank test for paired samples.
        
        Parameters
        ----------
        x : array-like
            First sample measurements
        y : array-like
            Second sample measurements
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'statistic': Wilcoxon test statistic
            - 'p_value': P-value of the test
            - 'effect_size': Effect size (r = Z/√N)
            - 'interpretation': Text interpretation of results
        """
        # ... existing code ...




