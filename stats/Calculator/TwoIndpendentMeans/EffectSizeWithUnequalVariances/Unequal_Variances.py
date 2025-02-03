

###############################################
# Effect Size for Unequal Variances ###########
###############################################

import numpy as np
import math
from scipy.stats import norm, nct, t

# Relevant Functions for Paired Samples t-test
##########################################

# 1. Pivotal CI Function
def Pivotal_ci_t(t_Score, df, sample_size, confidence_level):
    is_negative = False
    if t_Score < 0:
        is_negative = True
        t_Score = abs(t_Score)
    upper_limit = 1 - (1 - confidence_level) / 2
    lower_limit = (1 - confidence_level) / 2

    lower_criterion = [-t_Score, t_Score / 2, t_Score]
    upper_criterion = [t_Score, 2 * t_Score, 3 * t_Score]

    while nct.cdf(t_Score, df, lower_criterion[0]) < upper_limit:
        lower_criterion = [lower_criterion[0] - t_Score, lower_criterion[0], lower_criterion[2]]

    while nct.cdf(t_Score, df, upper_criterion[0]) < lower_limit:
        if nct.cdf(t_Score, df) < lower_limit:
            lower_ci = [0, nct.cdf(t_Score, df)]
            upper_criterion = [upper_criterion[0] / 4, upper_criterion[0], upper_criterion[2]]
    
    while nct.cdf(t_Score, df, upper_criterion[2]) > lower_limit:
        upper_criterion = [upper_criterion[0], upper_criterion[2], upper_criterion[2] + t_Score]

    lower_ci = 0.0
    diff_lower = 1
    while diff_lower > 0.00001:
        if nct.cdf(t_Score, df, lower_criterion[1]) < upper_limit:
            lower_criterion = [lower_criterion[0], (lower_criterion[0] + lower_criterion[1]) / 2, lower_criterion[1]]
        else:
            lower_criterion = [lower_criterion[1], (lower_criterion[1] + lower_criterion[2]) / 2, lower_criterion[2]]
        diff_lower = abs(nct.cdf(t_Score, df, lower_criterion[1]) - upper_limit)
        lower_ci = lower_criterion[1] / (np.sqrt(sample_size))
    
    upper_ci = 0.0
    diff_upper = 1
    while diff_upper > 0.00001:
        if nct.cdf(t_Score, df, upper_criterion[1]) < lower_limit:
            upper_criterion = [upper_criterion[0], (upper_criterion[0] + upper_criterion[1]) / 2, upper_criterion[1]]
        else:
            upper_criterion = [upper_criterion[1], (upper_criterion[1] + upper_criterion[2]) / 2, upper_criterion[2]]
        diff_upper = abs(nct.cdf(t_Score, df, upper_criterion[1]) - lower_limit)
        upper_ci = upper_criterion[1] / (np.sqrt(sample_size))
    if is_negative:
        return -upper_ci, -lower_ci
    else:
        return lower_ci, upper_ci
    

# 2. Central CI
def calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test(cohens_d, sample_size1, sample_size2, confidence_level): # This is a function that calculates the Confidence Intervals of the Effect size in One Sample Z_score test (or two dependent samples)
    sample_size = sample_size1+sample_size2
    df = sample_size - 2 # This is the Degrees of Freedom for one sample t-test
    correction_factor =  math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
    harmonic_sample_size = 2 / (1/sample_size1 + 1/sample_size2)
    standard_error_es = np.sqrt((df/(df-2)) * (2 / harmonic_sample_size ) * (1 + cohens_d**2 * harmonic_sample_size / 2)  - (cohens_d**2 / correction_factor**2))# This formula from Hedges, 1981, is the True formula for one sample t-test (The hedges and Olking (1985) formula is for one sampele z-test or where N is large enough)
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = cohens_d - standard_error_es * z_critical_value,  cohens_d + standard_error_es * z_critical_value
    return ci_lower, ci_upper, standard_error_es

class Indpendnent_samples_unequal_variance():       
    @staticmethod
    def Unequal_Variances_from_parameters(params: dict) -> dict:

        # Set params
        sample_mean_1 = params["Mean 1"]
        sample_mean_2 = params["Mean 2"]
        sample_sd_1 = params["Standard Deviation 1"]
        sample_sd_2 = params["Standard Deviation 2"]
        sample_size_1 = params["Sample Size 1"]
        sample_size_2 = params["Sample Size 2"]
        population_mean_diff = params["Difference in the Population"] # The default value should be 0
        confidence_level_percentages = params["Confidence Level"]

        # Calculation 
        confidence_level = confidence_level_percentages / 100
        sample_size = sample_size_1 + sample_size_2
        mean_difference = sample_mean_1 - sample_mean_2
        variance_1 = sample_sd_1**2
        variance_2 = sample_sd_2**2        
        Standard_Error_Welch_T = np.sqrt(variance_1/sample_size_1 + (variance_2/sample_size_2))
        Welchs_t =  (sample_mean_1 - sample_mean_2 - population_mean_diff) / Standard_Error_Welch_T
        df = (variance_1 / sample_size_1 + variance_2 / sample_size_2)**2 / ((variance_1 / sample_size_1)**2 / (sample_size_1 - 1) + (variance_2 / sample_size_2)**2 / (sample_size_2 - 1))
        p_value = min(float(t.sf((abs(Welchs_t)), df) * 2), 0.99999)      

        #Effect sizes based on the Welch's t
        harmonic_n = sample_size_1*sample_size_2/(sample_size_1 + sample_size_2) #This is the harmonic mean of n divided by 2 
        correction =  math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
        epsilon = Welchs_t / np.sqrt(harmonic_n)
        epsilon_unbiased = epsilon * correction     
        standard_error_epsilon_biased = df/(df-2)*(1/harmonic_n + epsilon_unbiased**2)-epsilon_unbiased**2/correction**2
        standard_error_epsilon_unbiased = standard_error_epsilon_biased * correction**2        
        standardizer_biased =  np.sqrt((variance_1 *sample_size_2 + variance_2 *sample_size_1) / (sample_size_1 + sample_size_2))
        standardizer_Unbiased = standardizer_biased / correction

        # Confidence_Intervals
        lower_ncp, upper_ncp = Pivotal_ci_t(Welchs_t, df, sample_size, confidence_level)

        epsilon_biased_lower_ci = (lower_ncp*(np.sqrt(sample_size))) / np.sqrt(harmonic_n)
        epsilon_biased_upper_ci = (upper_ncp*(np.sqrt(sample_size))) / np.sqrt(harmonic_n)
        epsilon_unbiased_lower_ci = (lower_ncp*(np.sqrt(sample_size))) / np.sqrt(harmonic_n) * correction
        epsilon_unbiased_upper_ci = (upper_ncp*(np.sqrt(sample_size))) / np.sqrt(harmonic_n) * correction


        results = {}

        results["Welch's t-score"] = round(Welchs_t, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["P-Value"] = round(p_value, 4)
        results["Mean Difference"] = round(mean_difference, 4)
        
        results["Aoki's Epsilon"] = round(epsilon, 4)
        results["Aoki's Epsilon Unbiased"] = round(epsilon_unbiased, 4)
        results["Aoki's Epsilon Standard Error"] = round(standard_error_epsilon_biased, 4)
        results["Aoki's Epsilon Unbiased Standard Error"] = round(standard_error_epsilon_unbiased, 4)
        results["Aoki's Epsilon Standardizer"] = round(standardizer_biased, 4)
        results["Aoki's Epsilon Unbiased Standardizer"] = round(standardizer_Unbiased, 4)
        results["Lower Confidence Interval Aoki's Epsilon"] = round(epsilon_biased_lower_ci, 4)
        results["Upper Confidence Interval Aoki's Epsilon"] = round(epsilon_biased_upper_ci, 4)
        results["Lower Confidence Interval Aoki's Epsilon Unbiased"] = round(epsilon_unbiased_lower_ci, 4)
        results["Upper Confidence Interval Aoki's Epsilon Unbiased"] = round(epsilon_unbiased_upper_ci, 4)
        results["Correction Factor"] = round(correction, 4)
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Epsilon"] ="Welch's \033[3mt\033[0m({}) = {:.3f}, {}{}, Aoki's ε = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(((int(round(df,3)) if float(df).is_integer() else round(df,3))), Welchs_t, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, epsilon, confidence_level_percentages, epsilon_biased_lower_ci, epsilon_biased_upper_ci)
        results["Statistical Line Epsilon Unbiased"] = "Welch's \033[3mt\033[0m({}) = {:.3f}, {}{}, Aoki's ε = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(((int(df) if float(round(df,3)).is_integer() else round(df,3))), Welchs_t, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, epsilon_unbiased, confidence_level_percentages, epsilon_biased_lower_ci*correction, epsilon_biased_upper_ci*correction)

        return results
    
    @staticmethod
    def Unequal_Variances_from_data(params: dict) -> dict:

        # Get params
        column_1 = params["column_1"]
        column_2 = params["column_2"]
        population_mean_diff = params["Difference in the Population"] # The default value should be 0
        confidence_level_percentages = params["Confidence Level"]

        # Calculation 
        confidence_level = confidence_level_percentages / 100
        sample_mean_1 = np.mean(column_1)
        sample_mean_2 = np.mean(column_2)
        sample_sd_1 = np.std(column_1, ddof = 1)
        sample_sd_2 = np.std(column_2, ddof = 1)
        sample_size = len (column_1)
        sample_size_1 = len(column_1)
        sample_size_2 = len(column_2)

        sample_size = sample_size_1 + sample_size_2
        mean_difference = sample_mean_1 - sample_mean_2
        variance_1 = sample_sd_1**2
        variance_2 = sample_sd_2**2
        
        #Welch's t statistics
        Standard_Error_Welch_T = np.sqrt(variance_1/sample_size_1 + (variance_2/sample_size_2))
        Welchs_t =  (sample_mean_1 - sample_mean_2 - population_mean_diff) / Standard_Error_Welch_T
        df = (variance_1 / sample_size_1 + variance_2 / sample_size_2)**2 / ((variance_1 / sample_size_1)**2 / (sample_size_1 - 1) + (variance_2 / sample_size_2)**2 / (sample_size_2 - 1))
        p_value = min(float(t.sf((abs(Welchs_t)), df) * 2), 0.99999)      

        #Effect sizes based on the Welch's t
        harmonic_n = sample_size_1*sample_size_2/(sample_size_1 + sample_size_2) #This is the harmonic mean of n divided by 2 
        correction =  math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))

        epsilon = Welchs_t / np.sqrt(harmonic_n)
        epsilon_unbiased = epsilon * correction
        
        standard_error_epsilon_biased = df/(df-2)*(1/harmonic_n + epsilon_unbiased**2)-epsilon_unbiased**2/correction**2
        standard_error_epsilon_unbiased = standard_error_epsilon_biased * correction**2
        
        standardizer_biased =  np.sqrt((variance_1 *sample_size_2 + variance_2 *sample_size_1) / (sample_size_1 + sample_size_2))
        standardizer_Unbiased = standardizer_biased / correction

        # Confidence_Intervals
        lower_ncp, upper_ncp = Pivotal_ci_t(Welchs_t, df, sample_size, confidence_level)

        epsilon_biased_lower_ci = (lower_ncp*(np.sqrt(sample_size))) / np.sqrt(harmonic_n)
        epsilon_biased_upper_ci = (upper_ncp*(np.sqrt(sample_size))) / np.sqrt(harmonic_n)
        epsilon_unbiased_lower_ci = (lower_ncp*(np.sqrt(sample_size))) / np.sqrt(harmonic_n) * correction
        epsilon_unbiased_upper_ci = (upper_ncp*(np.sqrt(sample_size))) / np.sqrt(harmonic_n) * correction


        results = {}

        results["Welch's t-score"] = round(Welchs_t, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["P-Value"] = round(p_value, 4)
        results["Mean Difference"] = round(mean_difference, 4)
        
        results["Aoki's Epsilon"] = round(epsilon, 4)
        results["Aoki's Epsilon Unbiased"] = round(epsilon_unbiased, 4)
        results["Aoki's Epsilon Standard Error"] = round(standard_error_epsilon_biased, 4)
        results["Aoki's Epsilon Unbiased Standard Error"] = round(standard_error_epsilon_unbiased, 4)
        results["Aoki's Epsilon Standardizer"] = round(standardizer_biased, 4)
        results["Aoki's Epsilon Unbiased Standardizer"] = round(standardizer_Unbiased, 4)
        results["Lower Confidence Interval Aoki's Epsilon"] = round(epsilon_biased_lower_ci, 4)
        results["Upper Confidence Interval Aoki's Epsilon"] = round(epsilon_biased_upper_ci, 4)
        results["Lower Confidence Interval Aoki's Epsilon Unbiased"] = round(epsilon_unbiased_lower_ci, 4)
        results["Upper Confidence Interval Aoki's Epsilon Unbiased"] = round(epsilon_unbiased_upper_ci, 4)
        results["Correction Factor"] = round(correction, 4)
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Epsilon"] ="Welch's \033[3mt\033[0m({}) = {:.3f}, {}{}, Aoki's ε = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(((int(round(df,3)) if float(df).is_integer() else round(df,3))), Welchs_t, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, epsilon, confidence_level_percentages, epsilon_biased_lower_ci, epsilon_biased_upper_ci)
        results["Statistical Line Epsilon Unbiased"] = "Welch's \033[3mt\033[0m({}) = {:.3f}, {}{}, Aoki's ε = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(((int(df) if float(round(df,3)).is_integer() else round(df,3))), Welchs_t, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, epsilon_unbiased, confidence_level_percentages, epsilon_biased_lower_ci*correction, epsilon_biased_upper_ci*correction)

        return results

    # Things to Consider
    # 1. Consider Adding Another Central CI based on other measures of SD's just for debugging and for comapring to other sources
    # 2. Consider Adding the Non-Central CI from Cousineau 2020 preprint
    # Consider Adding Sheih Effect Size
 




