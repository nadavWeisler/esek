
######################################################################
# Effect Size for two indpednent samples With A Control Group ########
######################################################################

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
def calculate_central_ci_from_cohens_d_one_sample_t_test(cohens_d, sample_size, confidence_level): # This is a function that calculates the Confidence Intervals of the Effect size in One Sample Z_score test (or two dependent samples)
    df = sample_size - 1 # This is the Degrees of Freedom for one sample t-test
    correction_factor =  math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
    standard_error_es = np.sqrt((df/(df-2)) * (1 / sample_size ) * (1 + cohens_d**2 * sample_size)  - (cohens_d**2 / correction_factor**2))# This formula from Hedges, 1981, is the True formula for one sample t-test (The hedges and Olking (1985) formula is for one sampele z-test or where N is large enough)
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = cohens_d - standard_error_es * z_critical_value,  cohens_d + standard_error_es * z_critical_value
    return ci_lower, ci_upper, standard_error_es

class Indpendent_Samples_Control_Group():
    @staticmethod
    def Two_indpendent_samples_with_Control_Group_from_parameters(params: dict) -> dict:
        
        # Set params
        sample_mean_Experimental = params["Sample Mean Experimental Group"]
        sample_mean_Control = params["Sample Mean Control Group"]
        sample_sd_Experimental = params["Standard Deviation Experimental Group"]
        sample_sd_Control = params["Standard Deviation Control Group"]
        sample_size_Experimental = params["Sample Size Experimental Group"]
        sample_size_Control = params["Sample Size Control Group"]
        population_mean_diff = params["Difference in the Population"] # The default value should be 0
        confidence_level_percentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentages / 100
        sample_size = sample_size_Experimental + sample_size_Control
        df = sample_size - 2
        sample_mean_difference = sample_mean_Experimental - sample_mean_Control
        standardizer_Glass_Delta = np.sqrt(((((sample_size_Experimental-1)*sample_sd_Experimental**2)) + ((sample_size_Control-1)*sample_sd_Control**2)) / (sample_size-2))
        standard_error = standardizer_Glass_Delta * np.sqrt((sample_size_Experimental+sample_size_Control)/(sample_size_Experimental*sample_size_Control))
        t_score = ((sample_mean_difference - population_mean_diff) / standard_error)
        t_score_glass = ((sample_mean_difference - population_mean_diff) / (sample_sd_Control/np.sqrt(sample_size_Control)))
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)      
        glass_delta = ((sample_mean_difference - population_mean_diff) / sample_sd_Control)
        df2 = sample_size_Control - 1
        correction = math.exp(math.lgamma(df2/2) - math.log(math.sqrt(df2/2)) - math.lgamma((df2-1)/2))
        Unbiased_Glass_Delta = glass_delta * correction
        standardizer_unbiased_Glass_Delta = standardizer_Glass_Delta / correction
        ci_lower_Glass_Delta_Pivotal, ci_upper_Glass_Delta_Pivotal =  Pivotal_ci_t (t_score_glass, df2, sample_size_Control, confidence_level)
        ci_lower_Glass_Delta_central, ci_upper_Glass_Delta_central, standard_error_Glass_Delta =  calculate_central_ci_from_cohens_d_one_sample_t_test(glass_delta, sample_size_Control, confidence_level)
        ci_lower_Glass_Delta_Unbiased_central, ci_upper_Glass_Delta_Unbiased_central, standard_error_Glass_Delta_Unbiased =  calculate_central_ci_from_cohens_d_one_sample_t_test(Unbiased_Glass_Delta, sample_size_Control, confidence_level)
 
        # Set results
        results = {}
        results["Glass' Delta"] = round(glass_delta, 4) # Known as Cohens dp
        results["Unbiased Glass'; Delta"] = round(Unbiased_Glass_Delta, 4) 
        results["Standard Error of Glass' Delta"] = round(standard_error_Glass_Delta, 4)
        results["Standard Error of Glass' Delta Unbiased"] = round(standard_error_Glass_Delta_Unbiased, 4)
        results["Standardizer Glass' Delta"] = round(standardizer_Glass_Delta, 4)
        results["Standardizer Glass' Delta Unbiased"] = round(standardizer_unbiased_Glass_Delta, 4)
        results["t-score"] = round(t_score, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["p-value"] = round(p_value, 4)
        results["Standard Error of the Mean Difference"] = round(standard_error, 4)
        results["Difference Between Samples"] = round(sample_mean_difference, 4)
        results["Difference in the Population"] = round(population_mean_diff, 4)
        
        results["Lower Pivotal CI's Glass Delta"] = round(ci_lower_Glass_Delta_Pivotal, 4)
        results["Upper Pivotal CI's Glass' Delta"] = round(ci_upper_Glass_Delta_Pivotal, 4)
        results["Lower Pivotal CI's Glass' Delta Unbiased"] = round(ci_lower_Glass_Delta_Pivotal*correction, 4)
        results["Upper Pivotal CI's Glass' Delta Unbiased"] = round(ci_upper_Glass_Delta_Pivotal*correction, 4)

        results["Lower Central CI's Glass Delta"] = round(ci_lower_Glass_Delta_central, 4)
        results["Upper Central CI's Glass Delta"] = round(ci_upper_Glass_Delta_central, 4)
        results["Lower Central CI's Glass' Delta Unbiased"] = round(ci_lower_Glass_Delta_Unbiased_central, 4)
        results["Upper Central CI's Glass' Delta Unbiased"] = round(ci_upper_Glass_Delta_Unbiased_central, 4)
        results["Correction Factor"] = round(correction, 4)

        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Glass' Delta"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Glass' Δ = {:.3f},  {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, Unbiased_Glass_Delta, confidence_level_percentages, ci_lower_Glass_Delta_Pivotal, ci_upper_Glass_Delta_Pivotal)
        results["Statistical Line Uniased Glass' Delta"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Unbiased Glass' Δ = {:.3f},  {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, glass_delta, confidence_level_percentages, ci_lower_Glass_Delta_Pivotal*correction, ci_upper_Glass_Delta_Pivotal*correction)

        return results
        
    @staticmethod
    def Two_indpendent_samples_with_Control_Group_from_data(params: dict) -> dict:
        
        # Set params
        column_1 = params["column_1"] # This is the coloumn of the Control Group
        column_2 = params["column_2"] # This is the coloumn of the Experimental Group
        population_mean_diff = params["Difference in the Population"] # The default value should be 0
        confidence_level_percentages = params["Confidence Level"]
        
        # Calculation
        confidence_level = confidence_level_percentages / 100

        sample_mean_Control = np.mean(column_1)
        sample_mean_Experimental = np.mean(column_2)
        sample_sd_Control = np.std(column_1, ddof = 1)
        sample_sd_Experimental = np.std(column_2, ddof = 1)
        sample_size_Control = len(column_1)
        sample_size_Experimental = len(column_2)
        sample_size = sample_size_Control + sample_size_Experimental
        df = sample_size - 2
        sample_mean_difference = sample_mean_Experimental - sample_mean_Control
        # Calculation
        sample_size = sample_size_Experimental + sample_size_Control
        df = sample_size - 2
        sample_mean_difference = sample_mean_Experimental - sample_mean_Control
        standardizer_Glass_Delta = np.sqrt(((((sample_size_Experimental-1)*sample_sd_Experimental**2)) + ((sample_size_Control-1)*sample_sd_Control**2)) / (sample_size-2))
        standard_error = standardizer_Glass_Delta * np.sqrt((sample_size_Experimental+sample_size_Control)/(sample_size_Experimental*sample_size_Control))
        t_score = ((sample_mean_difference - population_mean_diff) / standard_error)
        t_score_glass = ((sample_mean_difference - population_mean_diff) / (sample_sd_Control/np.sqrt(sample_size_Control)))
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)      
        glass_delta = ((sample_mean_difference - population_mean_diff) / sample_sd_Control)
        df2 = sample_size_Control - 1
        correction = math.exp(math.lgamma(df2/2) - math.log(math.sqrt(df2/2)) - math.lgamma((df2-1)/2))
        Unbiased_Glass_Delta = glass_delta * correction
        standardizer_unbiased_Glass_Delta = standardizer_Glass_Delta / correction
        ci_lower_Glass_Delta_Pivotal, ci_upper_Glass_Delta_Pivotal =  Pivotal_ci_t (t_score_glass, df2, sample_size_Control, confidence_level)
        ci_lower_Glass_Delta_central, ci_upper_Glass_Delta_central, standard_error_Glass_Delta =  calculate_central_ci_from_cohens_d_one_sample_t_test(glass_delta, sample_size_Control, confidence_level)
        ci_lower_Glass_Delta_Unbiased_central, ci_upper_Glass_Delta_Unbiased_central, standard_error_Glass_Delta_Unbiased =  calculate_central_ci_from_cohens_d_one_sample_t_test(Unbiased_Glass_Delta, sample_size_Control, confidence_level)
 
        # Set results
        results = {}
        results["Glass' Delta"] = round(glass_delta, 4) # Known as Cohens dp
        results["Unbiased Glass'; Delta"] = round(Unbiased_Glass_Delta, 4) 
        results["Standard Error of Glass' Delta"] = round(standard_error_Glass_Delta, 4)
        results["Standard Error of Glass' Delta Unbiased"] = round(standard_error_Glass_Delta_Unbiased, 4)
        results["Standardizer Glass' Delta"] = round(standardizer_Glass_Delta, 4)
        results["Standardizer Glass' Delta Unbiased"] = round(standardizer_unbiased_Glass_Delta, 4)
        results["t-score"] = round(t_score, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["p-value"] = round(p_value, 4)
        results["Standard Error of the Mean Difference"] = round(standard_error, 4)
        results["Difference Between Samples"] = round(sample_mean_difference, 4)
        results["Difference in the Population"] = round(population_mean_diff, 4)
        
        results["Lower Pivotal CI's Glass Delta"] = round(ci_lower_Glass_Delta_Pivotal, 4)
        results["Upper Pivotal CI's Glass' Delta"] = round(ci_upper_Glass_Delta_Pivotal, 4)
        results["Lower Pivotal CI's Glass' Delta Unbiased"] = round(ci_lower_Glass_Delta_Pivotal*correction, 4)
        results["Upper Pivotal CI's Glass' Delta Unbiased"] = round(ci_upper_Glass_Delta_Pivotal*correction, 4)

        results["Lower Central CI's Glass Delta"] = round(ci_lower_Glass_Delta_central, 4)
        results["Upper Central CI's Glass Delta"] = round(ci_upper_Glass_Delta_central, 4)
        results["Lower Central CI's Glass' Delta Unbiased"] = round(ci_lower_Glass_Delta_Unbiased_central, 4)
        results["Upper Central CI's Glass' Delta Unbiased"] = round(ci_upper_Glass_Delta_Unbiased_central, 4)
        results["Correction Factor"] = round(correction, 4)

        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Glass' Delta"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Glass' Δ = {:.3f},  {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, Unbiased_Glass_Delta, confidence_level_percentages, ci_lower_Glass_Delta_Pivotal, ci_upper_Glass_Delta_Pivotal)
        results["Statistical Line Uniased Glass' Delta"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Unbiased Glass' Δ = {:.3f},  {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, glass_delta, confidence_level_percentages, ci_lower_Glass_Delta_Pivotal*correction, ci_upper_Glass_Delta_Pivotal*correction)

        return results

   
    # Things to Consider
    # 1. Consider Adding Another Central CI based on other measures of SD's just for debugging and for comapring to other sources
    # 2. Consider Adding the Non-Central CI from Cousineau 2020 preprint
    # 3. Consider Changing the name of dpop to dmle (Laken calls it Cohens dpop)
    # 4. Control Group is only for indpendent samples, consider sing the depdennt measure version 
 


