
###############################################
# Effect Size for Indpendent Samples t-Test ###
###############################################

import numpy as np
import math
from scipy.stats import norm, nct, t
import rpy2.robjects as robjects
#robjects.r('install.packages("sadists")')
robjects.r('library(sadists)')
qlambdap = robjects.r['qlambdap']

# Relevant Functions for Paired Samples t-test
##########################################

# 1. Pivotal CI Function
def Pivotal_ci_t(t_Score, df, sample_size_1, sample_size_2, confidence_level):
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
        lower_ci = lower_criterion[1] * np.sqrt((sample_size_1 + sample_size_2)/(sample_size_1 * sample_size_2))
    
    upper_ci = 0.0
    diff_upper = 1
    while diff_upper > 0.00001:
        if nct.cdf(t_Score, df, upper_criterion[1]) < lower_limit:
            upper_criterion = [upper_criterion[0], (upper_criterion[0] + upper_criterion[1]) / 2, upper_criterion[1]]
        else:
            upper_criterion = [upper_criterion[1], (upper_criterion[1] + upper_criterion[2]) / 2, upper_criterion[2]]
        diff_upper = abs(nct.cdf(t_Score, df, upper_criterion[1]) - lower_limit)
        upper_ci = upper_criterion[1] * np.sqrt((sample_size_1 + sample_size_2)/(sample_size_1 * sample_size_2))
    if is_negative:
        return -upper_ci, -lower_ci
    else:
        return lower_ci, upper_ci
    

# 2. Central CI
def calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test(effect_size, sample_size1, sample_size2, confidence_level): # This is a function that calculates the Confidence Intervals of the Effect size in One Sample Z_score test (or two dependent samples)
    sample_size = sample_size1+sample_size2
    df = sample_size - 2 # This is the Degrees of Freedom for one sample t-test
    correction_factor =  math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
    harmonic_sample_size = 2 / (1/sample_size1 + 1/sample_size2)
    A = harmonic_sample_size / 2
    Standard_error_effect_size_True = np.sqrt(((df/(df-2)) * (1 / A ) * (1 + effect_size**2 * A)  - (effect_size**2 / correction_factor**2)))
    Standard_error_effect_size_Morris = np.sqrt((df/(df-2)) * (1 / A ) * (1 + effect_size**2 * A)  - (effect_size**2 / (1 - (3/(4*(df-1)-1)))**2))
    Standard_error_effect_size_Hedges =  np.sqrt((1/A) + effect_size**2/ (2*df)) 
    Standard_error_effect_size_Hedges_Olkin = np.sqrt((1/A) + effect_size**2/ (2*sample_size))
    Standard_error_effect_size_MLE = np.sqrt(Standard_error_effect_size_Hedges * ((df+2)/df))
    Standard_error_effect_size_Large_N = np.sqrt(1/ A * (1 + effect_size**2/8))
    Standard_error_effect_size_Small_N = np.sqrt(Standard_error_effect_size_Large_N * ((df+1)/(df-1)))
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = effect_size - Standard_error_effect_size_True * z_critical_value,  effect_size + Standard_error_effect_size_True * z_critical_value
    return ci_lower, ci_upper, Standard_error_effect_size_True, Standard_error_effect_size_Morris, Standard_error_effect_size_Hedges, Standard_error_effect_size_Hedges_Olkin, Standard_error_effect_size_MLE, Standard_error_effect_size_Large_N, Standard_error_effect_size_Small_N

class Indpendnent_samples_ttest():
    @staticmethod
    def two_independent_from_t_score (params: dict) -> dict:
        
        # Set params
        t_score = params["t Score"]
        sample_size_1 = params["Sample Size 1"]
        sample_size_2 = params["Sample Size 2"]
        confidence_level_percentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentages / 100
        sample_size = sample_size_1 + sample_size_2
        df = sample_size - 2
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)      
        cohens_ds = t_score * (np.sqrt(1/sample_size_1 + 1/sample_size_2))
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
        hedges_gs = cohens_ds * correction
        cohens_dpop = cohens_ds / np.sqrt((df/sample_size))
        t_score_dpop = cohens_dpop / np.sqrt((1/sample_size_1 + 1/sample_size_2))
        ci_lower_cohens_dpop_Pivotal, ci_upper_cohens_dpop_Pivotal =  Pivotal_ci_t (t_score_dpop, df, sample_size_1, sample_size_2, confidence_level)
        ci_lower_cohens_ds_Pivotal, ci_upper_cohens_ds_Pivotal =  Pivotal_ci_t (t_score, df, sample_size_1, sample_size_2, confidence_level)
        ci_lower_cohens_ds_central, ci_upper_cohens_ds_central, standard_error_cohens_ds_true, standard_error_cohens_ds_morris, standard_error_cohens_ds_hedges, standard_error_cohens_ds_hedges_olkin, standard_error_cohens_ds_MLE, standard_error_cohens_ds_Largen, standard_error_cohens_ds_Small_n =  calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test(cohens_ds, sample_size_1, sample_size_2, confidence_level)
        ci_lower_hedges_gs_central, ci_upper_hedges_gs_central,  standard_error_hedges_gs_true, standard_error_hedges_gs_morris, standard_error_hedges_gs_hedges, standard_error_hedges_gs_hedges_olkin, standard_error_hedges_gs_MLE, standard_error_hedges_gs_Largen, standard_error_hedges_gs_Small_n =  calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test(hedges_gs, sample_size_1, sample_size_2, confidence_level)
        ci_lower_cohens_dpop_central, ci_upper_cohens_dpop_central,  standard_error_cohens_dpop_true, standard_error_cohens_dpop_morris, standard_error_cohens_dpop_hedges, standard_error_cohens_dpop_hedges_olkin, standard_error_cohens_dpop_MLE, standard_error_cohens_dpop_Largen, standard_error_cohens_dpop_Small_n =  calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test(cohens_dpop, sample_size_1, sample_size_2, confidence_level)
            
        # Set results
        results = {}
        results["Cohen's ds"] = round(cohens_ds, 4) # Known as Cohens dp
        results["Hedges' gs"] = round(hedges_gs, 4) 
        results["Cohen's dpop"] = round(cohens_dpop, 4) # Also known as the MLE

        results["t-score"] = round(t_score, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["p-value"] = round(p_value, 4)

        # All type of Standard Errors
        results["Standard Error of Cohen's ds (True)"] = round(standard_error_cohens_ds_true, 4)
        results["Standard Error of Cohen's ds (Morris)"] = round(standard_error_cohens_ds_morris, 4)
        results["Standard Error of Cohen's ds (Hedges)"] = round(standard_error_cohens_ds_hedges, 4)
        results["Standard Error of Cohen's ds (Hedges_Olkin)"] = round(standard_error_cohens_ds_hedges_olkin, 4)
        results["Standard Error of Cohen's ds (MLE)"] = round(standard_error_cohens_ds_MLE, 4)
        results["Standard Error of Cohen's ds (Large N)"] = round(standard_error_cohens_ds_Largen, 4)
        results["Standard Error of Cohen's ds (Small N)"] = round(standard_error_cohens_ds_Small_n, 4)
        results["Standard Error of Hedges' gs (True)"] = round(standard_error_hedges_gs_true, 4)
        results["Standard Error of Hedges' gs (Morris)"] = round(standard_error_hedges_gs_morris, 4)
        results["Standard Error of Hedges' gs (Hedges)"] = round(standard_error_hedges_gs_hedges, 4)
        results["Standard Error of Hedges' gs (Hedges_Olkin)"] = round(standard_error_hedges_gs_hedges_olkin, 4)
        results["Standard Error of Hedges' gs (MLE)"] = round(standard_error_hedges_gs_MLE, 4)
        results["Standard Error of Hedges' gs (Large N)"] = round(standard_error_hedges_gs_Largen, 4)
        results["Standard Error of Hedges' gs (Small N)"] = round(standard_error_hedges_gs_Small_n, 4)
        results["Standard Error of Cohen's dpop (True)"] = round(standard_error_cohens_dpop_true, 4)
        results["Standard Error of Cohen's dpop (Morris)"] = round(standard_error_cohens_dpop_morris, 4)
        results["Standard Error of Cohen's dpop (Hedges)"] = round(standard_error_cohens_dpop_hedges, 4)
        results["Standard Error of Cohen's dpop (Hedges_Olkin)"] = round(standard_error_cohens_dpop_hedges_olkin, 4)
        results["Standard Error of Cohen's dpop (MLE)"] = round(standard_error_cohens_dpop_MLE, 4)
        results["Standard Error of Cohen's dpop (Large N)"] = round(standard_error_cohens_dpop_Largen, 4)
        results["Standard Error of Cohen's dpop (Small N)"] = round(standard_error_cohens_dpop_Small_n, 4)

        results["Lower Pivotal CI's Cohen's ds"] = round(ci_lower_cohens_ds_Pivotal, 4)
        results["Upper Pivotal CI's Cohen's ds"] = round(ci_upper_cohens_ds_Pivotal, 4)
        results["Lower Pivotal CI's Hedges' gs"] = round(ci_lower_cohens_ds_Pivotal*correction, 4)
        results["Upper Pivotal CI's Hedges' gs"] = round(ci_upper_cohens_ds_Pivotal*correction, 4)
        results["Lower Pivotal CI's Cohen's dpop"] = round(ci_lower_cohens_dpop_Pivotal,4)
        results["Upper Pivotal CI's Cohen's dpop"] = round(ci_upper_cohens_dpop_Pivotal,4)
        results["Lower Central CI's Cohen's ds"] = round(ci_lower_cohens_ds_central, 4)
        results["Upper Central CI's Cohen's ds"] = round(ci_upper_cohens_ds_central, 4)
        results["Lower Central CI's Hedges' gs"] = round(ci_lower_hedges_gs_central, 4)
        results["Upper Central CI's Hedges' gs"] = round(ci_upper_hedges_gs_central, 4)
        results["Lower Central CI's Cohen's dpop"] = round(((ci_lower_cohens_dpop_central)), 4)
        results["Upper Central CI's Cohen's dpop"] = round(((ci_upper_cohens_dpop_central)),4)
        results["Correction Factor"] = round(correction, 4)
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Cohen's ds"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's d\u209B = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cohens_ds, confidence_level_percentages, ci_lower_cohens_ds_Pivotal, ci_upper_cohens_ds_Pivotal)
        results["Statistical Line Hedges' gs"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Hedges' g\u209B = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, hedges_gs, confidence_level_percentages, ci_lower_cohens_ds_Pivotal*correction, ci_upper_cohens_ds_Pivotal*correction)
        results["Statistical Line Cohens' dpop"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's dpop = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cohens_dpop, confidence_level_percentages, ci_lower_cohens_ds_Pivotal/ (np.sqrt((df/sample_size))), ci_upper_cohens_ds_Pivotal/ (np.sqrt((df/sample_size))))

        return results
        
    @staticmethod
    def two_independent_from_parameters(params: dict) -> dict:
        
        # Set params
        sample_mean_1 = params["Sample Mean 1"]
        sample_mean_2 = params["Sample Mean 2"]
        sample_sd_1 = params["Standard Deviation Sample 1"]
        sample_sd_2 = params["Standard Deviation Sample 2"]
        sample_size_1 = params["Sample Size 1"]
        sample_size_2 = params["Sample Size 2"]
        population_mean_diff = params["Difference in the Population"] # The default value should be 0
        confidence_level_percentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentages / 100
        sample_size = sample_size_1 + sample_size_2
        df = sample_size - 2
        sample_mean_difference = sample_mean_1 - sample_mean_2
        standardizer_ds = np.sqrt(((((sample_size_1-1)*sample_sd_1**2)) + ((sample_size_2-1)*sample_sd_2**2)) / (sample_size-2))
        standardizer_dpop = np.sqrt(((((sample_size_1-1)*sample_sd_1**2)) + ((sample_size_2-1)*sample_sd_2**2)) / (sample_size))
        standard_error = standardizer_ds * np.sqrt((sample_size_1+sample_size_2)/(sample_size_1*sample_size_2))
        t_score = ((sample_mean_difference - population_mean_diff) / standard_error)
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)      
        cohens_ds = t_score * (np.sqrt(1/sample_size_1 + 1/sample_size_2))
        cohens_dpop = cohens_ds / np.sqrt((df/sample_size))
        t_score_dpop = cohens_dpop / np.sqrt((1/sample_size_1 + 1/sample_size_2))
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
        hedges_gs = cohens_ds * correction
        standardizer_hedges_gs = standardizer_ds / correction
        ci_lower_cohens_ds_Pivotal, ci_upper_cohens_ds_Pivotal =  Pivotal_ci_t (t_score, df, sample_size_1, sample_size_2, confidence_level)
        ci_lower_cohens_dpop_Pivotal, ci_upper_cohens_dpop_Pivotal =  Pivotal_ci_t (t_score_dpop, df, sample_size_1, sample_size_2, confidence_level)        
        ci_lower_cohens_ds_central, ci_upper_cohens_ds_central, standard_error_cohens_ds_true, standard_error_cohens_ds_morris, standard_error_cohens_ds_hedges, standard_error_cohens_ds_hedges_olkin, standard_error_cohens_ds_MLE, standard_error_cohens_ds_Largen, standard_error_cohens_ds_Small_n =  calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test(cohens_ds, sample_size_1, sample_size_2, confidence_level)
        ci_lower_hedges_gs_central, ci_upper_hedges_gs_central,  standard_error_hedges_gs_true, standard_error_hedges_gs_morris, standard_error_hedges_gs_hedges, standard_error_hedges_gs_hedges_olkin, standard_error_hedges_gs_MLE, standard_error_hedges_gs_Largen, standard_error_hedges_gs_Small_n =  calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test(hedges_gs, sample_size_1, sample_size_2, confidence_level)
        ci_lower_cohens_dpop_central, ci_upper_cohens_dpop_central,  standard_error_cohens_dpop_true, standard_error_cohens_dpop_morris, standard_error_cohens_dpop_hedges, standard_error_cohens_dpop_hedges_olkin, standard_error_cohens_dpop_MLE, standard_error_cohens_dpop_Largen, standard_error_cohens_dpop_Small_n =  calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test(cohens_dpop, sample_size_1, sample_size_2, confidence_level)

        # Ratio of Means
        ratio_of_means = sample_mean_1 / sample_mean_2
        Varaince_of_means_ratio = sample_sd_1**2/(sample_size_1*sample_mean_1**2) + sample_sd_2**2/(sample_size_2*sample_mean_2**2)
        Standard_Error_of_means_ratio = np.sqrt(Varaince_of_means_ratio)
        Degrees_of_freedom_means_ratio =  Varaince_of_means_ratio**2/(sample_sd_1**4/(sample_mean_1**4*(sample_size_1**3 - sample_size_1**2)) + sample_sd_2**4/(sample_mean_2**4*(sample_size_2**3 - sample_size_2**2))) 
        t_critical_value = t.ppf(confidence_level + ((1 - confidence_level) / 2), Degrees_of_freedom_means_ratio)
        Lower_CI_Means_Ratio = math.exp(np.log(ratio_of_means) - t_critical_value * np.sqrt(Varaince_of_means_ratio))
        Upper_CI_Means_Ratio = math.exp(np.log(ratio_of_means) + t_critical_value * np.sqrt(Varaince_of_means_ratio))


        # Set results
        results = {}
        results["Cohen's ds"] = round(cohens_ds, 4) # Known as Cohens dp
        results["Hedges' gs"] = round(hedges_gs, 4) 
        results["Cohen's dpop"] = round(cohens_dpop, 10) # Also known as the MLE
        results["Standardizer Cohen's ds"] = round(standardizer_ds, 4)
        results["Standardizer Hedges' gs"] = round(standardizer_hedges_gs, 4)
        results["Standardizer Cohen's dpop"] = round(standardizer_dpop, 4)
        results["t-score"] = round(t_score, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["p-value"] = round(p_value, 4)
        
        # All type of Effect Size Standard Errors
        results["Standard Error of Cohen's ds (True)"] = round(standard_error_cohens_ds_true, 4)
        results["Standard Error of Cohen's ds (Morris)"] = round(standard_error_cohens_ds_morris, 4)
        results["Standard Error of Cohen's ds (Hedges)"] = round(standard_error_cohens_ds_hedges, 4)
        results["Standard Error of Cohen's ds (Hedges_Olkin)"] = round(standard_error_cohens_ds_hedges_olkin, 4)
        results["Standard Error of Cohen's ds (MLE)"] = round(standard_error_cohens_ds_MLE, 4)
        results["Standard Error of Cohen's ds (Large N)"] = round(standard_error_cohens_ds_Largen, 4)
        results["Standard Error of Cohen's ds (Small N)"] = round(standard_error_cohens_ds_Small_n, 4)
        results["Standard Error of Hedges' gs (True)"] = round(standard_error_hedges_gs_true, 4)
        results["Standard Error of Hedges' gs (Morris)"] = round(standard_error_hedges_gs_morris, 4)
        results["Standard Error of Hedges' gs (Hedges)"] = round(standard_error_hedges_gs_hedges, 4)
        results["Standard Error of Hedges' gs (Hedges_Olkin)"] = round(standard_error_hedges_gs_hedges_olkin, 4)
        results["Standard Error of Hedges' gs (MLE)"] = round(standard_error_hedges_gs_MLE, 4)
        results["Standard Error of Hedges' gs (Large N)"] = round(standard_error_hedges_gs_Largen, 4)
        results["Standard Error of Hedges' gs (Small N)"] = round(standard_error_hedges_gs_Small_n, 4)
        results["Standard Error of Cohen's dpop (True)"] = round(standard_error_cohens_dpop_true, 4)
        results["Standard Error of Cohen's dpop (Morris)"] = round(standard_error_cohens_dpop_morris, 4)
        results["Standard Error of Cohen's dpop (Hedges)"] = round(standard_error_cohens_dpop_hedges, 4)
        results["Standard Error of Cohen's dpop (Hedges_Olkin)"] = round(standard_error_cohens_dpop_hedges_olkin, 4)
        results["Standard Error of Cohen's dpop (MLE)"] = round(standard_error_cohens_dpop_MLE, 4)
        results["Standard Error of Cohen's dpop (Large N)"] = round(standard_error_cohens_dpop_Largen, 4)
        results["Standard Error of Cohen's dpop (Small N)"] = round(standard_error_cohens_dpop_Small_n, 4)

        results["Difference Between Samples"] = round(sample_mean_difference, 4)
        results["Difference in the Population"] = round(population_mean_diff, 4)
        results["Lower Pivotal CI's Cohen's ds"] = round(ci_lower_cohens_ds_Pivotal, 4)
        results["Upper Pivotal CI's Cohen's ds"] = round(ci_upper_cohens_ds_Pivotal, 4)
        results["Lower Pivotal CI's Hedges' gs"] = round(ci_lower_cohens_ds_Pivotal*correction, 4)
        results["Upper Pivotal CI's Hedges' gs"] = round(ci_upper_cohens_ds_Pivotal*correction, 4)
        results["Lower Pivotal CI's Cohen's dpop"] = round(ci_lower_cohens_dpop_Pivotal,4)
        results["Upper Pivotal CI's Cohen's dpop"] = round(ci_upper_cohens_dpop_Pivotal,4)
        results["Lower Central CI's Cohen's ds"] = round(ci_lower_cohens_ds_central, 4)
        results["Upper Central CI's Cohen's ds"] = round(ci_upper_cohens_ds_central, 4)
        results["Lower Central CI's Hedges' gs"] = round(ci_lower_hedges_gs_central, 4)
        results["Upper Central CI's Hedges' gs"] = round(ci_upper_hedges_gs_central, 4)
        results["Lower Central CI's Cohen's dpop"] = round(((ci_lower_cohens_dpop_central)), 4)
        results["Upper Central CI's Cohen's dpop"] = round(((ci_upper_cohens_dpop_central)),4)
        results["Correction Factor"] = round(correction, 4)

        # Ratio of Means
        results["Ratio of Means"] = round(ratio_of_means, 4)
        results["Standard Error of Ratio of Means"] = round(Standard_Error_of_means_ratio, 4)
        results["Lower CI's Ratio of Means"] = round(Lower_CI_Means_Ratio, 4)
        results["Upper CI's Ratio of Means"] = round(Upper_CI_Means_Ratio, 4)
                
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"        
        results["Statistical Line Cohen's ds"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's d\u209B = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cohens_ds, confidence_level_percentages, ci_lower_cohens_ds_Pivotal, ci_upper_cohens_ds_Pivotal)
        results["Statistical Line Hedges' gs"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Hedges' g\u209B = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, hedges_gs, confidence_level_percentages, ci_lower_cohens_ds_Pivotal*correction, ci_upper_cohens_ds_Pivotal*correction)
        results["Statistical Line Cohens' dpop"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's dpop = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cohens_dpop, confidence_level_percentages, ci_lower_cohens_dpop_Pivotal, ci_upper_cohens_dpop_Pivotal)

        return results
        
    @staticmethod
    def two_independent_from_data(params: dict) -> dict:
        
        # Set params
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
        sample_size_1 = len(column_1)
        sample_size_2 = len(column_2)
        sample_size = sample_size_1 + sample_size_2
        df = sample_size - 2
        sample_mean_difference = sample_mean_1 - sample_mean_2
        standardizer_ds = np.sqrt(((((sample_size_1-1)*sample_sd_1**2)) + ((sample_size_2-1)*sample_sd_2**2)) / (sample_size-2))
        standardizer_dpop = np.sqrt(((((sample_size_1-1)*sample_sd_1**2)) + ((sample_size_2-1)*sample_sd_2**2)) / (sample_size))
        standard_error = standardizer_ds  * np.sqrt((sample_size_1+sample_size_2)/(sample_size_1*sample_size_2))
        t_score = ((sample_mean_difference - population_mean_diff) / standard_error)
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)        
        cohens_ds = t_score * (np.sqrt(1/sample_size_1 + 1/sample_size_2))
        cohens_dpop = cohens_ds / np.sqrt((df/sample_size))
        t_score_dpop = cohens_dpop / np.sqrt((1/sample_size_1 + 1/sample_size_2))
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
        hedges_gs = cohens_ds * correction
        standardizer_hedges_gs = standardizer_ds / correction
        ci_lower_cohens_ds_Pivotal, ci_upper_cohens_ds_Pivotal =  Pivotal_ci_t (t_score, df, sample_size_1, sample_size_2, confidence_level)
        ci_lower_cohens_dpop_Pivotal, ci_upper_cohens_dpop_Pivotal =  Pivotal_ci_t (t_score_dpop, df, sample_size_1, sample_size_2, confidence_level)
        ci_lower_cohens_ds_central, ci_upper_cohens_ds_central, standard_error_cohens_ds_true, standard_error_cohens_ds_morris, standard_error_cohens_ds_hedges, standard_error_cohens_ds_hedges_olkin, standard_error_cohens_ds_MLE, standard_error_cohens_ds_Largen, standard_error_cohens_ds_Small_n =  calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test(cohens_ds, sample_size_1, sample_size_2, confidence_level)
        ci_lower_hedges_gs_central, ci_upper_hedges_gs_central,  standard_error_hedges_gs_true, standard_error_hedges_gs_morris, standard_error_hedges_gs_hedges, standard_error_hedges_gs_hedges_olkin, standard_error_hedges_gs_MLE, standard_error_hedges_gs_Largen, standard_error_hedges_gs_Small_n =  calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test(hedges_gs, sample_size_1, sample_size_2, confidence_level)
        ci_lower_cohens_dpop_central, ci_upper_cohens_dpop_central,  standard_error_cohens_dpop_true, standard_error_cohens_dpop_morris, standard_error_cohens_dpop_hedges, standard_error_cohens_dpop_hedges_olkin, standard_error_cohens_dpop_MLE, standard_error_cohens_dpop_Largen, standard_error_cohens_dpop_Small_n =  calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test(cohens_dpop, sample_size_1, sample_size_2, confidence_level)

        # Set results
        results = {}
        results["Cohen's ds"] = round(cohens_ds, 4) # Known as Cohens dp
        results["Hedges' gs"] = round(hedges_gs, 4) 
        results["Cohen's dpop"] = round(cohens_dpop, 4) # Also known as the MLE
        
        # All type of Effect Sizes Standard Errors
        results["Standard Error of Cohen's ds (True)"] = round(standard_error_cohens_ds_true, 4)
        results["Standard Error of Cohen's ds (Morris)"] = round(standard_error_cohens_ds_morris, 4)
        results["Standard Error of Cohen's ds (Hedges)"] = round(standard_error_cohens_ds_hedges, 4)
        results["Standard Error of Cohen's ds (Hedges_Olkin)"] = round(standard_error_cohens_ds_hedges_olkin, 4)
        results["Standard Error of Cohen's ds (MLE)"] = round(standard_error_cohens_ds_MLE, 4)
        results["Standard Error of Cohen's ds (Large N)"] = round(standard_error_cohens_ds_Largen, 4)
        results["Standard Error of Cohen's ds (Small N)"] = round(standard_error_cohens_ds_Small_n, 4)
        results["Standard Error of Hedges' gs (True)"] = round(standard_error_hedges_gs_true, 4)
        results["Standard Error of Hedges' gs (Morris)"] = round(standard_error_hedges_gs_morris, 4)
        results["Standard Error of Hedges' gs (Hedges)"] = round(standard_error_hedges_gs_hedges, 4)
        results["Standard Error of Hedges' gs (Hedges_Olkin)"] = round(standard_error_hedges_gs_hedges_olkin, 4)
        results["Standard Error of Hedges' gs (MLE)"] = round(standard_error_hedges_gs_MLE, 4)
        results["Standard Error of Hedges' gs (Large N)"] = round(standard_error_hedges_gs_Largen, 4)
        results["Standard Error of Hedges' gs (Small N)"] = round(standard_error_hedges_gs_Small_n, 4)
        results["Standard Error of Cohen's dpop (True)"] = round(standard_error_cohens_dpop_true, 4)
        results["Standard Error of Cohen's dpop (Morris)"] = round(standard_error_cohens_dpop_morris, 4)
        results["Standard Error of Cohen's dpop (Hedges)"] = round(standard_error_cohens_dpop_hedges, 4)
        results["Standard Error of Cohen's dpop (Hedges_Olkin)"] = round(standard_error_cohens_dpop_hedges_olkin, 4)
        results["Standard Error of Cohen's dpop (MLE)"] = round(standard_error_cohens_dpop_MLE, 4)
        results["Standard Error of Cohen's dpop (Large N)"] = round(standard_error_cohens_dpop_Largen, 4)
        results["Standard Error of Cohen's dpop (Small N)"] = round(standard_error_cohens_dpop_Small_n, 4)

        results["Standardizer Cohen's ds"] = round(standardizer_ds, 4)
        results["Standardizer Hedges' gs"] = round(standardizer_hedges_gs, 4)
        results["Standardizer Cohen's dpop"] = round(standardizer_dpop, 4)
        results["t-score"] = round(t_score, 4)
        results["Standard Error of the Mean Difference"] = round(standard_error, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["p-value"] = round(p_value, 4)
        results["Sample Mean 1"] = round(sample_mean_1, 4)
        results["Sample Mean 2"] = round(sample_mean_2, 4)
        results["Difference Between Samples"] = round(sample_mean_difference, 4)
        results["Difference in the Population"] = round(population_mean_diff, 4)
        results["Sample Size 1"] = sample_size_1
        results["Sample Size 2"] = sample_size_2
        results["Total Sample Size"] = sample_size
        results["Sample Standard Deviation 1"] = round(sample_sd_1, 4)
        results["Sample Standard Deviation 2"] = round(sample_sd_2, 4)
        results["Lower Pivotal CI's Cohen's ds"] = round(ci_lower_cohens_ds_Pivotal, 4)
        results["Upper Pivotal CI's Cohen's ds"] = round(ci_upper_cohens_ds_Pivotal, 4)
        results["Lower Pivotal CI's Hedges' gs"] = round(ci_lower_cohens_ds_Pivotal*correction, 4)
        results["Upper Pivotal CI's Hedges' gs"] = round(ci_upper_cohens_ds_Pivotal*correction, 4)
        results["Lower Pivotal CI's Cohen's dpop"] = round(ci_lower_cohens_dpop_Pivotal,4)
        results["Upper Pivotal CI's Cohen's dpop"] = round(ci_upper_cohens_dpop_Pivotal,4)
        results["Lower Central CI's Cohen's ds"] = round(ci_lower_cohens_ds_central, 4)
        results["Upper Central CI's Cohen's ds"] = round(ci_upper_cohens_ds_central, 4)
        results["Lower Central CI's Hedges' gs"] = round(ci_lower_hedges_gs_central, 4)
        results["Upper Central CI's Hedges' gs"] = round(ci_upper_hedges_gs_central, 4)
        results["Lower Central CI's Cohen's dpop"] = round(((ci_lower_cohens_dpop_central)), 4)
        results["Upper Central CI's Cohen's dpop"] = round(((ci_upper_cohens_dpop_central)),4)
        results["Correction Factor"] = round(correction, 4)
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Cohen's ds"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's d\u209B = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cohens_ds, confidence_level_percentages, ci_lower_cohens_ds_Pivotal, ci_upper_cohens_ds_Pivotal)
        results["Statistical Line Hedges' gs"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Hedges' g\u209B = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, hedges_gs, confidence_level_percentages, ci_lower_cohens_ds_Pivotal*correction, ci_upper_cohens_ds_Pivotal*correction)
        results["Statistical Line Cohens' dpop"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's dpop = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cohens_dpop, confidence_level_percentages, ci_lower_cohens_dpop_Pivotal, ci_upper_cohens_dpop_Pivotal)

        results["Correction Factor"] = round(correction, 4)
        return results
    
    # Things to Consider
    # 1. Consider Adding Another Central CI based on other measures of SD's just for debugging and for comapring to other sources
    # 2. Consider Adding the Non-Central CI from Cousineau 2020 preprint
    # 3. Consider Changing the name of dpop to dmle (Laken's simply call it Cohens d) 
 

