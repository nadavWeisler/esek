################################################################################
# Effect Size for two dependent samples With A Control Group (Pre-Post) ########
################################################################################

import numpy as np
import math
from scipy.stats import norm, nct, t

# Relevant Functions for Paired Samples t-test
##############################################

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


# See the Meta analysis by Lipsey and Wilson (2001) for more information on the formulas used here
# They Suggested that researechers often used the Standerd Deviation of the Pre meausrements as the standardizer for the effect size

class dependent_Samples_Control_Group():
    @staticmethod
    def Two_dependent_samples_with_Pre_Post_from_parameters(params: dict) -> dict:
        """
        Analyze pre-post data with a control group to assess intervention effects.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            - 'Sample Mean Pre Group': Mean of pre-test measurements for the control group
            - 'Sample Mean Post Group': Mean of post-test measurements for the control group
            - 'Standard Deviation Pre Group': Standard deviation of pre-test measurements for the control group
            - 'Standard Deviation Post Group': Standard deviation of post-test measurements for the control group
            - 'Correlation between Pre and Post': Correlation between pre-test and post-test measurements
            - 'Sample Size': Total sample size
            - 'Difference in the Population': Difference in the population mean between the experimental and control groups
            - 'Confidence Level': Confidence level for the confidence intervals
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'Cohen's d (Pre-Post)': Cohen's d effect size
            - 'Hedges' g (Pre-Post)': Hedges' g effect size
            - 'Standard Error': Standard error of the effect size
            - 't-score': t-score of the effect size
            - 'Degrees of Freedom': Degrees of freedom for the t-test
            - 'p-value': p-value of the t-test
            - 'Standard Error of the Mean Difference': Standard error of the mean difference between the experimental and control groups
            - 'Difference Between Samples': Mean difference between the experimental and control groups
            - 'Difference in the Population': Difference in the population mean between the experimental and control groups
            - 'Lower Pivotal CI's Cohen's d': Lower bound of the pivotal confidence interval for Cohen's d
            - 'Upper Pivotal CI's Cohen's d': Upper bound of the pivotal confidence interval for Cohen's d
            - 'Lower Pivotal CI's Hedges' g': Lower bound of the pivotal confidence interval for Hedges' g
            - 'Upper Pivotal CI's Hedges' g': Upper bound of the pivotal confidence interval for Hedges' g
            - 'Lower Central CI's Cohen's d': Lower bound of the central confidence interval for Cohen's d
            - 'Upper Central CI's Cohen's d': Upper bound of the central confidence interval for Cohen's d
            - 'Lower Central CI's Hedges' g': Lower bound of the central confidence interval for Hedges' g
            - 'Upper Central CI's Hedges' g': Upper bound of the central confidence interval for Hedges' g
            - 'Correction Factor': Correction factor used for Hedges' g
            - 'Statistical Line Cohens'd': Statistical line for Cohen's d
            - 'Statistical Line Hedges' g': Statistical line for Hedges' g
        """
        
        # Set params
        sample_mean_Pre = params["Sample Mean Pre Group"]
        sample_mean_Post = params["Sample Mean Post Group"]
        sample_sd_Pre = params["Standard Deviation Pre Group"]
        sample_sd_Post = params["Standard Deviation Post Group"]
        correlation_pre_post = params["Correlation between Pre and Post"]
        sample_size = params["Sample Size"]
        population_mean_diff = params["Difference in the Population"] # The default value should be 0
        confidence_level_percentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentages / 100
        df = sample_size - 2
        sample_mean_difference = sample_mean_Post - sample_mean_Pre
        standard_error =  np.sqrt(sample_sd_Pre**2 + sample_sd_Post**2 - (2 * correlation_pre_post * sample_sd_Pre * sample_sd_Post))
        t_score = ((sample_mean_difference - population_mean_diff) / standard_error)
        t_score_pre = ((sample_mean_difference - population_mean_diff) / (sample_sd_Pre/np.sqrt(sample_size)))
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)      
        Cohens_d = ((sample_mean_difference - population_mean_diff) / sample_sd_Pre)
        df2 = sample_size - 1
        correction = math.exp(math.lgamma(df2/2) - math.log(math.sqrt(df2/2)) - math.lgamma((df2-1)/2))
        Hedges_pre_post = Cohens_d * correction
        ci_lower_Glass_Delta_Pivotal, ci_upper_Glass_Delta_Pivotal =  Pivotal_ci_t (t_score_pre, df2, sample_size, confidence_level)
        ci_lower_Glass_Delta_central, ci_upper_Glass_Delta_central, standard_error_Cohens_d =  calculate_central_ci_from_cohens_d_one_sample_t_test(Cohens_d, sample_size, confidence_level)
        ci_lower_Glass_Delta_Unbiased_central, ci_upper_Glass_Delta_Unbiased_central, standard_error_Hedges =  calculate_central_ci_from_cohens_d_one_sample_t_test(Hedges_pre_post, sample_size, confidence_level)
 
        # Set results
        results = {}
        results["Cohen's d (Pre-Post)"] = round(Cohens_d, 4) 
        results["Hedges' g (Pre-Post)"] = round(Hedges_pre_post, 4) 
        results["Standard Error"] = round(standard_error, 4)
        results["t-score"] = round(t_score, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["p-value"] = round(p_value, 4)
        results["Standard Error of the Mean Difference"] = round(standard_error, 4)
        results["Difference Between Samples"] = round(sample_mean_difference, 4)
        results["Difference in the Population"] = round(population_mean_diff, 4)
        
        results["Lower Pivotal CI's Cohen's d"] = round(ci_lower_Glass_Delta_Pivotal, 4)
        results["Upper Pivotal CI's Cohen's d"] = round(ci_upper_Glass_Delta_Pivotal, 4)
        results["Lower Pivotal CI's Hedges' g"] = round(ci_lower_Glass_Delta_Pivotal*correction, 4)
        results["Upper Pivotal CI's Hedges' g"] = round(ci_upper_Glass_Delta_Pivotal*correction, 4)

        results["Lower Central CI's Cohen's d"] = round(ci_lower_Glass_Delta_central, 4)
        results["Upper Central CI's Cohen's d"] = round(ci_upper_Glass_Delta_central, 4)
        results["Lower Central CI's Hedges' g"] = round(ci_lower_Glass_Delta_Unbiased_central, 4)
        results["Upper Central CI's Hedges' g"] = round(ci_upper_Glass_Delta_Unbiased_central, 4)
        results["Correction Factor"] = round(correction, 4)

        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Cohens'd"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Glass' Δ = {:.3f},  {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, Hedges_pre_post, confidence_level_percentages, ci_lower_Glass_Delta_Pivotal, ci_upper_Glass_Delta_Pivotal)
        results["Statistical Line Hedges' g"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Unbiased Glass' Δ = {:.3f},  {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, Cohens_d, confidence_level_percentages, ci_lower_Glass_Delta_Pivotal*correction, ci_upper_Glass_Delta_Pivotal*correction)

        return results
        
    @staticmethod
    def Two_dependent_samples_with_Control_Group_from_data(params: dict) -> dict:
        
        # Set params
        column_1 = params["column_1"] # This is the coloumn of the Control Group
        column_2 = params["column_2"] # This is the coloumn of the Experimental Group
        population_mean_diff = params["Difference in the Population"] # The default value should be 0
        confidence_level_percentages = params["Confidence Level"]
        
        # Calculation
        confidence_level = confidence_level_percentages / 100

        sample_mean_Pre = np.mean(column_1)
        sample_mean_Post = np.mean(column_2)
        sample_sd_Pre = np.std(column_1, ddof = 1)
        sample_sd_Post = np.std(column_2, ddof = 1)
        sample_size_Control = len(column_1)
        sample_size_Experimental = len(column_2)
        sample_size = sample_size_Control + sample_size_Experimental
        df = sample_size - 2
        sample_mean_difference = sample_mean_Pre - sample_mean_Post
        correlation_pre_post = np.corrcoef(column_1, column_2)[0][1]
        
        
        # Calculation
        confidence_level = confidence_level_percentages / 100
        df = sample_size - 2
        sample_mean_difference = sample_mean_Post - sample_mean_Pre
        standard_error =  np.sqrt(sample_sd_Pre**2 + sample_sd_Post**2 - (2 * correlation_pre_post * sample_sd_Pre * sample_sd_Post))
        t_score = ((sample_mean_difference - population_mean_diff) / standard_error)
        t_score_pre = ((sample_mean_difference - population_mean_diff) / (sample_sd_Pre/np.sqrt(sample_size)))
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)      
        Cohens_d = ((sample_mean_difference - population_mean_diff) / sample_sd_Pre)
        df2 = sample_size - 1
        correction = math.exp(math.lgamma(df2/2) - math.log(math.sqrt(df2/2)) - math.lgamma((df2-1)/2))
        Hedges_pre_post = Cohens_d * correction
        ci_lower_Glass_Delta_Pivotal, ci_upper_Glass_Delta_Pivotal =  Pivotal_ci_t (t_score_pre, df2, sample_size, confidence_level)
        ci_lower_Glass_Delta_central, ci_upper_Glass_Delta_central, standard_error_Cohens_d =  calculate_central_ci_from_cohens_d_one_sample_t_test(Cohens_d, sample_size, confidence_level)
        ci_lower_Glass_Delta_Unbiased_central, ci_upper_Glass_Delta_Unbiased_central, standard_error_Hedges =  calculate_central_ci_from_cohens_d_one_sample_t_test(Hedges_pre_post, sample_size, confidence_level)
 
        # Set results
        results = {}
        results["Cohen's d (Pre-Post)"] = round(Cohens_d, 4) 
        results["Hedges' g (Pre-Post)"] = round(Hedges_pre_post, 4) 
        results["Standard Error"] = round(standard_error, 4)
        results["t-score"] = round(t_score, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["p-value"] = round(p_value, 4)
        results["Standard Error of the Mean Difference"] = round(standard_error, 4)
        results["Difference Between Samples"] = round(sample_mean_difference, 4)
        results["Difference in the Population"] = round(population_mean_diff, 4)
        
        results["Lower Pivotal CI's Cohen's d"] = round(ci_lower_Glass_Delta_Pivotal, 4)
        results["Upper Pivotal CI's Cohen's d"] = round(ci_upper_Glass_Delta_Pivotal, 4)
        results["Lower Pivotal CI's Hedges' g"] = round(ci_lower_Glass_Delta_Pivotal*correction, 4)
        results["Upper Pivotal CI's Hedges' g"] = round(ci_upper_Glass_Delta_Pivotal*correction, 4)

        results["Lower Central CI's Cohen's d"] = round(ci_lower_Glass_Delta_central, 4)
        results["Upper Central CI's Cohen's d"] = round(ci_upper_Glass_Delta_central, 4)
        results["Lower Central CI's Hedges' g"] = round(ci_lower_Glass_Delta_Unbiased_central, 4)
        results["Upper Central CI's Hedges' g"] = round(ci_upper_Glass_Delta_Unbiased_central, 4)
        results["Correction Factor"] = round(correction, 4)

        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Cohens d (Pre-Post)"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Glass' Δ = {:.3f},  {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, Hedges_pre_post, confidence_level_percentages, ci_lower_Glass_Delta_Pivotal, ci_upper_Glass_Delta_Pivotal)
        results["Statistical Line Hedges' g (Pre-Post)"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Unbiased Glass' Δ = {:.3f},  {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, Cohens_d, confidence_level_percentages, ci_lower_Glass_Delta_Pivotal*correction, ci_upper_Glass_Delta_Pivotal*correction)

        return results

   

 


