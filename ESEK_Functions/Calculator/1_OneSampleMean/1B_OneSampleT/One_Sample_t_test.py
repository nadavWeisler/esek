###############################################
##### Effect Size for One Sample t-Test #######
###############################################

import numpy as np
import math
from scipy.stats import norm, nct, t

# Relevant Functions for One Sample t-test
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
def calculate_central_ci_one_sample_t_test(effect_size, sample_size, confidence_level): # This is a function that calculates the Confidence Intervals of the Effect size in One Sample Z_score test (or two dependent samples)
    df = sample_size - 1 # This is the Degrees of Freedom for one sample t-test
    correction_factor =  math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
    Standard_error_effect_size_True = np.sqrt(((df/(df-2)) * (1 / sample_size ) * (1 + effect_size**2 * sample_size)  - (effect_size**2 / correction_factor**2)))
    Standard_error_effect_size_Morris = np.sqrt((df/(df-2)) * (1 / sample_size ) * (1 + effect_size**2 * sample_size)  - (effect_size**2 / (1 - (3/(4*(df-1)-1)))**2))
    Standard_error_effect_size_Hedges =  np.sqrt((1/sample_size) + effect_size**2/ (2*df)) 
    Standard_error_effect_size_Hedges_Olkin = np.sqrt((1/sample_size) + effect_size**2/ (2*sample_size))
    Standard_error_effect_size_MLE = np.sqrt(Standard_error_effect_size_Hedges * ((df+2)/df))
    Standard_error_effect_size_Large_N = np.sqrt(1/ sample_size * (1 + effect_size**2/8))
    Standard_error_effect_size_Small_N = np.sqrt(Standard_error_effect_size_Large_N * ((df+1)/(df-1)))
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = effect_size - Standard_error_effect_size_True * z_critical_value,  effect_size + Standard_error_effect_size_True * z_critical_value
    return ci_lower, ci_upper, Standard_error_effect_size_True, Standard_error_effect_size_Morris, Standard_error_effect_size_Hedges, Standard_error_effect_size_Hedges_Olkin, Standard_error_effect_size_MLE, Standard_error_effect_size_Large_N, Standard_error_effect_size_Small_N

# 3. Non Central Parameter CI's (see Cousineau)
def CI_NCP_one_Sample(Effect_Size, sample_size, confidence_level):
    NCP_value = Effect_Size * math.sqrt(sample_size)
    CI_NCP_low = (nct.ppf(1/2 - confidence_level/2, (sample_size - 1), loc=0, scale=1, nc=NCP_value)) / NCP_value * Effect_Size
    CI_NCP_High = (nct.ppf(1/2 + confidence_level/2, (sample_size - 1), loc=0, scale=1, nc=NCP_value)) / NCP_value * Effect_Size
    return CI_NCP_low, CI_NCP_High

class One_Sample_ttest():
    @staticmethod
    def one_sample_from_t_score(params: dict) -> dict:
    
    # Get params
       t_score = params["t score"]
       sample_size = params["Sample Size"]
       confidence_level_percentage = params["Confidence Level"]

       # Calculation
       confidence_level = (confidence_level_percentage / 100)
       df = sample_size - 1
       p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)
       cohens_d = (t_score/np.sqrt(df)) # This is Cohen's d and it is calculated based on the sample's standard deviation
       correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
       hedges_g = correction*cohens_d
       ci_lower_cohens_d_central, ci_upper_cohens_d_central, Standard_error_cohens_d_true, Standard_error_cohens_d_morris, Standard_error_cohens_d_hedges, Standard_error_cohens_d_hedges_olkin, Standard_error_cohens_d_MLE, Standard_error_cohens_d_Largen, Standard_error_cohens_d_Small_n =  calculate_central_ci_one_sample_t_test (cohens_d, sample_size, confidence_level)
       ci_lower_hedges_g_central, ci_upper_hedges_g_central, Standard_error_hedges_g_true, Standard_error_hedges_g_morris, Standard_error_hedges_g_hedges, Standard_error_hedges_g_hedges_olkin, Standard_error_hedges_g_MLE, Standard_error_hedges_g_Largen, Standard_error_hedges_g_Small_n =  calculate_central_ci_one_sample_t_test (hedges_g, sample_size, confidence_level)
       ci_lower_cohens_d_Pivotal, ci_upper_cohens_d_Pivotal =  Pivotal_ci_t (t_score, df, sample_size, confidence_level)
       ci_lower_hedges_g_Pivotal, ci_upper_hedges_g_Pivotal =  Pivotal_ci_t (t_score, df, sample_size, confidence_level)
       ci_lower_cohens_d_NCP, ci_upper_cohens_d_NCP =  CI_NCP_one_Sample (cohens_d, sample_size, confidence_level)
       ci_lower_hedges_g_NCP, ci_upper_hedges_g_NCP =  CI_NCP_one_Sample (hedges_g, sample_size, confidence_level)

              
       # Set results
       results = {}
       results["Cohen's d"] = round(cohens_d, 4)
       results["Hedges' g"] = round(hedges_g, 4)
       results["t score"] = round(t_score, 4)
       results["Degrees of Freedom"] = round(df, 4)
       results["p-value"] = round(p_value, 4)
       results["Standard Error of Cohen's d (True)"] = round(Standard_error_cohens_d_true, 4)
       results["Standard Error of Cohen's d (Morris)"] = round(Standard_error_cohens_d_morris, 4)
       results["Standard Error of Cohen's d (Hedges)"] = round(Standard_error_cohens_d_hedges, 4)
       results["Standard Error of Cohen's d (Hedges_Olkin)"] = round(Standard_error_cohens_d_hedges_olkin, 4)
       results["Standard Error of Cohen's d (MLE)"] = round(Standard_error_cohens_d_MLE, 4)
       results["Standard Error of Cohen's d (Large N)"] = round(Standard_error_cohens_d_Largen, 4)
       results["Standard Error of Cohen's d (Small N)"] = round(Standard_error_cohens_d_Small_n, 4)
       results["Standard Error of Hedges' g (True)"] = round(Standard_error_hedges_g_true, 4)
       results["Standard Error of Hedges' g (Morris)"] = round(Standard_error_hedges_g_morris, 4)
       results["Standard Error of Hedges' g (Hedges)"] = round(Standard_error_hedges_g_hedges, 4)
       results["Standard Error of Hedges' g (Hedges_Olkin)"] = round(Standard_error_hedges_g_hedges_olkin, 4)
       results["Standard Error of Hedges' g (MLE)"] = round(Standard_error_hedges_g_MLE, 4)
       results["Standard Error of Hedges' g (Large N)"] = round(Standard_error_hedges_g_Largen, 4)
       results["Standard Error of Hedges' g (Small N)"] = round(Standard_error_hedges_g_Small_n, 4)
       results["Lower Central CI's Cohen's d"] = round(ci_lower_cohens_d_central, 4)
       results["Upper Central CI's Cohen's d"] = round(ci_upper_cohens_d_central, 4)
       results["Lower Central CI's Hedges' g"] = round(ci_lower_hedges_g_central, 4)
       results["Upper Central CI's Hedges' g"] = round(ci_upper_hedges_g_central, 4)
       results["Lower Pivotal CI's Cohen's d"] = round(ci_lower_cohens_d_Pivotal, 4)
       results["Upper Pivotal CI's Cohen's d"] = round(ci_upper_cohens_d_Pivotal, 4)
       results["Lower Pivotal CI's Hedges' g"] = round(ci_lower_hedges_g_Pivotal*correction, 4)
       results["Upper Pivotal CI's Hedges' g"] = round(ci_upper_hedges_g_Pivotal*correction, 4)
       results["Lower NCP CI's Cohen's d"] = round(ci_lower_cohens_d_NCP, 4)      
       results["Upper NCP CI's Cohen's d"] = round(ci_upper_cohens_d_NCP, 4)
       results["Lower NCP CI's Hedges' g"] = round(ci_lower_hedges_g_NCP, 4)
       results["Upper NCP CI's Hedges' g"] = round(ci_upper_hedges_g_NCP, 4)
       results["Correction Factor"] = round(correction, 4)
       formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
       results["Statistical Line Cohen's d"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's d = {:.3f}, {}% CI(Pivotal) [{:.3f},{:.3f}]".format(int(df), t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cohens_d ,confidence_level_percentage, round(ci_lower_cohens_d_Pivotal,3), round(ci_upper_cohens_d_Pivotal,3))
       results["Statistical Line Hedges' g"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Hedges' g = {:.3f}, {}% CI(Pivotal) [{:.3f},{:.3f}]".format(int(df), t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, hedges_g, confidence_level_percentage, round(ci_lower_hedges_g_Pivotal*correction, 3), round(ci_upper_hedges_g_Pivotal*correction, 3))
           
       return results
    
    @staticmethod
    def one_sample_from_params(params: dict) -> dict:
        
        # Set params
        population_mean = params["Population Mean"]
        sample_mean = params["Mean Sample"]
        sample_sd = params["Standard Deviation Sample"]
        sample_size = params["Sample Size"]
        confidence_level_percentage = params["Confidence Level"]

        # Calculation
        confidence_level = (confidence_level_percentage / 100)
        df = sample_size-1
        standard_error = sample_sd/np.sqrt(df) #This is the standrt error of mean's estimate in o ne samaple t-test
        t_score = (sample_mean - population_mean)/standard_error #This is the t score in the test which is used to calculate the p-value
        cohens_d = ((sample_mean - population_mean)/sample_sd) #This is the effect size for one sample t-test Cohen's d
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
        hedges_g = cohens_d*correction #This is the actual corrected effect size        
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)
        ci_lower_cohens_d_central, ci_upper_cohens_d_central, Standard_error_cohens_d_true, Standard_error_cohens_d_morris, Standard_error_cohens_d_hedges, Standard_error_cohens_d_hedges_olkin, Standard_error_cohens_d_MLE, Standard_error_cohens_d_Largen, Standard_error_cohens_d_Small_n =  calculate_central_ci_one_sample_t_test (cohens_d, sample_size, confidence_level)
        ci_lower_hedges_g_central, ci_upper_hedges_g_central, Standard_error_hedges_g_true, Standard_error_hedges_g_morris, Standard_error_hedges_g_hedges, Standard_error_hedges_g_hedges_olkin, Standard_error_hedges_g_MLE, Standard_error_hedges_g_Largen, Standard_error_hedges_g_Small_n =  calculate_central_ci_one_sample_t_test (hedges_g, sample_size, confidence_level)
        ci_lower_cohens_d_Pivotal, ci_upper_cohens_d_Pivotal =  Pivotal_ci_t (t_score, df, sample_size, confidence_level)
        ci_lower_hedges_g_Pivotal, ci_upper_hedges_g_Pivotal =  Pivotal_ci_t (t_score, df, sample_size, confidence_level)
        ci_lower_cohens_d_NCP, ci_upper_cohens_d_NCP =  CI_NCP_one_Sample (cohens_d, sample_size, confidence_level)
        ci_lower_hedges_g_NCP, ci_upper_hedges_g_NCP =  CI_NCP_one_Sample (hedges_g, sample_size, confidence_level)
        
        # Set results
        results = {}
        results["Cohen's d"] = round(cohens_d, 4)
        results["Hedges' g"] = round(hedges_g, 4)
        results["t score"] = round(t_score, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["p-value"] = round(p_value, 4)
        results["Standardizer Cohen's d (Sample's Standard Deviation)"] = round(sample_sd, 4)
        results["Standardizer Hedge's g"] = round(sample_sd /correction, 4)
        results["Standard Error of the Mean"] = round(standard_error, 4)
        results["Standard Error of Cohen's d (True)"] = round(Standard_error_cohens_d_true, 4)
        results["Standard Error of Cohen's d (Morris)"] = round(Standard_error_cohens_d_morris, 4)
        results["Standard Error of Cohen's d (Hedges)"] = round(Standard_error_cohens_d_hedges, 4)
        results["Standard Error of Cohen's d (Hedges_Olkin)"] = round(Standard_error_cohens_d_hedges_olkin, 4)
        results["Standard Error of Cohen's d (MLE)"] = round(Standard_error_cohens_d_MLE, 4)
        results["Standard Error of Cohen's d (Large N)"] = round(Standard_error_cohens_d_Largen, 4)
        results["Standard Error of Cohen's d (Small N)"] = round(Standard_error_cohens_d_Small_n, 4)
        results["Standard Error of Hedges' g (True)"] = round(Standard_error_hedges_g_true, 4)
        results["Standard Error of Hedges' g (Morris)"] = round(Standard_error_hedges_g_morris, 4)
        results["Standard Error of Hedges' g (Hedges)"] = round(Standard_error_hedges_g_hedges, 4)
        results["Standard Error of Hedges' g (Hedges_Olkin)"] = round(Standard_error_hedges_g_hedges_olkin, 4)
        results["Standard Error of Hedges' g (MLE)"] = round(Standard_error_hedges_g_MLE, 4)
        results["Standard Error of Hedges' g (Large N)"] = round(Standard_error_hedges_g_Largen, 4)
        results["Standard Error of Hedges' g (Small N)"] = round(Standard_error_hedges_g_Small_n, 4)
        results["Lower Central CI's Cohen's d"] = round(ci_lower_cohens_d_central, 4)
        results["Upper Central CI's Cohen's d"] = round(ci_upper_cohens_d_central, 4)
        results["Lower Central CI's Hedges' g"] = round(ci_lower_hedges_g_central, 4)
        results["Upper Central CI's Hedges' g"] = round(ci_upper_hedges_g_central, 4)
        results["Lower Pivotal CI's Cohen's d"] = round(ci_lower_cohens_d_Pivotal, 4)
        results["Upper Pivotal CI's Cohen's d"] = round(ci_upper_cohens_d_Pivotal, 4)
        results["Lower Pivotal CI's Hedges' g"] = round(ci_lower_hedges_g_Pivotal*correction, 4)
        results["Upper Pivotal CI's Hedges' g"] = round(ci_upper_hedges_g_Pivotal*correction, 4)
        results["Lower NCP CI's Cohen's d"] = round(ci_lower_cohens_d_NCP, 4)      
        results["Upper NCP CI's Cohen's d"] = round(ci_upper_cohens_d_NCP, 4)
        results["Lower NCP CI's Hedges' g"] = round(ci_lower_hedges_g_NCP, 4)
        results["Upper NCP CI's Hedges' g"] = round(ci_upper_hedges_g_NCP, 4)
        results["Correction Factor"] = round(correction, 4)
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Cohen's d"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's d = {:.3f}, {}% CI(Pivotal) [{:.3f},{:.3f}]".format(int(df), t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cohens_d ,confidence_level_percentage, round(ci_lower_cohens_d_Pivotal,3), round(ci_upper_cohens_d_Pivotal,3))
        results["Statistical Line Hedges' g"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Hedges' g = {:.3f}, {}% CI(Pivotal) [{:.3f},{:.3f}]".format(int(df), t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, hedges_g, confidence_level_percentage, round(ci_lower_hedges_g_Pivotal*correction, 3), round(ci_upper_hedges_g_Pivotal*correction, 3))
                
        return results

    @staticmethod
    def one_sample_from_data(params: dict) -> dict:
        
        # Set params
        column_1 = params["column 1"]
        population_mean = params["Population's Mean"] #Default should be 0 if not mentioned
        confidence_level_percentage = params["Confidence Level"]

        # Calculation
        confidence_level = (confidence_level_percentage / 100)
        sample_mean = np.mean(column_1)
        sample_sd = np.std(column_1, ddof = 1)
        sample_size = len(column_1)
        df = sample_size-1 # Degrees of freedom one sample t-test
        standard_error = sample_sd/np.sqrt(df) #This is the standrd error of mean's estimate in one samaple t-test
        t_score = (sample_mean - population_mean)/standard_error #This is the t score in the test which is used to calculate the p-value
        cohens_d = ((sample_mean - population_mean)/sample_sd) #This is the effect size for one sample t-test Cohen's d
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
        hedges_g = cohens_d*correction #This is the actual corrected effect size        
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)
        ci_lower_cohens_d_central, ci_upper_cohens_d_central, Standard_error_cohens_d_true, Standard_error_cohens_d_morris, Standard_error_cohens_d_hedges, Standard_error_cohens_d_hedges_olkin, Standard_error_cohens_d_MLE, Standard_error_cohens_d_Largen, Standard_error_cohens_d_Small_n =  calculate_central_ci_one_sample_t_test (cohens_d, sample_size, confidence_level)
        ci_lower_hedges_g_central, ci_upper_hedges_g_central, Standard_error_hedges_g_true, Standard_error_hedges_g_morris, Standard_error_hedges_g_hedges, Standard_error_hedges_g_hedges_olkin, Standard_error_hedges_g_MLE, Standard_error_hedges_g_Largen, Standard_error_hedges_g_Small_n =  calculate_central_ci_one_sample_t_test (hedges_g, sample_size, confidence_level)
        ci_lower_cohens_d_Pivotal, ci_upper_cohens_d_Pivotal =  Pivotal_ci_t (t_score, df, sample_size, confidence_level)
        ci_lower_hedges_g_Pivotal, ci_upper_hedges_g_Pivotal =  Pivotal_ci_t (t_score, df, sample_size, confidence_level)
        ci_lower_cohens_d_NCP, ci_upper_cohens_d_NCP =  CI_NCP_one_Sample (cohens_d, sample_size, confidence_level)
        ci_lower_hedges_g_NCP, ci_upper_hedges_g_NCP =  CI_NCP_one_Sample (hedges_g, sample_size, confidence_level)

        # Set results
        results = {}
        results["Cohen's d"] = round(cohens_d, 4)
        results["Hedges' g"] = round(hedges_g, 4)
        results["t_score"] = round(t_score, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["p-value"] = round(p_value, 4)
        results["Standard Error of Cohen's d (True)"] = round(Standard_error_cohens_d_true, 4)
        results["Standard Error of Cohen's d (Morris)"] = round(Standard_error_cohens_d_morris, 4)
        results["Standard Error of Cohen's d (Hedges)"] = round(Standard_error_cohens_d_hedges, 4)
        results["Standard Error of Cohen's d (Hedges_Olkin)"] = round(Standard_error_cohens_d_hedges_olkin, 4)
        results["Standard Error of Cohen's d (MLE)"] = round(Standard_error_cohens_d_MLE, 4)
        results["Standard Error of Cohen's d (Large N)"] = round(Standard_error_cohens_d_Largen, 4)
        results["Standard Error of Cohen's d (Small N)"] = round(Standard_error_cohens_d_Small_n, 4)
        results["Standard Error of Hedges' g (True)"] = round(Standard_error_hedges_g_true, 4)
        results["Standard Error of Hedges' g (Morris)"] = round(Standard_error_hedges_g_morris, 4)
        results["Standard Error of Hedges' g (Hedges)"] = round(Standard_error_hedges_g_hedges, 4)
        results["Standard Error of Hedges' g (Hedges_Olkin)"] = round(Standard_error_hedges_g_hedges_olkin, 4)
        results["Standard Error of Hedges' g (MLE)"] = round(Standard_error_hedges_g_MLE, 4)
        results["Standard Error of Hedges' g (Large N)"] = round(Standard_error_hedges_g_Largen, 4)
        results["Standard Error of Hedges' g (Small N)"] = round(Standard_error_hedges_g_Small_n, 4)
        results["Standard Error of the Mean"] = round(standard_error, 4)
        results["Standardizer Cohen's d (Sample's Standard Deviation)"] = round(sample_sd, 4)
        results["Standardizer Hedge's g"] = round(sample_sd /correction, 4)
        results["Sample's Mean"] = round(sample_mean, 4)
        results["Population's Mean"] = round(population_mean, 4)
        results["Means Difference"] = round(sample_mean - population_mean, 4)
        results["Sample Size"] = round(sample_size, 4)
        results["Sample's Standard Deviation"] = round(sample_sd, 4)
        results["Lower Central CI's Cohen's d"] = round(ci_lower_cohens_d_central, 4)
        results["Upper Central CI's Cohen's d"] = round(ci_upper_cohens_d_central, 4)
        results["Lower NCP CI's Cohen's d"] = round(ci_lower_cohens_d_NCP, 4)      
        results["Upper NCP CI's Cohen's d"] = round(ci_upper_cohens_d_NCP, 4)

        results["Lower Pivotal CI's Cohen's d"] = round(ci_lower_cohens_d_Pivotal, 4)
        results["Upper Pivotal CI's Cohen's d"] = round(ci_upper_cohens_d_Pivotal, 4)
        results["Lower Pivotal CI's Hedges' g"] = round(ci_lower_hedges_g_Pivotal*correction, 4)
        results["Upper Pivotal CI's Hedges' g"] = round(ci_upper_hedges_g_Pivotal*correction, 4)
        results["Lower Central CI's Hedges' g"] = round(ci_lower_hedges_g_central, 4)
        results["Upper Central CI's Hedges' g"] = round(ci_upper_hedges_g_central, 4)
        results["Lower NCP CI's Hedges' g"] = round(ci_lower_hedges_g_NCP, 4)
        results["Upper NCP CI's Hedges' g"] = round(ci_upper_hedges_g_NCP, 4)
        results["Correction Factor"] = round(correction, 4)    
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Cohen's d"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's d = {:.3f}, {}% CI(Pivotal) [{:.3f},{:.3f}]".format(int(df), t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cohens_d ,confidence_level_percentage, round(ci_lower_cohens_d_Pivotal,3), round(ci_upper_cohens_d_Pivotal,3))
        results["Statistical Line Hedges' g"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Hedges' g = {:.3f}, {}% CI(Pivotal) [{:.3f},{:.3f}]".format(int(df), t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, hedges_g, confidence_level_percentage, round(ci_lower_hedges_g_Pivotal*correction, 3), round(ci_upper_hedges_g_Pivotal*correction, 3))
            
        return results

    
    # Things to consider

    # 1. Using a different default for CI - maybe switch to the NCP's one
    # 2. imporve the Pivotal accuracy to match r functions...