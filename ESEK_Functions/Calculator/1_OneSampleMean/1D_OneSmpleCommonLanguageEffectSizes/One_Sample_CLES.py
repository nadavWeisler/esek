

########################################################
##### Common Labguage Effect Size for One Sample #######
########################################################

import numpy as np
import math
from scipy.stats import norm, nct, t, trim_mean

# Relevant Functions for The Common Languge Effect Size
#######################################################

# 1. Pivotal CI Function
def pivotal_ci_t(t_Score, df, sample_size, confidence_level):
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

# 3. Non Central Parameter CI's (see Cousineau)
def CI_NCP_one_Sample(Effect_Size, sample_size, confidence_level):
    NCP_value = Effect_Size * math.sqrt(sample_size)
    CI_NCP_low = (nct.ppf(1/2 - confidence_level/2, (sample_size - 1), loc=0, scale=1, nc=NCP_value)) / NCP_value * Effect_Size
    CI_NCP_High = (nct.ppf(1/2 + confidence_level/2, (sample_size - 1), loc=0, scale=1, nc=NCP_value)) / NCP_value * Effect_Size
    return CI_NCP_low, CI_NCP_High

####### Relevant functions for the Robust measure of effect size
############################################
def density(x):
    x = np.array(x)
    return x**2 * norm.pdf(x)

def area_under_function(f, a, b, *args, function_a=None, function_b=None, limit=10, eps=1e-5):
    if function_a is None:
        function_a = f(a, *args)
    if function_b is None:
        function_b = f(b, *args)
    midpoint = (a + b) / 2
    f_midpoint = f(midpoint, *args)
    area_trapezoidal = ((function_a + function_b) * (b - a)) / 2
    area_simpson = ((function_a + 4 * f_midpoint + function_b) * (b - a)) / 6
    if abs(area_trapezoidal - area_simpson) < eps or limit == 0:
        return area_simpson
    return area_under_function(f, a, midpoint, *args, function_a=function_a, function_b=f_midpoint, limit=limit-1, eps=eps) + area_under_function(f, midpoint, b, *args, function_a=f_midpoint, function_b=function_b, limit=limit-1, eps=eps)

def WinsorizedVariance(x, trimming_level=0.2):
    y = np.sort(x)
    n = len(x)
    ibot = int(np.floor(trimming_level * n)) + 1
    itop = n - ibot + 1
    xbot = y[ibot-1] 
    xtop = y[itop-1]
    y = np.where(y <= xbot, xbot, y)
    y = np.where(y >= xtop, xtop, y)
    winvar = np.std(y, ddof=1)**2
    return winvar

def WinsorizedCorrelation(x, y, trimming_level=0.2):
    sample_size = len(x)
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    trimming_size = int(np.floor(trimming_level * sample_size)) + 1
    x_lower = x_sorted[trimming_size - 1]
    x_upper = x_sorted[sample_size - trimming_size]
    y_lower = y_sorted[trimming_size - 1]
    y_upper = y_sorted[sample_size - trimming_size]
    x_winzsorized = np.clip(x, x_lower, x_upper)
    y_winzsorized = np.clip(y, y_lower, y_upper)
    winsorized_correlation = np.corrcoef(x_winzsorized, y_winzsorized)[0, 1]
    winsorized_covariance = np.cov(x_winzsorized, y_winzsorized)[0, 1]
    test_statistic = winsorized_correlation * np.sqrt((sample_size - 2) / (1 - winsorized_correlation**2))
    Number_of_trimmed_values = int(np.floor(trimming_level * sample_size))
    p_value = 2 * (1 - t.cdf(np.abs(test_statistic), sample_size - 2*Number_of_trimmed_values - 2))
    return {'cor': winsorized_correlation, 'cov': winsorized_covariance, 'p.value': p_value, 'n': sample_size, 'test_statistic': test_statistic}


class One_Sample_ttest():
    @staticmethod
    def one_sample_from_t_score(params: dict) -> dict:
    
    # Get params
       t_score = params["t score"]
       sample_size = params["Sample Size"]
       confidence_level_percentages = params["Confidence Level"]

       # Calculation
       confidence_level = confidence_level_percentages/100
       df = sample_size - 1
       p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)
       cohens_d = (t_score/np.sqrt(df)) # This is Cohen's d and it is calculated based on the sample's standard deviation
       correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
       hedges_g = correction*cohens_d
       cles_d = norm.cdf(cohens_d) * 100
       cles_g = norm.cdf(hedges_g) * 100

       ci_lower_cohens_d_central, ci_upper_cohens_d_central, standrat_error_cohens_d =  calculate_central_ci_from_cohens_d_one_sample_t_test (cohens_d, sample_size, confidence_level)
       ci_lower_hedges_g_central, ci_upper_hedges_g_central, standrat_error_hedges_g =  calculate_central_ci_from_cohens_d_one_sample_t_test (hedges_g, sample_size, confidence_level)
       ci_lower_cohens_d_pivotal, ci_upper_cohens_d_pivotal =  pivotal_ci_t (t_score, df, sample_size, confidence_level)
       ci_lower_hedges_g_pivotal, ci_upper_hedges_g_pivotal =  pivotal_ci_t (t_score, df, sample_size, confidence_level)
       ci_lower_cohens_d_NCP, ci_upper_cohens_d_NCP =  CI_NCP_one_Sample (cohens_d, sample_size, confidence_level)
       ci_lower_hedges_g_NCP, ci_upper_hedges_g_NCP =  CI_NCP_one_Sample (hedges_g, sample_size, confidence_level)
              
       # Set results
       results = {}

       results["Lower Central CI's CLd"] = np.around(norm.cdf(ci_lower_cohens_d_central) * 100, 4)
       results["Upper Central CI's CLd"] = np.around(norm.cdf(ci_upper_cohens_d_central) *100, 4)
       results["Lower Central CI's CLg"] = np.around(norm.cdf(ci_lower_hedges_g_central) *100, 4)
       results["Upper Central CI's CLg"] = np.around(norm.cdf(ci_upper_hedges_g_central) *100, 4)
    
       results["Lower Non-Central CI's CLd"] = np.around(norm.cdf(ci_lower_cohens_d_NCP) *100, 4)
       results["Upper Non-Central CI's CLd"] = np.around(norm.cdf(ci_upper_cohens_d_NCP) *100, 4)
       results["Lower Non-Central CI's CLg"] = np.around(norm.cdf(ci_lower_hedges_g_NCP) *100, 4)
       results["Upper Non-Central CI's CLg"] = np.around(norm.cdf(ci_upper_hedges_g_NCP) *100, 4)

       results["Lower Pivotal CI's CLd"] = np.around(norm.cdf(ci_lower_cohens_d_pivotal) *100, 4)
       results["Upper Pivotal CI's CLd"] = np.around(norm.cdf(ci_upper_cohens_d_pivotal) *100, 4)
       results["Lower Pivotal CI's CLg"] = np.around(norm.cdf(ci_lower_hedges_g_pivotal*correction) *100, 4)
       results["Upper Pivotal CI's CLg"] = np.around(norm.cdf(ci_upper_hedges_g_pivotal*correction) *100, 4)
       formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001" 
       results["Statistical Line CLd"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLd = {:.3f}, {}% CI(pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cles_d,confidence_level_percentages,  np.around(norm.cdf(ci_lower_cohens_d_pivotal) *100, 3),  np.around(norm.cdf(ci_upper_cohens_d_pivotal) *100, 3))
       results["Statistical Line CLg"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLg = {:.3f}, {}% CI(pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cles_g,confidence_level_percentages,  np.around(norm.cdf(ci_lower_hedges_g_pivotal*correction) *100, 3), np.around(norm.cdf(ci_upper_hedges_g_pivotal*correction) *100, 3))

       return results
    
    @staticmethod
    def one_sample_from_params(params: dict) -> dict:
        
        # Set params
        population_mean = params["Population Mean"]
        sample_mean = params["Mean Sample"]
        sample_sd = params["Standard Deviation Sample"]
        sample_size = params["Sample Size"]
        confidence_level_percentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentages/100
        df = sample_size-1
        standard_error = sample_sd/np.sqrt(df) #This is the standrt error of mean's estimate in o ne samaple t-test
        t_score = (sample_mean - population_mean)/standard_error #This is the t score in the test which is used to calculate the p-value
        cohens_d = ((sample_mean - population_mean)/sample_sd) #This is the effect size for one sample t-test Cohen's d
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
        hedges_g = cohens_d*correction #This is the actual corrected effect size  
        cles_d = norm.cdf(cohens_d) * 100
        cles_g = norm.cdf(hedges_g) * 100      
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)
        ci_lower_cohens_d_central, ci_upper_cohens_d_central, standrat_error_cohens_d =  calculate_central_ci_from_cohens_d_one_sample_t_test (cohens_d, sample_size, confidence_level)
        ci_lower_hedges_g_central, ci_upper_hedges_g_central, standrat_error_hedges_g =  calculate_central_ci_from_cohens_d_one_sample_t_test (hedges_g, sample_size, confidence_level)
        ci_lower_cohens_d_pivotal, ci_upper_cohens_d_pivotal =  pivotal_ci_t (t_score, df, sample_size, confidence_level)
        ci_lower_hedges_g_pivotal, ci_upper_hedges_g_pivotal =  pivotal_ci_t (t_score, df, sample_size, confidence_level)
        ci_lower_cohens_d_NCP, ci_upper_cohens_d_NCP =  CI_NCP_one_Sample (cohens_d, sample_size, confidence_level)
        ci_lower_hedges_g_NCP, ci_upper_hedges_g_NCP =  CI_NCP_one_Sample (hedges_g, sample_size, confidence_level)
        
        # Set results
        results = {}

        results["Mcgraw & Wong, Common Language Effect Size (CLd)"] = np.around(cles_d, 4)
        results["Mcgraw & Wong, Unbiased Common Language Effect Size (CLg)"] = np.around(cles_g, 4)
        results["t-score"] = np.around(t_score, 4)
        results["degrees of freedom"] = np.around(df, 4)
        results["p-value"] = np.around(p_value, 4)
        
        results["Lower Central CI's CLd"] = np.around(norm.cdf(ci_lower_cohens_d_central) * 100, 4)
        results["Upper Central CI's CLd"] = np.around(norm.cdf(ci_upper_cohens_d_central) *100, 4)
        results["Lower Central CI's CLg"] = np.around(norm.cdf(ci_lower_hedges_g_central) *100, 4)
        results["Upper Central CI's CLg"] = np.around(norm.cdf(ci_upper_hedges_g_central) *100, 4)
        
        results["Lower Non-Central CI's CLd"] = np.around(norm.cdf(ci_lower_cohens_d_NCP) *100, 4)
        results["Upper Non-Central CI's CLd"] = np.around(norm.cdf(ci_upper_cohens_d_NCP) *100, 4)
        results["Lower Non-Central CI's CLg"] = np.around(norm.cdf(ci_lower_hedges_g_NCP) *100, 4)
        results["Upper Non-Central CI's CLg"] = np.around(norm.cdf(ci_upper_hedges_g_NCP) *100, 4)

        results["Lower Pivotal CI's CLd"] = np.around(norm.cdf(ci_lower_cohens_d_pivotal) *100, 4)
        results["Upper Pivotal CI's CLd"] = np.around(norm.cdf(ci_upper_cohens_d_pivotal) *100, 4)
        results["Lower Pivotal CI's CLg"] = np.around(norm.cdf(ci_lower_hedges_g_pivotal*correction) *100, 4)
        results["Upper Pivotal CI's CLg"] = np.around(norm.cdf(ci_upper_hedges_g_pivotal*correction) *100, 4)
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line CLd"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLd = {:.3f}, {}% CI(pivotal) [{:.3f}, {:.3f}]".format(int(df), t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cles_d,confidence_level_percentages,  np.around(norm.cdf(ci_lower_cohens_d_pivotal) *100, 3),  np.around(norm.cdf(ci_upper_cohens_d_pivotal) *100, 3))
        results["Statistical Line CLg"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLg = {:.3f}, {}% CI(pivotal) [{:.3f}, {:.3f}]".format(int(df), t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cles_g,confidence_level_percentages,  np.around(norm.cdf(ci_lower_hedges_g_pivotal*correction) *100, 3), np.around(norm.cdf(ci_upper_hedges_g_pivotal*correction) *100, 3))

        return results
    
    @staticmethod
    def one_sample_from_data(params: dict) -> dict:
        
        # Set params
        column_1 = params["column 1"]
        population_mean = params["Population's Mean"] #Default should be 0 if not mentioned
        confidence_level_percentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentages/100
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
        ci_lower_cohens_d_central, ci_upper_cohens_d_central, standrat_error_cohens_d =  calculate_central_ci_from_cohens_d_one_sample_t_test (cohens_d, sample_size, confidence_level)
        ci_lower_hedges_g_central, ci_upper_hedges_g_central, standrat_error_hedges_g =  calculate_central_ci_from_cohens_d_one_sample_t_test (hedges_g, sample_size, confidence_level)
        ci_lower_cohens_d_pivotal, ci_upper_cohens_d_pivotal =  pivotal_ci_t (t_score, df, sample_size, confidence_level)
        ci_lower_hedges_g_pivotal, ci_upper_hedges_g_pivotal =  pivotal_ci_t (t_score, df, sample_size, confidence_level)
        ci_lower_cohens_d_NCP, ci_upper_cohens_d_NCP =  CI_NCP_one_Sample (cohens_d, sample_size, confidence_level)
        ci_lower_hedges_g_NCP, ci_upper_hedges_g_NCP =  CI_NCP_one_Sample (hedges_g, sample_size, confidence_level)
        cles_d = norm.cdf(cohens_d) * 100
        cles_g = norm.cdf(hedges_g) * 100      

        # Non Parametric Common Language Effect Sizes
        # Consider Adding Common language effect size to test the probability of a sample value to be larger than the median in the population...(as in matlab mes package)  

        results = {}

        results["Mcgraw & Wong, Common Language Effect Size (CLd)"] = np.around(cles_d, 4)
        results["Mcgraw & Wong, Unbiased Common Language Effect Size (CLg)"] = np.around(cles_g, 4)
        results["t-score"] = np.around(t_score, 4)
        results["degrees of freedom"] = np.around(df, 4)
        results["p-value"] = np.around(p_value, 4)

        results["Lower Central CI's CLd"] = np.around(norm.cdf(ci_lower_cohens_d_central) * 100, 4)
        results["Upper Central CI's CLd"] = np.around(norm.cdf(ci_upper_cohens_d_central) *100, 4)
        results["Lower Central CI's CLg"] = np.around(norm.cdf(ci_lower_hedges_g_central) *100, 4)
        results["Upper Central CI's CLg"] = np.around(norm.cdf(ci_upper_hedges_g_central) *100, 4)
        
        results["Lower Non-Central CI's CLd"] = np.around(norm.cdf(ci_lower_cohens_d_NCP) *100, 4)
        results["Upper Non-Central CI's CLd"] = np.around(norm.cdf(ci_upper_cohens_d_NCP) *100, 4)
        results["Lower Non-Central CI's CLg"] = np.around(norm.cdf(ci_lower_hedges_g_NCP) *100, 4)
        results["Upper Non-Central CI's CLg"] = np.around(norm.cdf(ci_upper_hedges_g_NCP) *100, 4)

        results["Lower Pivotal CI's CLd"] = np.around(norm.cdf(ci_lower_cohens_d_pivotal) *100, 4)
        results["Upper Pivotal CI's CLd"] = np.around(norm.cdf(ci_upper_cohens_d_pivotal) *100, 4)
        results["Lower Pivotal CI's CLg"] = np.around(norm.cdf(ci_lower_hedges_g_pivotal*correction) *100, 4)
        results["Upper Pivotal CI's CLg"] = np.around(norm.cdf(ci_upper_hedges_g_pivotal*correction) *100, 4)
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line CLd"] = "\033[3mt\033[0m({}) = {:.3f}, {} {}, CLd = {:.3f}, {}% CI(pivotal) [{:.3f}, {:.3f}]".format(int(df), t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cles_d,confidence_level_percentages,  np.around(norm.cdf(ci_lower_cohens_d_pivotal) *100, 3),  np.around(norm.cdf(ci_upper_cohens_d_pivotal) *100, 3))
        results["Statistical Line CLg"] = "\033[3mt\033[0m({}) = {:.3f}, {} {}, CLg = {:.3f}, {}% CI(pivotal) [{:.3f}, {:.3f}]".format(int(df), t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cles_g,confidence_level_percentages,  np.around(norm.cdf(ci_lower_hedges_g_pivotal*correction) *100, 3), np.around(norm.cdf(ci_upper_hedges_g_pivotal*correction) *100, 3))

        return results


    @staticmethod    
    def Robust_One_Sample(params: dict) -> dict:
        
        # Set Parameters
        column_1 = params["column 1"]
        trimming_level = params["Trimming Level"] #The default should be 0.2
        Population_Mean = params["Population Mean"] #The default should be 0.2
        reps = params["Number of Bootstrap Samples"]
        confidence_level_percentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentages/100
        sample_size = len(column_1)
        difference = np.array(column_1) - Population_Mean
        correction = np.sqrt(area_under_function(density, norm.ppf(trimming_level), norm.ppf(1 - trimming_level))+ 2 * (norm.ppf(trimming_level) ** 2) * trimming_level)
        trimmed_mean_1 = trim_mean(column_1, trimming_level)
        Winsorized_Standard_Deviation_1 = np.sqrt(WinsorizedVariance(column_1))
        
        # Algina, Penfield, Kesselman robust effect size (AKP)
        Standardizer = np.sqrt(WinsorizedVariance(difference, trimming_level))
        trimmed_mean = trim_mean(difference, trimming_level)
        akp_effect_size = correction * (trimmed_mean - Population_Mean) / Standardizer

        # Confidence Intervals for AKP effect size using Bootstrapping
        Bootstrap_difference = []
        for _ in range(reps):
            # Generate bootstrap samples
            difference_bootstrap = np.random.choice(difference, len(difference), replace=True)
            Bootstrap_difference.append(difference_bootstrap)

        Trimmed_means_of_Bootstrap = (trim_mean(Bootstrap_difference, trimming_level, axis=1))
        Standardizers_of_Bootstrap = np.sqrt([WinsorizedVariance(array, trimming_level) for array in Bootstrap_difference])
        AKP_effect_size_Bootstrap = (correction * (Trimmed_means_of_Bootstrap - Population_Mean) / Standardizers_of_Bootstrap)
        lower_ci_akp_boot = np.percentile(AKP_effect_size_Bootstrap, ((1 - confidence_level) - ((1 - confidence_level)/2)) * 100)
        upper_ci_akp_boot = np.percentile(AKP_effect_size_Bootstrap, ((confidence_level) + ((1 - confidence_level)/2)) * 100)

        # Yuen Test Statistics 
        non_winsorized_sample_size = len(column_1) - 2 * np.floor(trimming_level * len(column_1))
        df = non_winsorized_sample_size - 1
        Yuen_Standrd_Error = Winsorized_Standard_Deviation_1 / ((1 - 2 * trimming_level) * np.sqrt(len(column_1)))    
        difference_trimmed_means = trimmed_mean_1 - Population_Mean
        Yuen_Statistic = difference_trimmed_means / Yuen_Standrd_Error
        Yuen_p_value = 2 * (1 - t.cdf(np.abs(Yuen_Statistic), df))
        
        # Set results
        results = {}
        
        results["Robust Effect Size AKP"] = round(akp_effect_size, 4)
        results["Lower Confidence Interval Robust AKP"] = round(lower_ci_akp_boot, 4)
        results["Upper Confidence Interval Robust AKP"] = round(upper_ci_akp_boot, 4)
        
        # Descreptive Statistics
        results["Trimmed Mean 1"] = round(trimmed_mean_1, 4)
        results["Winsorized Standard Deviation 1"] = round(Winsorized_Standard_Deviation_1, 4)
        
        #Inferntial Statistic Table
        results["Yuen's T statistic"] = round(Yuen_Statistic, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["p-value"] = np.around(Yuen_p_value, 4)
        results["Difference Between Means"] = round(difference_trimmed_means, 4)
        results["Standard Error"] = round(Yuen_Standrd_Error, 4)

        formatted_p_value = "{:.3f}".format(Yuen_p_value).lstrip('0') if Yuen_p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Robust AKP Effect Size"] = "Yuen's \033[3mt\033[0m({}) = {:.3f}, {}{}, \033[3mAKP\033[0m = {:.3f}, {}% CI(bootstrap) [{:.3f}, {:.3f}]".format(int(df), Yuen_Statistic, '\033[3mp = \033[0m' if Yuen_p_value >= 0.001 else '', formatted_p_value, akp_effect_size, confidence_level_percentages, lower_ci_akp_boot, upper_ci_akp_boot)


        return results
    
    # Things to consider
    # 1. Consider adding a robuts one sample measures here
    # 2. Consider adding a the z value tranformed option
    # 3. Consider adding the mes matlab package for one sample CLES


