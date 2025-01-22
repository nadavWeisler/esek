
##################
# CI Constructor #
##################

from scipy.stats import norm, nct, t, gmean
import numpy as np
import math 
import rpy2.robjects as robjects
#robjects.r('install.packages("sadists")')
robjects.r('library(sadists)')
qlambdap = robjects.r['qlambdap']


# Relevant Functions for CI Construction for the Cohen's d family

# 1. Cohen's d One sample/Paired samples Z-test
def calculate_central_ci_from_cohens_d_one_sample(cohens_d, sample_size, confidence_level): # This is a function that calculates the Confidence Intervals of the Effect size in One Sample Z_score test (or two dependent samples)
    standard_error_es = np.sqrt((1 / sample_size) + ((cohens_d**2 / (2 * sample_size)))) #Note that since the effect size in the population and its standart deviation are unknown we can estimate it based on the sample. For the one sample case we will use the Hedges and Olkin 1985 Formula to estimate the standart deviation of the effect size
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = cohens_d - standard_error_es * z_critical_value,  cohens_d + standard_error_es * z_critical_value
    return ci_lower, ci_upper, standard_error_es


# 2. Confidence Intervals for One Sample t-test 

# 2.1 Pivotal CI's
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
    

def NCT_ci_t(t_Score, df, sample_size, confidence_level):
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
        lower_ci = lower_criterion[1]
    
    upper_ci = 0.0
    diff_upper = 1
    while diff_upper > 0.00001:
        if nct.cdf(t_Score, df, upper_criterion[1]) < lower_limit:
            upper_criterion = [upper_criterion[0], (upper_criterion[0] + upper_criterion[1]) / 2, upper_criterion[1]]
        else:
            upper_criterion = [upper_criterion[1], (upper_criterion[1] + upper_criterion[2]) / 2, upper_criterion[2]]
        diff_upper = abs(nct.cdf(t_Score, df, upper_criterion[1]) - lower_limit)
        upper_ci = upper_criterion[1]
    if is_negative:
        return -upper_ci, -lower_ci
    else:
        return lower_ci, upper_ci
    
# 2.2 Central CI's Fnctions    

# 2.2.1 Central CI One Sample t-test (or paired samples)
def calculate_central_ci_paired_samples_t_test(effect_size, sample_size, confidence_level): # This is a function that calculates the Confidence Intervals of the Effect size in One Sample t_score test (or two dependent samples)
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

# 2.2.2 This function calculates Standard Errors for paired sample design based on Goulet-Pelletier and Cousineau
def calculate_SE_pooled_paired_samples_t_test(effect_size, sample_size, correlation, confidence_level): 
    df = sample_size - 1 # This is the Degrees of Freedom for one sample t-test
    correction_factor =  math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
    A = sample_size / (2*(1-correlation))
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



# 2.3 Non Central Parameter CI's (see Cousineau)
def CI_NCP_one_Sample(Effect_Size, sample_size, confidence_level):
    NCP_value = Effect_Size * math.sqrt(sample_size)
    CI_NCP_low = (nct.ppf(1/2 - confidence_level/2, (sample_size - 1), loc=0, scale=1, nc=NCP_value)) / NCP_value * Effect_Size
    CI_NCP_High = (nct.ppf(1/2 + confidence_level/2, (sample_size - 1), loc=0, scale=1, nc=NCP_value)) / NCP_value * Effect_Size
    return CI_NCP_low, CI_NCP_High

# 2.4 More dav confidence intervals Functions

# 2.4.1 lambda_prime
def CI_adjusted_lambda_prime_Paired_Samples(Effect_Size, Standard_Devisation_1, Standard_Deviation_2, sample_size, correlation,  confidence_level):
    Corrected_Correlation = correlation * (gmean([Standard_Devisation_1**2, Standard_Deviation_2**2]) / np.mean((Standard_Devisation_1**2, Standard_Deviation_2**2)))
    df = sample_size - 1
    df_corrected =  2 / (1+correlation**2) * df
    correction1 = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
    correction2 = math.exp(math.lgamma(df_corrected/2) - math.log(math.sqrt(df_corrected/2)) - math.lgamma((df_corrected-1)/2))
    Lambda = float(Effect_Size  * correction1 *np.sqrt(sample_size / (2 * (1 - Corrected_Correlation)))) 
    Lower_CI_adjusted_lambda = qlambdap(1/2 - confidence_level / 2, df= (2/(1+correlation**2)*(sample_size-1)), t=Lambda) / (2 * (1 - Corrected_Correlation)) / correction2 # type: ignore
    Upper_CI_adjusted_lambda = qlambdap(1/2 + confidence_level / 2, df= (2/(1+correlation**2)*(sample_size-1)), t=Lambda) / (2 * (1 - Corrected_Correlation)) / correction2 # type: ignore
    return Lower_CI_adjusted_lambda, Upper_CI_adjusted_lambda

# 2.4.2. MAG CI's (Combination of Morris (2000), Algina and Kesselman 2003), and Goulet-Pelletier & Cousineau (2018))
def CI_MAG_Paired_Samples(Effect_Size, Standard_Devisation_1, Standard_Deviation_2, sample_size, correlation,  confidence_level):
    Corrected_Correlation = correlation * (gmean([Standard_Devisation_1**2, Standard_Deviation_2**2]) / np.mean((Standard_Devisation_1**2, Standard_Deviation_2**2)))
    df = sample_size - 1
    correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
    Lambda = float(Effect_Size  * correction**2 *np.sqrt(sample_size / (2 * (1 - Corrected_Correlation)))) 
    Lower_CI_adjusted_MAG = nct.ppf(1/2 - confidence_level / 2, df= df, nc=Lambda) / np.sqrt(sample_size / (2 * (1 - Corrected_Correlation)))
    Upper_CI_adjusted_MAG = nct.ppf(1/2 + confidence_level / 2, df= df, nc=Lambda) / np.sqrt(sample_size / (2 * (1 - Corrected_Correlation)))
    return Lower_CI_adjusted_MAG, Upper_CI_adjusted_MAG

# 2.4.3 Morris (2000)
def CI_Morris_Paired_Samples(Effect_Size, sample_size, correlation,  confidence_level):
    df = sample_size - 1
    correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
    Cohens_d_Variance_corrected = ((df / (df-2)) * 2 * (1- correlation) / sample_size * (1 + Effect_Size**2 * sample_size / (2 *(1-correlation))) - Effect_Size**2 / correction**2) * correction**2
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower_Morris, ci_upper_Morris = Effect_Size - np.sqrt(Cohens_d_Variance_corrected) * z_critical_value,  Effect_Size + np.sqrt(Cohens_d_Variance_corrected) * z_critical_value
    return ci_lower_Morris, ci_upper_Morris

# 2.4.4. t prime CI's - Goulet-Pelletier & Cousineau (2021)
def CI_t_prime_Paired_Samples(Effect_Size, Standard_Devisation_1, Standard_Deviation_2, sample_size, correlation,  confidence_level):
    Corrected_Correlation = correlation * (gmean([Standard_Devisation_1**2, Standard_Deviation_2**2]) / np.mean((Standard_Devisation_1**2, Standard_Deviation_2**2)))
    df = sample_size - 1
    df_corrected =  2 / (1+correlation**2) * df
    correction = math.exp(math.lgamma(df_corrected/2) - math.log(math.sqrt(df_corrected/2)) - math.lgamma((df_corrected-1)/2))
    Lambda = float(Effect_Size  * correction *np.sqrt(sample_size / (2 * (1 - Corrected_Correlation)))) 
    Lower_CI_adjusted_lambda = qlambdap(1/2 - confidence_level / 2, df= (2/(1+correlation**2)*(sample_size-1)), t=Lambda) / np.sqrt(sample_size / (2 * (1 - Corrected_Correlation))) # type: ignore
    Upper_CI_adjusted_lambda = qlambdap(1/2 + confidence_level / 2, df= (2/(1+correlation**2)*(sample_size-1)), t=Lambda) / np.sqrt(sample_size / (2 * (1 - Corrected_Correlation))) # type: ignore
    return Lower_CI_adjusted_lambda, Upper_CI_adjusted_lambda

# 2.4.5 Algina & Kesselman, 2003
def CI_t_Algina_Keselman(Effect_Size, Standard_Devisation_1, Standard_Deviation_2, sample_size, correlation,  confidence_level):
    df = sample_size - 1
    Corrected_Correlation = correlation * (gmean([Standard_Devisation_1**2, Standard_Deviation_2**2]) / np.mean((Standard_Devisation_1**2, Standard_Deviation_2**2)))
    Constant =  np.sqrt(sample_size / (2*(1-Corrected_Correlation)))
    lower_CI_NCT, upper_CI_NCT = NCT_ci_t(Effect_Size * Constant, df, sample_size, confidence_level)
    lower_CI_Algina_Keselman, upper_CI_Algina_Keselman = lower_CI_NCT / np.sqrt(sample_size/(2*(1-Corrected_Correlation))), upper_CI_NCT / np.sqrt(sample_size/(2*(1-Corrected_Correlation)))
    return lower_CI_Algina_Keselman, upper_CI_Algina_Keselman
 
 
# 3. Independent Samples Z-test
def calculate_central_ci_from_cohens_d_two_samples(cohens_d, sample_size_1, sample_size_2, confidence_level):
    standard_error_es = np.sqrt(((sample_size_1 + sample_size_2)/(sample_size_1 * sample_size_2)) + ((cohens_d ** 2 / (2 * (sample_size_1 + sample_size_2)))))
    z_critical_value = norm.ppf(confidence_level + ((1-confidence_level)/2))
    ci_lower = cohens_d - standard_error_es * z_critical_value
    ci_upper = cohens_d + standard_error_es * z_critical_value
    return ci_lower, ci_upper, standard_error_es


# 4. Independent Samples t-test
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
    Standard_error_effect_size_MLE = np.sqrt((2 / harmonic_sample_size) * ((df+2)/df) * (1 + (effect_size**2 * A / (2*df))))
    Standard_error_effect_size_Large_N = np.sqrt((2/harmonic_sample_size) * (1 + effect_size**2/8))
    Standard_error_effect_size_Small_N = np.sqrt(((df+1) / (df-1)) * (2/harmonic_sample_size) * (1 + effect_size**2/8))
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = effect_size - Standard_error_effect_size_True * z_critical_value,  effect_size + Standard_error_effect_size_True * z_critical_value
    return ci_lower, ci_upper, Standard_error_effect_size_True, Standard_error_effect_size_Morris, Standard_error_effect_size_Hedges, Standard_error_effect_size_Hedges_Olkin, Standard_error_effect_size_MLE, Standard_error_effect_size_Large_N, Standard_error_effect_size_Small_N



########################
########################
#                      #
# 1. Cohen's d Family  #
#                      #
########################
########################

class CI_Constructor_Cohens_D():
    
    ##########################################
    ## 1.1 Cohen's d CI One Sample Z-test ####
    ##########################################

    @staticmethod
    def Cohens_d_one_sample_z_test(params: dict) -> dict:
        
        Cohensd = params["Cohensd"]
        sample_size = params["Sample Size"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100
        ci_lower, ci_upper, se = calculate_central_ci_from_cohens_d_one_sample(Cohensd, sample_size, confidence_level)


        results = {}
        results["Central Confidence Interval"] = np.array([round(ci_lower, 4), round(ci_upper, 4)])

        return results



    ##############################################
    ## 1.2 Cohen's d CI Paired Samples Z-test ####
    ##############################################

    @staticmethod
    def Cohens_d_Paired_samples_z_test(params: dict) -> dict:
        
        Cohensd = params["Cohensd"]
        sample_size = params["Number of Pairs"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100
        ci_lower, ci_upper, se = calculate_central_ci_from_cohens_d_one_sample(Cohensd, sample_size, confidence_level)


        results = {}
        results["Central Confidence Interval"] = np.array([round(ci_lower, 4), round(ci_upper, 4)])

        return results


    ###############################################
    ## 1.3 Cohen's d Indpednent Samples Z-test ####
    ###############################################

    @staticmethod
    def Cohens_d_independent_samples_z_test(params: dict) -> dict:
        
        Cohensd = params["Cohensd"]
        sample_size_1 = params["Sample Size 1"]
        sample_size_2 = params["Sample Size 2"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100

        ci_lower, ci_upper, se = calculate_central_ci_from_cohens_d_two_samples(Cohensd, sample_size_1, sample_size_2, confidence_level)

        results = {}
        results["Central Confidence Interval"] = np.array([round(ci_lower, 4), round(ci_upper, 4)])

        return results



    ##########################################
    ## 1.4 Cohen's d CI One Sample t-test ####
    ##########################################
  
    @staticmethod
    def Cohens_d_one_sample_t_test(params: dict) -> dict:
        
        Cohensd = params["Cohensd"]
        sample_size = params["Sample Size"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100
        ci_lower, ci_upper, Standard_error_effect_size_True, Standard_error_effect_size_Morris, Standard_error_effect_size_Hedges, \
        Standard_error_effect_size_Hedges_Olkin, Standard_error_effect_size_MLE, \
        Standard_error_effect_size_Large_N, Standard_error_effect_size_Small_N = calculate_central_ci_paired_samples_t_test(Cohensd, sample_size, confidence_level)
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
        t_score = Cohensd * np.sqrt(sample_size)
        lower_pivotal, upper_pivotal = Pivotal_ci_t(t_score, df = sample_size-1, sample_size = sample_size, confidence_level = confidence_level)
        ci_lower_cohens_d_NCP, ci_upper_cohens_d_NCP =  CI_NCP_one_Sample (Cohensd, sample_size, confidence_level)

        confidence_intervals_pivotal = np.array([lower_pivotal, upper_pivotal])
        confidence_intervals_true = np.array([ci_lower, ci_upper])
        confidence_intervals_Morris = np.array([ Cohensd - zcrit * Standard_error_effect_size_Morris, Cohensd + zcrit * Standard_error_effect_size_Morris])
        confidence_intervals_Hedges = np.array([ Cohensd - zcrit * Standard_error_effect_size_Hedges, Cohensd + zcrit * Standard_error_effect_size_Hedges])
        confidence_intervals_Hedges_Olkin = np.array([ Cohensd - zcrit * Standard_error_effect_size_Hedges_Olkin, Cohensd + zcrit * Standard_error_effect_size_Hedges_Olkin])
        confidence_intervals_MLE = np.array([ Cohensd - zcrit * Standard_error_effect_size_MLE, Cohensd + zcrit * Standard_error_effect_size_MLE])
        confidence_intervals_Large_N = np.array([ Cohensd - zcrit * Standard_error_effect_size_Large_N, Cohensd + zcrit * Standard_error_effect_size_Large_N])
        confidence_intervals_Small_N = np.array([ Cohensd - zcrit * Standard_error_effect_size_Small_N, Cohensd + zcrit * Standard_error_effect_size_Small_N])

        # Hedges g
        df = sample_size - 1
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
        hedges_g = correction*Cohensd
        ci_lower_hedges_g_central, ci_upper_hedges_g_central, Standard_error_hedges_g_true, Standard_error_hedges_g_morris, Standard_error_hedges_g_hedges, \
        Standard_error_hedges_g_hedges_olkin, Standard_error_hedges_g_MLE, Standard_error_hedges_g_Largen, Standard_error_hedges_g_Small_n =  calculate_central_ci_paired_samples_t_test (hedges_g, sample_size, confidence_level)
        ci_lower_hedges_g_NCP, ci_upper_hedges_g_NCP =  CI_NCP_one_Sample (hedges_g, sample_size, confidence_level)

        confidence_intervals_pivotal_hedges_g = np.array([lower_pivotal * correction, upper_pivotal * correction])
        confidence_intervals_true_hedges_g = np.array([ci_lower_hedges_g_central, ci_upper_hedges_g_central])
        confidence_intervals_Morris_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_morris, hedges_g + zcrit * Standard_error_hedges_g_morris])
        confidence_intervals_Hedges_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_hedges, hedges_g + zcrit * Standard_error_hedges_g_hedges])
        confidence_intervals_Hedges_Olkin_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_hedges_olkin, hedges_g + zcrit * Standard_error_hedges_g_hedges_olkin])
        confidence_intervals_MLE_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_MLE, hedges_g + zcrit * Standard_error_hedges_g_MLE])
        confidence_intervals_Large_N_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_Largen, hedges_g + zcrit * Standard_error_hedges_g_Largen])
        confidence_intervals_Small_N_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_Small_n, hedges_g + zcrit * Standard_error_hedges_g_Small_n])


        results = {}
        results["Cohen's d Pivotal Confidence Interval"] = confidence_intervals_pivotal
        results["Cohen's d  Non_Central Pivotal Confidence Interval"] = np.array([round(ci_lower_cohens_d_NCP, 4), round(ci_upper_cohens_d_NCP, 4)])
        results["Cohen's d Central Confidence Interval (True)"] = confidence_intervals_true
        results["Cohen's d Central Confidence Interval (Morris)"] = confidence_intervals_Morris
        results["Cohen's d Central Confidence Interval (Hedges)"] = confidence_intervals_Hedges
        results["Cohen's d Central Confidence Interval (Hedges-Olkin)"] = confidence_intervals_Hedges_Olkin
        results["Cohen's d Central Confidence Interval (MLE)"] = confidence_intervals_MLE
        results["Cohen's d Central Confidence Interval (Large N)"] = confidence_intervals_Large_N
        results["Cohen's d Central Confidence Interval (Small N)"] = confidence_intervals_Small_N

        results["Hedges' g Pivotal Confidence Interval"] = confidence_intervals_pivotal_hedges_g
        results["Hedges' g  Non_Central Pivotal Confidence Interval"] = np.array([round(ci_lower_hedges_g_NCP, 4), round(ci_upper_hedges_g_NCP, 4)])
        results["Hedges' g Central Confidence Interval (True)"] = confidence_intervals_true_hedges_g
        results["Hedges' g Central Confidence Interval (Morris)"] = confidence_intervals_Morris_hedges_g
        results["Hedges' g Central Confidence Interval (Hedges)"] = confidence_intervals_Hedges_hedges_g
        results["Hedges' g Central Confidence Interval (Hedges-Olkin)"] = confidence_intervals_Hedges_Olkin_hedges_g
        results["Hedges' g Central Confidence Interval (MLE)"] = confidence_intervals_MLE_hedges_g
        results["Hedges' g Central Confidence Interval (Large N)"] = confidence_intervals_Large_N_hedges_g
        results["Hedges' g Central Confidence Interval (Small N)"] = confidence_intervals_Small_N_hedges_g

        return results



    #########################################
    ## 1.5 Hedges' g CI One Sample t-test ###
    #########################################
  
    @staticmethod
    def hedges_g_one_sample(params: dict) -> dict:        
        Hedges_g = params["Hedges g"]
        sample_size = params["Sample Size"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100
        ci_lower, ci_upper, Standard_error_effect_size_True, Standard_error_effect_size_Morris, Standard_error_effect_size_Hedges, \
        Standard_error_effect_size_Hedges_Olkin, Standard_error_effect_size_MLE, \
        Standard_error_effect_size_Large_N, Standard_error_effect_size_Small_N = calculate_central_ci_paired_samples_t_test(Hedges_g, sample_size, confidence_level)
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
        df = sample_size - 1
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))

        t_score = Hedges_g * np.sqrt(sample_size) / correction
        lower_pivotal, upper_pivotal = Pivotal_ci_t(t_score, df = sample_size-1, sample_size = sample_size, confidence_level = confidence_level)
        ci_lower_cohens_d_NCP, ci_upper_cohens_d_NCP =  CI_NCP_one_Sample (Hedges_g, sample_size, confidence_level)

        confidence_intervals_pivotal = np.array([lower_pivotal * correction , upper_pivotal * correction])
        confidence_intervals_true = np.array([ci_lower, ci_upper])
        confidence_intervals_Morris = np.array([ Hedges_g - zcrit * Standard_error_effect_size_Morris, Hedges_g + zcrit * Standard_error_effect_size_Morris])
        confidence_intervals_Hedges = np.array([ Hedges_g - zcrit * Standard_error_effect_size_Hedges, Hedges_g + zcrit * Standard_error_effect_size_Hedges])
        confidence_intervals_Hedges_Olkin = np.array([ Hedges_g - zcrit * Standard_error_effect_size_Hedges_Olkin, Hedges_g + zcrit * Standard_error_effect_size_Hedges_Olkin])
        confidence_intervals_MLE = np.array([ Hedges_g - zcrit * Standard_error_effect_size_MLE, Hedges_g + zcrit * Standard_error_effect_size_MLE])
        confidence_intervals_Large_N = np.array([ Hedges_g - zcrit * Standard_error_effect_size_Large_N, Hedges_g + zcrit * Standard_error_effect_size_Large_N])
        confidence_intervals_Small_N = np.array([ Hedges_g - zcrit * Standard_error_effect_size_Small_N, Hedges_g + zcrit * Standard_error_effect_size_Small_N])

        results = {}
        results["Hedges' g Pivotal Confidence Interval"] = confidence_intervals_pivotal
        results["Hedges' g Non_Central Pivotal Confidence Interval"] = np.array([round(ci_lower_cohens_d_NCP, 4), round(ci_upper_cohens_d_NCP, 4)])
        results["Hedges' g Central Confidence Interval (True)"] = confidence_intervals_true
        results["Hedges' g Central Confidence Interval (Morris)"] = confidence_intervals_Morris
        results["Hedges' g Central Confidence Interval (Hedges)"] = confidence_intervals_Hedges
        results["Hedges' g Central Confidence Interval (Hedges-Olkin)"] = confidence_intervals_Hedges_Olkin
        results["Hedges' g Central Confidence Interval (MLE)"] = confidence_intervals_MLE
        results["Hedges' g Central Confidence Interval (Large N)"] = confidence_intervals_Large_N
        results["Hedges' g Central Confidence Interval (Small N)"] = confidence_intervals_Small_N

        return results
    



    ########################################
    ## 1.6 Cohens dz CI Paired Samples #####
    ########################################

    @staticmethod
    def Cohens_d_Paired_t_test_dz(params: dict) -> dict:
        
        Cohensd = params["Cohensd"]
        sample_size = params["Number of Pairs"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100
        ci_lower, ci_upper, Standard_error_effect_size_True, Standard_error_effect_size_Morris, Standard_error_effect_size_Hedges, \
        Standard_error_effect_size_Hedges_Olkin, Standard_error_effect_size_MLE, \
        Standard_error_effect_size_Large_N, Standard_error_effect_size_Small_N = calculate_central_ci_paired_samples_t_test(Cohensd, sample_size, confidence_level)
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
        t_score = Cohensd * np.sqrt(sample_size)
        lower_pivotal, upper_pivotal = Pivotal_ci_t(t_score, df = sample_size-1, sample_size = sample_size, confidence_level = confidence_level)
        ci_lower_cohens_d_NCP, ci_upper_cohens_d_NCP =  CI_NCP_one_Sample (Cohensd, sample_size, confidence_level)

        confidence_intervals_pivotal = np.array([lower_pivotal, upper_pivotal])
        confidence_intervals_true = np.array([ci_lower, ci_upper])
        confidence_intervals_Morris = np.array([ Cohensd - zcrit * Standard_error_effect_size_Morris, Cohensd + zcrit * Standard_error_effect_size_Morris])
        confidence_intervals_Hedges = np.array([ Cohensd - zcrit * Standard_error_effect_size_Hedges, Cohensd + zcrit * Standard_error_effect_size_Hedges])
        confidence_intervals_Hedges_Olkin = np.array([ Cohensd - zcrit * Standard_error_effect_size_Hedges_Olkin, Cohensd + zcrit * Standard_error_effect_size_Hedges_Olkin])
        confidence_intervals_MLE = np.array([ Cohensd - zcrit * Standard_error_effect_size_MLE, Cohensd + zcrit * Standard_error_effect_size_MLE])
        confidence_intervals_Large_N = np.array([ Cohensd - zcrit * Standard_error_effect_size_Large_N, Cohensd + zcrit * Standard_error_effect_size_Large_N])
        confidence_intervals_Small_N = np.array([ Cohensd - zcrit * Standard_error_effect_size_Small_N, Cohensd + zcrit * Standard_error_effect_size_Small_N])

        # Hedges g
        df = sample_size - 1
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
        hedges_g = correction*Cohensd
        ci_lower_hedges_g_central, ci_upper_hedges_g_central, Standard_error_hedges_g_true, Standard_error_hedges_g_morris, Standard_error_hedges_g_hedges, \
        Standard_error_hedges_g_hedges_olkin, Standard_error_hedges_g_MLE, Standard_error_hedges_g_Largen, Standard_error_hedges_g_Small_n =  calculate_central_ci_paired_samples_t_test (hedges_g, sample_size, confidence_level)
        ci_lower_hedges_g_NCP, ci_upper_hedges_g_NCP =  CI_NCP_one_Sample (hedges_g, sample_size, confidence_level)

        confidence_intervals_pivotal_hedges_g = np.array([lower_pivotal * correction, upper_pivotal * correction])
        confidence_intervals_true_hedges_g = np.array([ci_lower_hedges_g_central, ci_upper_hedges_g_central])
        confidence_intervals_Morris_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_morris, hedges_g + zcrit * Standard_error_hedges_g_morris])
        confidence_intervals_Hedges_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_hedges, hedges_g + zcrit * Standard_error_hedges_g_hedges])
        confidence_intervals_Hedges_Olkin_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_hedges_olkin, hedges_g + zcrit * Standard_error_hedges_g_hedges_olkin])
        confidence_intervals_MLE_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_MLE, hedges_g + zcrit * Standard_error_hedges_g_MLE])
        confidence_intervals_Large_N_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_Largen, hedges_g + zcrit * Standard_error_hedges_g_Largen])
        confidence_intervals_Small_N_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_Small_n, hedges_g + zcrit * Standard_error_hedges_g_Small_n])


        results = {}
        results["Cohen's dz Pivotal Confidence Interval"] = confidence_intervals_pivotal
        results["Cohen's dz Non_Central Pivotal Confidence Interval"] = np.array([round(ci_lower_cohens_d_NCP, 4), round(ci_upper_cohens_d_NCP, 4)])
        results["Cohen's dz Central Confidence Interval (True)"] = confidence_intervals_true
        results["Cohen's dz Central Confidence Interval (Morris)"] = confidence_intervals_Morris
        results["Cohen's dz Central Confidence Interval (Hedges)"] = confidence_intervals_Hedges
        results["Cohen's dz Central Confidence Interval (Hedges-Olkin)"] = confidence_intervals_Hedges_Olkin
        results["Cohen's dz Central Confidence Interval (MLE)"] = confidence_intervals_MLE
        results["Cohen's dz Central Confidence Interval (Large N)"] = confidence_intervals_Large_N
        results["Cohen's dz Central Confidence Interval (Small N)"] = confidence_intervals_Small_N

        results["Hedges' gz Pivotal Confidence Interval"] = confidence_intervals_pivotal_hedges_g
        results["Hedges' gz Cohen's d Non_Central Pivotal Confidence Interval"] = np.array([round(ci_lower_hedges_g_NCP, 4), round(ci_upper_hedges_g_NCP, 4)])
        results["Hedges' gz Central Confidence Interval (True)"] = confidence_intervals_true_hedges_g
        results["Hedges' gz Central Confidence Interval (Morris)"] = confidence_intervals_Morris_hedges_g
        results["Hedges' gz Central Confidence Interval (Hedges)"] = confidence_intervals_Hedges_hedges_g
        results["Hedges' gz Central Confidence Interval (Hedges-Olkin)"] = confidence_intervals_Hedges_Olkin_hedges_g
        results["Hedges' gz Central Confidence Interval (MLE)"] = confidence_intervals_MLE_hedges_g
        results["Hedges' gz Central Confidence Interval (Large N)"] = confidence_intervals_Large_N_hedges_g
        results["Hedges' gz Central Confidence Interval (Small N)"] = confidence_intervals_Small_N_hedges_g

        return results



    ########################################
    ## 1.7 Hedges' gz CI Paired Samples ####
    ########################################

    @staticmethod
    def Cohens_d_Paired_t_test_gz(params: dict) -> dict:
        Hedges_g = params["Hedges g"]
        sample_size = params["Number of Pairs"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100
        ci_lower, ci_upper, Standard_error_effect_size_True, Standard_error_effect_size_Morris, Standard_error_effect_size_Hedges, \
        Standard_error_effect_size_Hedges_Olkin, Standard_error_effect_size_MLE, \
        Standard_error_effect_size_Large_N, Standard_error_effect_size_Small_N = calculate_central_ci_paired_samples_t_test(Hedges_g, sample_size, confidence_level)
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
        df = sample_size - 1
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))

        t_score = Hedges_g * np.sqrt(sample_size) / correction
        lower_pivotal, upper_pivotal = Pivotal_ci_t(t_score, df = sample_size-1, sample_size = sample_size, confidence_level = confidence_level)
        ci_lower_cohens_d_NCP, ci_upper_cohens_d_NCP =  CI_NCP_one_Sample (Hedges_g, sample_size, confidence_level)

        confidence_intervals_pivotal = np.array([lower_pivotal * correction , upper_pivotal * correction])
        confidence_intervals_true = np.array([ci_lower, ci_upper])
        confidence_intervals_Morris = np.array([ Hedges_g - zcrit * Standard_error_effect_size_Morris, Hedges_g + zcrit * Standard_error_effect_size_Morris])
        confidence_intervals_Hedges = np.array([ Hedges_g - zcrit * Standard_error_effect_size_Hedges, Hedges_g + zcrit * Standard_error_effect_size_Hedges])
        confidence_intervals_Hedges_Olkin = np.array([ Hedges_g - zcrit * Standard_error_effect_size_Hedges_Olkin, Hedges_g + zcrit * Standard_error_effect_size_Hedges_Olkin])
        confidence_intervals_MLE = np.array([ Hedges_g - zcrit * Standard_error_effect_size_MLE, Hedges_g + zcrit * Standard_error_effect_size_MLE])
        confidence_intervals_Large_N = np.array([ Hedges_g - zcrit * Standard_error_effect_size_Large_N, Hedges_g + zcrit * Standard_error_effect_size_Large_N])
        confidence_intervals_Small_N = np.array([ Hedges_g - zcrit * Standard_error_effect_size_Small_N, Hedges_g + zcrit * Standard_error_effect_size_Small_N])

        results = {}
        results["Hedges' gz Pivotal Confidence Interval"] = confidence_intervals_pivotal
        results["Hedges' gz Non_Central Pivotal Confidence Interval"] = np.array([round(ci_lower_cohens_d_NCP, 4), round(ci_upper_cohens_d_NCP, 4)])
        results["Hedges' gz Central Confidence Interval (True)"] = confidence_intervals_true
        results["Hedges' gz Central Confidence Interval (Morris)"] = confidence_intervals_Morris
        results["Hedges' gz Central Confidence Interval (Hedges)"] = confidence_intervals_Hedges
        results["Hedges' gz Central Confidence Interval (Hedges-Olkin)"] = confidence_intervals_Hedges_Olkin
        results["Hedges' gz Central Confidence Interval (MLE)"] = confidence_intervals_MLE
        results["Hedges' gz Central Confidence Interval (Large N)"] = confidence_intervals_Large_N
        results["Hedges' gz Central Confidence Interval (Small N)"] = confidence_intervals_Small_N

        return results





    ########################################
    ## 1.8 Cohens drm CI Paired Samples ####
    ########################################

    @staticmethod
    def Cohens_d_Paired_t_test_drm(params: dict) -> dict:
        
        Cohensd = params["Cohensd"]
        sample_size = params["Number of Pairs"]
        correlation = params["Correlation"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100
        ci_lower, ci_upper, Standard_error_effect_size_True, Standard_error_effect_size_Morris, Standard_error_effect_size_Hedges, \
        Standard_error_effect_size_Hedges_Olkin, Standard_error_effect_size_MLE, \
        Standard_error_effect_size_Large_N, Standard_error_effect_size_Small_N = calculate_central_ci_paired_samples_t_test(Cohensd, sample_size, confidence_level)
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
        tcrit = t.ppf(1 - (1 - confidence_level) / 2, sample_size - 1)
        t_score = (Cohensd* (np.sqrt(2*(1-correlation))) * np.sqrt(sample_size))
        t_drm = t_score * (np.sqrt(2 *(1 - correlation)))
        df = sample_size - 1
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
        hedges_g = Cohensd * correction


        # Confidence Intervals for drm
        lower_pivotal, upper_pivotal = Pivotal_ci_t(t_drm, df = sample_size-1, sample_size = sample_size, confidence_level = confidence_level)
        ci_lower_cohens_d_NCP, ci_upper_cohens_d_NCP =  CI_NCP_one_Sample (Cohensd, sample_size, confidence_level)
        confidence_intervals_pivotal = np.array([lower_pivotal , upper_pivotal])
        confidence_intervals_true = np.array([ci_lower, ci_upper])
        confidence_intervals_Morris = np.array([ Cohensd - zcrit * Standard_error_effect_size_Morris, Cohensd + zcrit * Standard_error_effect_size_Morris])
        confidence_intervals_Hedges = np.array([ Cohensd - zcrit * Standard_error_effect_size_Hedges, Cohensd + zcrit * Standard_error_effect_size_Hedges])
        confidence_intervals_Hedges_Olkin = np.array([ Cohensd - zcrit * Standard_error_effect_size_Hedges_Olkin, Cohensd + zcrit * Standard_error_effect_size_Hedges_Olkin])
        confidence_intervals_MLE = np.array([ Cohensd - zcrit * Standard_error_effect_size_MLE, Cohensd + zcrit * Standard_error_effect_size_MLE])
        confidence_intervals_Large_N = np.array([ Cohensd - zcrit * Standard_error_effect_size_Large_N, Cohensd + zcrit * Standard_error_effect_size_Large_N])
        confidence_intervals_Small_N = np.array([ Cohensd - zcrit * Standard_error_effect_size_Small_N, Cohensd + zcrit * Standard_error_effect_size_Small_N])
        standard_error_becker = np.sqrt( (1/sample_size + Cohensd**2/(2*sample_size))*(2-2*correlation) )
        confidence_intervals_becker= np.array([ Cohensd - tcrit * standard_error_becker, Cohensd + tcrit * standard_error_becker])

        # Confidence Intervals for grm
        lower_pivotal, upper_pivotal = Pivotal_ci_t(t_drm, df = sample_size-1, sample_size = sample_size, confidence_level = confidence_level)
        ci_lower_cohens_d_NCP_hedges_g, ci_upper_cohens_d_NCP_hedges_g =  CI_NCP_one_Sample (hedges_g, sample_size, confidence_level)
        confidence_intervals_pivotal = np.array([lower_pivotal , upper_pivotal])
        confidence_intervals_true_hedges_g = np.array([ci_lower, ci_upper])
        confidence_intervals_Morris_hedges_g = np.array([ hedges_g - zcrit * Standard_error_effect_size_Morris, hedges_g + zcrit * Standard_error_effect_size_Morris])
        confidence_intervals_Hedges_hedges_g = np.array([ hedges_g - zcrit * Standard_error_effect_size_Hedges, hedges_g + zcrit * Standard_error_effect_size_Hedges])
        confidence_intervals_Hedges_Olkin_hedges_g = np.array([ hedges_g - zcrit * Standard_error_effect_size_Hedges_Olkin, hedges_g + zcrit * Standard_error_effect_size_Hedges_Olkin])
        confidence_intervals_MLE_hedges_g = np.array([ hedges_g - zcrit * Standard_error_effect_size_MLE, hedges_g + zcrit * Standard_error_effect_size_MLE])
        confidence_intervals_Large_N_hedges_g = np.array([ hedges_g - zcrit * Standard_error_effect_size_Large_N, hedges_g + zcrit * Standard_error_effect_size_Large_N])
        confidence_intervals_Small_N_hedges_g = np.array([ hedges_g - zcrit * Standard_error_effect_size_Small_N, hedges_g + zcrit * Standard_error_effect_size_Small_N])
        standard_error_becker_hedges_g = np.sqrt( (1/sample_size + hedges_g**2/(2*sample_size))*(2-2*correlation) )
        confidence_intervals_becker_hedges_g = np.array([ hedges_g - tcrit * standard_error_becker, hedges_g + tcrit * standard_error_becker])

        results = {}
        results["Cohen's drm Pivotal Confidence Interval"] = confidence_intervals_pivotal
        results["Cohen's drm Non Central Pivotal Confidence Interval"] = np.array([round(ci_lower_cohens_d_NCP, 4), round(ci_upper_cohens_d_NCP, 4)])
        results["Cohen's drm Central Confidence Interval (True)"] = confidence_intervals_true
        results["Cohen's drm Central Confidence Interval (Morris)"] = confidence_intervals_Morris
        results["Cohen's drm Central Confidence Interval (Hedges)"] = confidence_intervals_Hedges
        results["Cohen's drm Central Confidence Interval (Hedges-Olkin)"] = confidence_intervals_Hedges_Olkin
        results["Cohen's drm Central Confidence Interval (MLE)"] = confidence_intervals_MLE
        results["Cohen's drm Central Confidence Interval (Large N)"] = confidence_intervals_Large_N
        results["Cohen's drm Central Confidence Interval (Small N)"] = confidence_intervals_Small_N
        results["Cohen's drm Central Confidence Interval (Becker 1996)"] = confidence_intervals_becker

        results["Hedges' grm"] = hedges_g
        results["Hedges' grm Pivotal Confidence Interval"] = confidence_intervals_pivotal * correction
        results["Hedges' grm Non Central Pivotal Confidence Interval"] = np.array([round(ci_lower_cohens_d_NCP_hedges_g, 4), round(ci_upper_cohens_d_NCP_hedges_g, 4)])
        results["Hedges' grm Central Confidence Interval (True)"] = confidence_intervals_true_hedges_g
        results["Hedges' grm Central Confidence Interval (Morris)"] = confidence_intervals_Morris_hedges_g 
        results["Hedges' grm Central Confidence Interval (Hedges)"] = confidence_intervals_Hedges_hedges_g 
        results["Hedges' grm Central Confidence Interval (Hedges-Olkin)"] = confidence_intervals_Hedges_Olkin_hedges_g 
        results["Hedges' grm Central Confidence Interval (MLE)"] = confidence_intervals_MLE_hedges_g 
        results["Hedges' grm Central Confidence Interval (Large N)"] = confidence_intervals_Large_N_hedges_g 
        results["Hedges' grm Central Confidence Interval (Small N)"] = confidence_intervals_Small_N_hedges_g 
        results["Hedges' grm Central Confidence Interval (Becker 1996)"] = confidence_intervals_becker_hedges_g 
        return results
    



    ######################################
    ## 1.9 Hedges' grm paired Samples ####
    ######################################

    @staticmethod
    def Hedges_grm_Paired_t_test(params: dict) -> dict:
        
        hedges_g = params["Hedges g"]
        sample_size = params["Number of Pairs"]
        correlation = params["Correlation"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100
        
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
        tcrit = t.ppf(1 - (1 - confidence_level) / 2, sample_size - 1)
        df = sample_size - 1
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))

        Cohensd = hedges_g / correction
        t_score = (Cohensd* (np.sqrt(2*(1-correlation))) * np.sqrt(sample_size))
        t_drm = t_score * (np.sqrt(2 *(1 - correlation)))
        lower_pivotal, upper_pivotal = Pivotal_ci_t(t_drm, df = sample_size-1, sample_size = sample_size, confidence_level = confidence_level)
        standard_error_becker = np.sqrt( (1/sample_size + hedges_g**2/(2*sample_size))*(2-2*correlation) )
        confidence_intervals_becker= np.array([ hedges_g - tcrit * standard_error_becker, hedges_g + tcrit * standard_error_becker])

        # Hedges grm
        df = sample_size - 1
        hedges_g = correction*Cohensd
        ci_lower_hedges_g_central, ci_upper_hedges_g_central, Standard_error_hedges_g_true, Standard_error_hedges_g_morris, Standard_error_hedges_g_hedges, \
        Standard_error_hedges_g_hedges_olkin, Standard_error_hedges_g_MLE, Standard_error_hedges_g_Largen, Standard_error_hedges_g_Small_n =  calculate_central_ci_paired_samples_t_test (hedges_g, sample_size, confidence_level)
        ci_lower_hedges_g_NCP, ci_upper_hedges_g_NCP =  CI_NCP_one_Sample (hedges_g, sample_size, confidence_level)

        confidence_intervals_pivotal_hedges_g = np.array([lower_pivotal, upper_pivotal])
        confidence_intervals_true_hedges_g = np.array([ci_lower_hedges_g_central, ci_upper_hedges_g_central])
        confidence_intervals_Morris_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_morris, hedges_g + zcrit * Standard_error_hedges_g_morris])
        confidence_intervals_Hedges_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_hedges, hedges_g + zcrit * Standard_error_hedges_g_hedges])
        confidence_intervals_Hedges_Olkin_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_hedges_olkin, hedges_g + zcrit * Standard_error_hedges_g_hedges_olkin])
        confidence_intervals_MLE_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_MLE, hedges_g + zcrit * Standard_error_hedges_g_MLE])
        confidence_intervals_Large_N_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_Largen, hedges_g + zcrit * Standard_error_hedges_g_Largen])
        confidence_intervals_Small_N_hedges_g = np.array([ hedges_g - zcrit * Standard_error_hedges_g_Small_n, hedges_g + zcrit * Standard_error_hedges_g_Small_n])

        results = {}

        results["Hedges' grm Pivotal Confidence Interval"] = confidence_intervals_pivotal_hedges_g
        results["Hedges' grm Non_Central Pivotal Confidence Interval"] = np.array([round(ci_lower_hedges_g_NCP, 4), round(ci_upper_hedges_g_NCP, 4)])
        results["Hedges' grm Central Confidence Interval (True)"] = confidence_intervals_true_hedges_g
        results["Hedges' grm Central Confidence Interval (Morris)"] = confidence_intervals_Morris_hedges_g
        results["Hedges' grm Central Confidence Interval (Hedges)"] = confidence_intervals_Hedges_hedges_g
        results["Hedges' grm Central Confidence Interval (Hedges-Olkin)"] = confidence_intervals_Hedges_Olkin_hedges_g
        results["Hedges' grm Central Confidence Interval (MLE)"] = confidence_intervals_MLE_hedges_g
        results["Hedges' grm Central Confidence Interval (Large N)"] = confidence_intervals_Large_N_hedges_g
        results["Hedges' grm Central Confidence Interval (Small N)"] = confidence_intervals_Small_N_hedges_g
        results["Hedges' grm Central Confidence Interval (becker)"] = confidence_intervals_becker * correction


        return results



    #####################################
    ## 1.10 Cohens dav Paired Samples ###
    #####################################

    @staticmethod
    def Cohens_dav_Paired_t_test_dav(params: dict) -> dict:
        
        Cohensdav = params["Cohensd"]
        sample_size = params["Number of Pairs"]
        sd1 = params["Standard Deviation Group 1"]
        sd2 = params["Standard Deviation Group 2"]
        correlation = params["Correlation"]

        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100
        t_score = Cohensdav * np.sqrt(sample_size)
        df = sample_size -1
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2)) # For Hedges gav
        hedges_gav = correction * Cohensdav


        # 1. Central Confidence Intervals 
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
        tcrit = t.ppf(1 - (1 - confidence_level) / 2, df)

        # 1.1 Central Confidence Intervals using the Z-Distribution with A = n-1
        ci_lower, ci_upper, Standard_error_effect_size_True, Standard_error_effect_size_Morris, Standard_error_effect_size_Hedges, \
        Standard_error_effect_size_Hedges_Olkin, Standard_error_effect_size_MLE, \
        Standard_error_effect_size_Large_N, Standard_error_effect_size_Small_N = calculate_central_ci_paired_samples_t_test(Cohensdav, sample_size, confidence_level)
        confidence_intervals_true = np.array([ci_lower, ci_upper])
        confidence_intervals_Morris = np.array([ Cohensdav - zcrit * Standard_error_effect_size_Morris, Cohensdav + zcrit * Standard_error_effect_size_Morris])
        confidence_intervals_Hedges = np.array([ Cohensdav - zcrit * Standard_error_effect_size_Hedges, Cohensdav + zcrit * Standard_error_effect_size_Hedges])
        confidence_intervals_Hedges_Olkin = np.array([ Cohensdav - zcrit * Standard_error_effect_size_Hedges_Olkin, Cohensdav + zcrit * Standard_error_effect_size_Hedges_Olkin])
        confidence_intervals_Hedges_Olkin_tcrit = np.array([ Cohensdav - tcrit * Standard_error_effect_size_Hedges_Olkin, Cohensdav + tcrit * Standard_error_effect_size_Hedges_Olkin])
        confidence_intervals_MLE = np.array([ Cohensdav - zcrit * Standard_error_effect_size_MLE, Cohensdav + zcrit * Standard_error_effect_size_MLE])
        confidence_intervals_Large_N = np.array([ Cohensdav - zcrit * Standard_error_effect_size_Large_N, Cohensdav + zcrit * Standard_error_effect_size_Large_N])
        confidence_intervals_Small_N = np.array([ Cohensdav - zcrit * Standard_error_effect_size_Small_N, Cohensdav + zcrit * Standard_error_effect_size_Small_N])
        standard_error_becker = np.sqrt( (1/sample_size + Cohensdav**2/(2*sample_size))*(2-2*correlation) )
        confidence_intervals_becker_zdist = np.array([ Cohensdav - zcrit * standard_error_becker, Cohensdav + zcrit * standard_error_becker])
        confidence_intervals_becker_tdist = np.array([ Cohensdav - tcrit * standard_error_becker, Cohensdav + tcrit * standard_error_becker])

        # 1.2 Central Confidence Intervals using the Z-Distribution with A = n / (2*(1-correlation)
        ci_lower_pooled, ci_upper_pooled, Standard_error_effect_size_True_pooled, Standard_error_effect_size_Morris_pooled, Standard_error_effect_size_Hedges_pooled, \
        Standard_error_effect_size_Hedges_Olkin_pooled, Standard_error_effect_size_MLE_pooled, \
        Standard_error_effect_size_Large_N_pooled, Standard_error_effect_size_Small_N_pooled = calculate_SE_pooled_paired_samples_t_test(Cohensdav, sample_size, correlation, confidence_level)
        confidence_intervals_true_pooled = np.array([ci_lower, ci_upper])
        confidence_intervals_Morris_pooled = np.array([ Cohensdav - zcrit * Standard_error_effect_size_Morris, Cohensdav + zcrit * Standard_error_effect_size_Morris])
        confidence_intervals_Hedges_pooled = np.array([ Cohensdav - zcrit * Standard_error_effect_size_Hedges, Cohensdav + zcrit * Standard_error_effect_size_Hedges])
        confidence_intervals_Hedges_Olkin_pooled = np.array([ Cohensdav - zcrit * Standard_error_effect_size_Hedges_Olkin, Cohensdav + zcrit * Standard_error_effect_size_Hedges_Olkin])
        confidence_intervals_MLE_pooled = np.array([ Cohensdav - zcrit * Standard_error_effect_size_MLE, Cohensdav + zcrit * Standard_error_effect_size_MLE])
        confidence_intervals_Large_N_pooled = np.array([ Cohensdav - zcrit * Standard_error_effect_size_Large_N, Cohensdav + zcrit * Standard_error_effect_size_Large_N])
        confidence_intervals_Small_N_pooled = np.array([ Cohensdav - zcrit * Standard_error_effect_size_Small_N, Cohensdav + zcrit * Standard_error_effect_size_Small_N])


        # 1.3 Central Confidence Intervals for Hedges' gav using the Z-Distribution with A = n-1
        ci_lower, ci_upper, Standard_error_effect_size_True, Standard_error_effect_size_Morris, Standard_error_effect_size_Hedges, \
        Standard_error_effect_size_Hedges_Olkin, Standard_error_effect_size_MLE, \
        Standard_error_effect_size_Large_N, Standard_error_effect_size_Small_N = calculate_central_ci_paired_samples_t_test(hedges_gav, sample_size, confidence_level)
        confidence_intervals_true_hedges_gav = np.array([ci_lower, ci_upper])
        confidence_intervals_Morris_hedges_gav = np.array([ hedges_gav - zcrit * Standard_error_effect_size_Morris, hedges_gav + zcrit * Standard_error_effect_size_Morris])
        confidence_intervals_Hedges_hedges_gav = np.array([ hedges_gav - zcrit * Standard_error_effect_size_Hedges, hedges_gav + zcrit * Standard_error_effect_size_Hedges])
        confidence_intervals_Hedges_Olkin_hedges_gav = np.array([ hedges_gav - zcrit * Standard_error_effect_size_Hedges_Olkin, hedges_gav + zcrit * Standard_error_effect_size_Hedges_Olkin])
        confidence_intervals_Hedges_Olkin_tcrit_hedges_gav = np.array([ hedges_gav - tcrit * Standard_error_effect_size_Hedges_Olkin, hedges_gav + tcrit * Standard_error_effect_size_Hedges_Olkin])
        confidence_intervals_MLE_hedges_gav = np.array([ hedges_gav - zcrit * Standard_error_effect_size_MLE, hedges_gav + zcrit * Standard_error_effect_size_MLE])
        confidence_intervals_Large_N_hedges_gav = np.array([ hedges_gav - zcrit * Standard_error_effect_size_Large_N, hedges_gav + zcrit * Standard_error_effect_size_Large_N])
        confidence_intervals_Small_N_hedges_gav = np.array([ hedges_gav - zcrit * Standard_error_effect_size_Small_N, hedges_gav + zcrit * Standard_error_effect_size_Small_N])
        standard_error_becker_hedges_gav = np.sqrt( (1/sample_size + hedges_gav**2/(2*sample_size))*(2-2*correlation) )
        confidence_intervals_becker_zdist_hedges_gav = np.array([ hedges_gav - zcrit * standard_error_becker_hedges_gav, hedges_gav + zcrit * standard_error_becker_hedges_gav])
        confidence_intervals_becker_tdist_hedges_gav = np.array([ hedges_gav - tcrit * standard_error_becker_hedges_gav, hedges_gav + tcrit * standard_error_becker_hedges_gav])


        # 2. Non-Central Confidence Intervals        
        # 2.1 Pivotal Confidence Intervals  
        lower_pivotal, upper_pivotal = Pivotal_ci_t(t_score, df = sample_size-1, sample_size = sample_size, confidence_level = confidence_level)
        confidence_intervals_pivotal = np.array([lower_pivotal, upper_pivotal])

        # 2.2 Algina & Keselman, 2003
        lower_ci_algina_keselman, upper_ci_algina_keselman = CI_t_Algina_Keselman(Cohensdav, sd1, sd2, sample_size, correlation, confidence_level)

        # 2.3 t prime Confidence Intervals
        lower_ci_tprime_dav , upper_ci_tprime_dav = CI_t_prime_Paired_Samples(Cohensdav, sd1, sd2, sample_size, float(correlation), confidence_level)
        
        # 2.4 lambda prime Confidence Intervals
        lower_ci_lambda_prime_dav , upper_ci_lambda_prime_dav = CI_adjusted_lambda_prime_Paired_Samples(Cohensdav, sd1, sd2, sample_size, float(correlation), confidence_level)
        
        # 2.5 MAG CI's
        lower_ci_MAG_dav , upper_ci_MAG_dav = CI_MAG_Paired_Samples(Cohensdav, sd1, sd2, sample_size, correlation, confidence_level)
        
        # 2.6 Morris CI's
        lower_ci_Morris_dav , upper_ci_Morris_dav = CI_Morris_Paired_Samples(Cohensdav, sample_size, correlation, confidence_level)

        # 2.7 Goulette-Pelettier & Cousinaue
        ci_lower_Cohens_dav_NCP, ci_upper_Cohens_dav_NCP =  CI_NCP_one_Sample (Cohensdav, sample_size, confidence_level)


        results = {}


        # Cohen's dav CI

        # Non Central CI's        
        results["Cohen's dav Pivotal Confidence Interval (as in SPSS)"] = confidence_intervals_pivotal
        results["Cohen's dav Goulet-Pelletier & Cousineau Confidence Interval"] = np.array([round(ci_lower_Cohens_dav_NCP, 4), round(ci_upper_Cohens_dav_NCP, 4)])
        results["Cohen's dav Algina & Keselman (as in ESCI excel)"] = np.array([lower_ci_algina_keselman, upper_ci_algina_keselman])
        results["Cohen's dav Morris"] = np.array((lower_ci_Morris_dav), (upper_ci_Morris_dav))
        results["Cohen's dav t-prime"] = (lower_ci_tprime_dav)[0], (upper_ci_tprime_dav)[0]
        results["Cohen's dav Lambda Prime"] = (lower_ci_lambda_prime_dav)[0], (upper_ci_lambda_prime_dav)[0]
        results["Cohen's dav MAG"] = (lower_ci_MAG_dav), (upper_ci_MAG_dav)
        
        # Central CI's
        results["Cohen's dav Central Confidence Interval (True)"] = confidence_intervals_true
        results["Cohen's dav Central Confidence Interval (Morris)"] = confidence_intervals_Morris
        results["Cohen's dav Central Confidence Interval (Hedges)"] = confidence_intervals_Hedges
        results["Cohen's dav Central Confidence Interval (Hedges-Olkin Zdist)"] = confidence_intervals_Hedges_Olkin
        results["Cohen's dav Central Confidence Interval (Hedges-Olkin tdist) (as in pingouin package in python)"] = confidence_intervals_Hedges_Olkin_tcrit
        results["Cohen's dav Central Confidence Interval (MLE)"] = confidence_intervals_MLE
        results["Cohen's dav Central Confidence Interval (Large N)"] = confidence_intervals_Large_N
        results["Cohen's dav Central Confidence Interval (Small N)"] = confidence_intervals_Small_N
        results["Cohen's dav Becker Zdist"] = confidence_intervals_becker_zdist
        results["Cohen's dav Becker tdist"] = confidence_intervals_becker_tdist

        # Central CI's
        results["Hedges' gav Central Confidence Interval (True)"] = confidence_intervals_true_hedges_gav 
        results["Hedges' gav Central Confidence Interval (Morris)"] = confidence_intervals_Morris_hedges_gav 
        results["Hedges' gav Central Confidence Interval (Hedges)"] = confidence_intervals_Hedges_hedges_gav 
        results["Hedges' gav Central Confidence Interval (Hedges-Olkin)"] = confidence_intervals_Hedges_Olkin_hedges_gav 
        results["Cohen's dav Central Confidence Interval (Hedges-Olkin tdist) (as in pingouin package in python)"] = confidence_intervals_Hedges_Olkin_tcrit_hedges_gav 
        results["Hedges' gav Central Confidence Interval (MLE)"] = confidence_intervals_MLE_hedges_gav 
        results["Hedges' gav Central Confidence Interval (Large N)"] = confidence_intervals_Large_N_hedges_gav 
        results["Hedges' gav Central Confidence Interval (Small N)"] = confidence_intervals_Small_N_hedges_gav 
        results["Hedges' gav Becker Zdist"] = confidence_intervals_becker_zdist_hedges_gav 
        results["Hedges' gav Becker tdist"] = confidence_intervals_becker_tdist_hedges_gav 
    
        return results


    ##########################################
    ## 1.11 Hedges' gav CI Paired Samples ####
    ##########################################

    @staticmethod
    def Cohens_d_Paired_t_test_gav(params: dict) -> dict:
        
        hedges_g = params["Hedges g"]
        sample_size = params["Number of Pairs"]
        sd1 = params["Standard Deviation Group 1"]
        sd2 = params["Standard Deviation Group 2"]
        correlation = params["Correlation"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100
        df = sample_size -1
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2)) # For Hedges gav

        cohens_d = hedges_g / correction
        t_score = cohens_d * np.sqrt(sample_size)


        # 1. Central Confidence Intervals 
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
        tcrit = t.ppf(1 - (1 - confidence_level) / 2, df)

        # 1.1 Central Confidence Intervals using the Z-Distribution with A = n-1
        ci_lower, ci_upper, Standard_error_effect_size_True, Standard_error_effect_size_Morris, Standard_error_effect_size_Hedges, \
        Standard_error_effect_size_Hedges_Olkin, Standard_error_effect_size_MLE, \
        Standard_error_effect_size_Large_N, Standard_error_effect_size_Small_N = calculate_central_ci_paired_samples_t_test(hedges_g, sample_size, confidence_level)
        confidence_intervals_true = np.array([ci_lower, ci_upper])
        confidence_intervals_Morris = np.array([ hedges_g - zcrit * Standard_error_effect_size_Morris, hedges_g + zcrit * Standard_error_effect_size_Morris])
        confidence_intervals_Hedges = np.array([ hedges_g - zcrit * Standard_error_effect_size_Hedges, hedges_g + zcrit * Standard_error_effect_size_Hedges])
        confidence_intervals_Hedges_Olkin = np.array([ hedges_g - zcrit * Standard_error_effect_size_Hedges_Olkin, hedges_g + zcrit * Standard_error_effect_size_Hedges_Olkin])
        confidence_intervals_Hedges_Olkin_tcrit = np.array([ hedges_g - tcrit * Standard_error_effect_size_Hedges_Olkin, hedges_g + tcrit * Standard_error_effect_size_Hedges_Olkin])
        confidence_intervals_MLE = np.array([ hedges_g - zcrit * Standard_error_effect_size_MLE, hedges_g + zcrit * Standard_error_effect_size_MLE])
        confidence_intervals_Large_N = np.array([ hedges_g - zcrit * Standard_error_effect_size_Large_N, hedges_g + zcrit * Standard_error_effect_size_Large_N])
        confidence_intervals_Small_N = np.array([ hedges_g - zcrit * Standard_error_effect_size_Small_N, hedges_g + zcrit * Standard_error_effect_size_Small_N])
        standard_error_becker = np.sqrt( (1/sample_size + hedges_g**2/(2*sample_size))*(2-2*correlation) )
        confidence_intervals_becker_zdist = np.array([ hedges_g - zcrit * standard_error_becker, hedges_g + zcrit * standard_error_becker])
        confidence_intervals_becker_tdist = np.array([ hedges_g - tcrit * standard_error_becker, hedges_g + tcrit * standard_error_becker])

        # 1.2 Central Confidence Intervals using the Z-Distribution with A = n / (2*(1-correlation)
        ci_lower_pooled, ci_upper_pooled, Standard_error_effect_size_True_pooled, Standard_error_effect_size_Morris_pooled, Standard_error_effect_size_Hedges_pooled, \
        Standard_error_effect_size_Hedges_Olkin_pooled, Standard_error_effect_size_MLE_pooled, \
        Standard_error_effect_size_Large_N_pooled, Standard_error_effect_size_Small_N_pooled = calculate_SE_pooled_paired_samples_t_test(hedges_g, sample_size, correlation, confidence_level)
        confidence_intervals_true_pooled = np.array([ci_lower, ci_upper])
        confidence_intervals_Morris_pooled = np.array([ hedges_g - zcrit * Standard_error_effect_size_Morris, hedges_g + zcrit * Standard_error_effect_size_Morris])
        confidence_intervals_Hedges_pooled = np.array([ hedges_g - zcrit * Standard_error_effect_size_Hedges, hedges_g + zcrit * Standard_error_effect_size_Hedges])
        confidence_intervals_Hedges_Olkin_pooled = np.array([ hedges_g - zcrit * Standard_error_effect_size_Hedges_Olkin, hedges_g + zcrit * Standard_error_effect_size_Hedges_Olkin])
        confidence_intervals_MLE_pooled = np.array([ hedges_g - zcrit * Standard_error_effect_size_MLE, hedges_g + zcrit * Standard_error_effect_size_MLE])
        confidence_intervals_Large_N_pooled = np.array([ hedges_g - zcrit * Standard_error_effect_size_Large_N, hedges_g + zcrit * Standard_error_effect_size_Large_N])
        confidence_intervals_Small_N_pooled = np.array([ hedges_g - zcrit * Standard_error_effect_size_Small_N, hedges_g + zcrit * Standard_error_effect_size_Small_N])


        # 2. Non-Central Confidence Intervals        
        # 2.1 Pivotal Confidence Intervals  
        lower_pivotal, upper_pivotal = Pivotal_ci_t(t_score, df = sample_size-1, sample_size = sample_size, confidence_level = confidence_level)
        confidence_intervals_pivotal = np.array([lower_pivotal, upper_pivotal])

        # 2.2 Algina & Keselman, 2003
        lower_ci_algina_keselman, upper_ci_algina_keselman = CI_t_Algina_Keselman(hedges_g, sd1, sd2, sample_size, correlation, confidence_level)

        # 2.3 t prime Confidence Intervals
        lower_ci_tprime_dav , upper_ci_tprime_dav = CI_t_prime_Paired_Samples(hedges_g, sd1, sd2, sample_size, float(correlation), confidence_level)
        
        # 2.4 lambda prime Confidence Intervals
        lower_ci_lambda_prime_dav , upper_ci_lambda_prime_dav = CI_adjusted_lambda_prime_Paired_Samples(hedges_g, sd1, sd2, sample_size, float(correlation), confidence_level)
        
        # 2.5 MAG CI's
        lower_ci_MAG_dav , upper_ci_MAG_dav = CI_MAG_Paired_Samples(hedges_g, sd1, sd2, sample_size, correlation, confidence_level)
        
        # 2.6 Morris CI's
        lower_ci_Morris_dav , upper_ci_Morris_dav = CI_Morris_Paired_Samples(hedges_g, sample_size, correlation, confidence_level)

        # 2.7 Goulette-Pelettier & Cousinaue
        ci_lower_Cohens_dav_NCP, ci_upper_Cohens_dav_NCP =  CI_NCP_one_Sample (hedges_g, sample_size, confidence_level)


        results = {}

        # Hedges' gav CI
        # Non Central CI's
        results["Hedges' gav Pivotal Confidence Interval"] = confidence_intervals_pivotal * correction
    
        # Central CI's
        results["Hedges' gav Central Confidence Interval (True)"] = confidence_intervals_true 
        results["Hedges' gav Central Confidence Interval (Morris)"] = confidence_intervals_Morris 
        results["Hedges' gav Central Confidence Interval (Hedges)"] = confidence_intervals_Hedges 
        results["Hedges' gav Central Confidence Interval (Hedges-Olkin)"] = confidence_intervals_Hedges_Olkin 
        results["Hedges' gav Central Confidence Interval (Hedges-Olkin tdist) (as in pingouin package in python)"] = confidence_intervals_Hedges_Olkin_tcrit 
        results["Hedges' gav Central Confidence Interval (MLE)"] = confidence_intervals_MLE 
        results["Hedges' gav Central Confidence Interval (Large N)"] = confidence_intervals_Large_N 
        results["Hedges' gav Central Confidence Interval (Small N)"] = confidence_intervals_Small_N 
        results["Hedges' gav Becker Zdist"] = confidence_intervals_becker_zdist 
        results["Hedges' gav Becker tdist"] = confidence_intervals_becker_tdist 
    
        return results


    ##########################################
    ## 1.12 Cohens ds Independent Samples ####
    ##########################################

    @staticmethod
    def Cohens_ds_independent_samples_t_test(params: dict) -> dict:
        
        Cohensd = params["Cohensd"]
        sample_size_1 = params["Sample Size 1"]
        sample_size_2 = params["Sample Size 2"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        
        # Calculations
        confidence_level = Confidnece_Level_Percentages / 100
        sample_size = sample_size_1 + sample_size_2
        df = sample_size - 2
        t_score = Cohensd / np.sqrt((1/sample_size_1 + 1/sample_size_2))
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2)) # For Hedges gav
        hedgesg = Cohensd * correction
        cohens_dpop = Cohensd / np.sqrt((df/sample_size))
        t_score_dpop = cohens_dpop / np.sqrt((1/sample_size_1 + 1/sample_size_2))

        # Standard Errors Calculations
        ci_lower, ci_upper, Standard_error_effect_size_True, Standard_error_effect_size_Morris, Standard_error_effect_size_Hedges, \
        Standard_error_effect_size_Hedges_Olkin, Standard_error_effect_size_MLE, \
        Standard_error_effect_size_Large_N, Standard_error_effect_size_Small_N = calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test(Cohensd, sample_size_1, sample_size_2, confidence_level)

        ci_lower_hedges_gs, ci_upper_hedges_gs, Standard_error_effect_size_True_hedges_gs, Standard_error_effect_size_Morris_hedges_gs, Standard_error_effect_size_Hedges_hedges_gs, \
        Standard_error_effect_size_Hedges_Olkin_hedges_gs, Standard_error_effect_size_MLE_hedges_gs, \
        Standard_error_effect_size_Large_N_hedges_gs, Standard_error_effect_size_Small_N_hedges_gs = calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test(hedgesg, sample_size_1, sample_size_2, confidence_level)

        ci_lower_dpop, ci_upper_dpop, Standard_error_effect_size_True_dpop, Standard_error_effect_size_Morris_dpop, Standard_error_effect_size_Hedges_dpop, \
        Standard_error_effect_size_Hedges_Olkin_dpop, Standard_error_effect_size_MLE_dpop, \
        Standard_error_effect_size_Large_N_dpop, Standard_error_effect_size_Small_N_dpop = calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test(Cohensd, sample_size_1, sample_size_2, confidence_level)


        # Non-Central Confidence Intervals
        lower_pivotal_nct, upper_pivotal_nct = NCT_ci_t(t_score, df = sample_size-2, sample_size = sample_size, confidence_level = confidence_level)
        constant = np.sqrt((sample_size_1 + sample_size_2)/(sample_size_1 * sample_size_2))
        confidence_intervals_pivotal = np.array([lower_pivotal_nct * constant , upper_pivotal_nct * constant])

        lower_pivotal_nct_dpop, upper_pivotal_nct_dpop = NCT_ci_t(t_score_dpop, df = sample_size-2, sample_size = sample_size, confidence_level = confidence_level)
        confidence_intervals_pivotal_dpop = np.array([lower_pivotal_nct_dpop * constant , upper_pivotal_nct_dpop * constant])
        
        
        # Central Confidence Intervals
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
        tcrit = t.ppf(1 - (1 - confidence_level) / 2, df)

        confidence_intervals_true = np.array([ci_lower, ci_upper])
        confidence_intervals_Morris = np.array([ Cohensd - zcrit * Standard_error_effect_size_Morris, Cohensd + zcrit * Standard_error_effect_size_Morris])
        confidence_intervals_Hedges = np.array([ Cohensd - zcrit * Standard_error_effect_size_Hedges, Cohensd + zcrit * Standard_error_effect_size_Hedges])
        confidence_intervals_Hedges_Olkin = np.array([ Cohensd - zcrit * Standard_error_effect_size_Hedges_Olkin, Cohensd + zcrit * Standard_error_effect_size_Hedges_Olkin])
        confidence_intervals_MLE = np.array([ Cohensd - zcrit * Standard_error_effect_size_MLE, Cohensd + zcrit * Standard_error_effect_size_MLE])
        confidence_intervals_Large_N = np.array([ Cohensd - zcrit * Standard_error_effect_size_Large_N, Cohensd + zcrit * Standard_error_effect_size_Large_N])
        confidence_intervals_Small_N = np.array([ Cohensd - zcrit * Standard_error_effect_size_Small_N, Cohensd + zcrit * Standard_error_effect_size_Small_N])
        confidence_intervals_Hedges_Olkin_tcrit = np.array([ Cohensd - tcrit * Standard_error_effect_size_Hedges_Olkin, Cohensd + tcrit * Standard_error_effect_size_Hedges_Olkin])
        Hunter_Schmidt_SE = np.sqrt(((sample_size -1) / (sample_size - 3)) * ((4 / (sample_size)) * (1 + Cohensd**2/8)))
        confidence_interval_hunter_schmidt = np.array([ Cohensd - zcrit * Hunter_Schmidt_SE, Cohensd + zcrit * Hunter_Schmidt_SE])

        confidence_intervals_true_hedges_gs = np.array([ci_lower_hedges_gs, ci_upper_hedges_gs])
        confidence_intervals_Morris_hedges_gs = np.array([ hedgesg - zcrit * Standard_error_effect_size_Morris_hedges_gs, hedgesg + zcrit * Standard_error_effect_size_Morris_hedges_gs])
        confidence_intervals_Hedges_hedges_gs = np.array([ hedgesg - zcrit * Standard_error_effect_size_Hedges_hedges_gs, hedgesg + zcrit * Standard_error_effect_size_Hedges_hedges_gs])
        confidence_intervals_Hedges_Olkin_hedges_gs = np.array([ hedgesg - zcrit * Standard_error_effect_size_Hedges_Olkin_hedges_gs, hedgesg + zcrit * Standard_error_effect_size_Hedges_Olkin_hedges_gs])
        confidence_intervals_MLE_hedges_gs = np.array([ hedgesg - zcrit * Standard_error_effect_size_MLE_hedges_gs, hedgesg + zcrit * Standard_error_effect_size_MLE_hedges_gs])
        confidence_intervals_Large_N_hedges_gs = np.array([ hedgesg - zcrit * Standard_error_effect_size_Large_N_hedges_gs, hedgesg + zcrit * Standard_error_effect_size_Large_N_hedges_gs])
        confidence_intervals_Small_N_hedges_gs = np.array([ hedgesg - zcrit * Standard_error_effect_size_Small_N_hedges_gs, hedgesg + zcrit * Standard_error_effect_size_Small_N_hedges_gs])
        confidence_intervals_Hedges_Olkin_tcrit_hedges_gs = np.array([ hedgesg - tcrit * Standard_error_effect_size_Hedges_Olkin, hedgesg + tcrit * Standard_error_effect_size_Hedges_Olkin])
        Hunter_Schmidt_SE_hedges_gs = np.sqrt(  ( (sample_size -1) / (sample_size - 3)) * ((4 / (sample_size)) * (1 + hedgesg**2/8)))
        confidence_interval_hunter_schmidt_hedges_g = np.array([ hedgesg - zcrit * Hunter_Schmidt_SE_hedges_gs, hedgesg + zcrit * Hunter_Schmidt_SE_hedges_gs])

        confidence_intervals_true_dpop = np.array([ci_lower_dpop, ci_upper_dpop])
        confidence_intervals_Morris_dpop = np.array([ cohens_dpop - zcrit * Standard_error_effect_size_Morris_dpop, cohens_dpop + zcrit * Standard_error_effect_size_Morris_dpop])
        confidence_intervals_Hedges_dpop = np.array([ cohens_dpop - zcrit * Standard_error_effect_size_Hedges_dpop, cohens_dpop + zcrit * Standard_error_effect_size_Hedges_dpop])
        confidence_intervals_Hedges_Olkin_dpop = np.array([ cohens_dpop - zcrit * Standard_error_effect_size_Hedges_Olkin_dpop, cohens_dpop + zcrit * Standard_error_effect_size_Hedges_Olkin_dpop])
        confidence_intervals_MLE_dpop = np.array([ cohens_dpop - zcrit * Standard_error_effect_size_MLE_dpop, cohens_dpop + zcrit * Standard_error_effect_size_MLE_dpop])
        confidence_intervals_Large_N_dpop = np.array([ cohens_dpop - zcrit * Standard_error_effect_size_Large_N_dpop, cohens_dpop + zcrit * Standard_error_effect_size_Large_N_dpop])
        confidence_intervals_Small_N_dpop = np.array([ cohens_dpop - zcrit * Standard_error_effect_size_Small_N_dpop, cohens_dpop + zcrit * Standard_error_effect_size_Small_N_dpop])
        confidence_intervals_Hedges_Olkin_tcrit_dpop = np.array([ cohens_dpop - tcrit * Standard_error_effect_size_Hedges_Olkin_dpop, cohens_dpop + tcrit * Standard_error_effect_size_Hedges_Olkin_dpop])
        Hunter_Schmidt_SE_dpop = np.sqrt(  ( (sample_size -1) / (sample_size - 3)) * ((4 / (sample_size)) * (1 + cohens_dpop**2/8)))
        confidence_interval_hunter_schmidt_dpop = np.array([ cohens_dpop - zcrit * Hunter_Schmidt_SE_dpop, cohens_dpop + zcrit * Hunter_Schmidt_SE_dpop])


        results = {}
        results["Cohen's ds Pivotal Confidence Interval"] = confidence_intervals_pivotal
        results["Cohen's ds Central Confidence Interval (True)"] = confidence_intervals_true
        results["Cohen's ds Central Confidence Interval (Morris)"] = confidence_intervals_Morris
        results["Cohen's ds Central Confidence Interval (Hedges)"] = confidence_intervals_Hedges
        results["Cohen's ds Central Confidence Interval (Hedges-Olkin Zdist)"] = confidence_intervals_Hedges_Olkin
        results["Cohen's ds Central Confidence Interval (Hedges-Olkin tdist) (as in pingouin package in python)"] = confidence_intervals_Hedges_Olkin_tcrit
        results["Cohen's ds Central Confidence Interval (MLE)"] = confidence_intervals_MLE
        results["Cohen's ds Central Confidence Interval (Large N)"] = confidence_intervals_Large_N
        results["Cohen's ds Central Confidence Interval (Small N)"] = confidence_intervals_Small_N
        results["Cohen's ds Central Confidence Interval (Hunter Schmidt)"] = confidence_interval_hunter_schmidt

        results["Hedges' gs"] = hedgesg
        results["Hedges' gs Pivotal Confidence Interval"] = confidence_intervals_pivotal * correction
        results["Hedges' gs Central Confidence Interval (True)"] = confidence_intervals_true_hedges_gs
        results["Hedges' gs Central Confidence Interval (Morris)"] = confidence_intervals_Morris_hedges_gs
        results["Hedges' gs Central Confidence Interval (Hedges)"] = confidence_intervals_Hedges_hedges_gs
        results["Hedges' gs Central Confidence Interval (Hedges-Olkin Zdist)"] = confidence_intervals_Hedges_Olkin_hedges_gs
        results["Hedges' gs Central Confidence Interval (Hedges-Olkin tdist) (as in pingouin package in python)"] = confidence_intervals_Hedges_Olkin_tcrit_hedges_gs
        results["Hedges' gs Central Confidence Interval (MLE)"] = confidence_intervals_MLE_hedges_gs
        results["Hedges' gs Central Confidence Interval (Large N)"] = confidence_intervals_Large_N_hedges_gs
        results["Hedges' gs Central Confidence Interval (Small N)"] = confidence_intervals_Small_N_hedges_gs
        results["Hedges' gs Central Confidence Interval (Hunter Schmidt)"] = confidence_interval_hunter_schmidt_hedges_g

        results["Cohen's dpop"] = cohens_dpop
        results["Cohen's dpop Pivotal Confidence Interval"] = confidence_intervals_pivotal_dpop
        results["Cohen's dpop Central Confidence Interval (True)"] = confidence_intervals_true_dpop
        results["Cohen's dpop Central Confidence Interval (Morris)"] = confidence_intervals_Morris_dpop
        results["Cohen's dpop Central Confidence Interval (Hedges)"] = confidence_intervals_Hedges_dpop
        results["Cohen's dpop Central Confidence Interval (Hedges-Olkin Zdist)"] = confidence_intervals_Hedges_Olkin_dpop
        results["Cohen's dpop Central Confidence Interval (Hedges-Olkin tdist) (as in pingouin package in python)"] = confidence_intervals_Hedges_Olkin_tcrit_dpop
        results["Cohen's dpop Central Confidence Interval (MLE)"] = confidence_intervals_MLE_dpop
        results["Cohen's dpop Central Confidence Interval (Large N)"] = confidence_intervals_Large_N_dpop
        results["Cohen's dpop Central Confidence Interval (Small N)"] = confidence_intervals_Small_N_dpop
        results["Cohen's dpop Central Confidence Interval (Hunter Schmidt)"] = confidence_interval_hunter_schmidt_dpop


        return results



    ###########################################
    ## 1.13 Hedges' gs Independent Samples ####
    ###########################################

    @staticmethod
    def Cohens_gs_independent_samples_t_test(params: dict) -> dict:

        hedgesg = params["Hedges g"]
        sample_size_1 = params["Sample Size 1"]
        sample_size_2 = params["Sample Size 2"]
        Confidnece_Level_Percentages = params["Confidence Level"]
  
        # Calculations
        confidence_level = Confidnece_Level_Percentages / 100
        sample_size = sample_size_1 + sample_size_2
        df = sample_size - 2
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2)) # For Hedges gav
        Cohensd = hedgesg / correction
        t_score = Cohensd / np.sqrt((1/sample_size_1 + 1/sample_size_2)) 

        # Standard Errors Calculations
        ci_lower_hedges_gs, ci_upper_hedges_gs, Standard_error_effect_size_True_hedges_gs, Standard_error_effect_size_Morris_hedges_gs, Standard_error_effect_size_Hedges_hedges_gs, \
        Standard_error_effect_size_Hedges_Olkin_hedges_gs, Standard_error_effect_size_MLE_hedges_gs, \
        Standard_error_effect_size_Large_N_hedges_gs, Standard_error_effect_size_Small_N_hedges_gs = calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test(hedgesg, sample_size_1, sample_size_2, confidence_level)



        # Non-Central Confidence Intervals
        lower_pivotal_nct, upper_pivotal_nct = NCT_ci_t(t_score, df = sample_size-2, sample_size = sample_size, confidence_level = confidence_level)
        constant = np.sqrt((sample_size_1 + sample_size_2)/(sample_size_1 * sample_size_2))
        confidence_intervals_pivotal = np.array([lower_pivotal_nct * constant , upper_pivotal_nct * constant])
        
        
        # Central Confidence Intervals
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
        tcrit = t.ppf(1 - (1 - confidence_level) / 2, df)

        confidence_intervals_true_hedges_gs = np.array([ci_lower_hedges_gs, ci_upper_hedges_gs])
        confidence_intervals_Morris_hedges_gs = np.array([ hedgesg - zcrit * Standard_error_effect_size_Morris_hedges_gs, hedgesg + zcrit * Standard_error_effect_size_Morris_hedges_gs])
        confidence_intervals_Hedges_hedges_gs = np.array([ hedgesg - zcrit * Standard_error_effect_size_Hedges_hedges_gs, hedgesg + zcrit * Standard_error_effect_size_Hedges_hedges_gs])
        confidence_intervals_Hedges_Olkin_hedges_gs = np.array([ hedgesg - zcrit * Standard_error_effect_size_Hedges_Olkin_hedges_gs, hedgesg + zcrit * Standard_error_effect_size_Hedges_Olkin_hedges_gs])
        confidence_intervals_MLE_hedges_gs = np.array([ hedgesg - zcrit * Standard_error_effect_size_MLE_hedges_gs, hedgesg + zcrit * Standard_error_effect_size_MLE_hedges_gs])
        confidence_intervals_Large_N_hedges_gs = np.array([ hedgesg - zcrit * Standard_error_effect_size_Large_N_hedges_gs, hedgesg + zcrit * Standard_error_effect_size_Large_N_hedges_gs])
        confidence_intervals_Small_N_hedges_gs = np.array([ hedgesg - zcrit * Standard_error_effect_size_Small_N_hedges_gs, hedgesg + zcrit * Standard_error_effect_size_Small_N_hedges_gs])
        confidence_intervals_Hedges_Olkin_tcrit_hedges_gs = np.array([ hedgesg - tcrit * Standard_error_effect_size_Hedges_Olkin_hedges_gs, hedgesg + tcrit * Standard_error_effect_size_Hedges_Olkin_hedges_gs])
        Hunter_Schmidt_SE_hedges_gs = np.sqrt(  ( (sample_size -1) / (sample_size - 3)) * ((4 / (sample_size)) * (1 + hedgesg**2/8)))
        confidence_interval_hunter_schmidt_hedges_g = np.array([ hedgesg - zcrit * Hunter_Schmidt_SE_hedges_gs, hedgesg + zcrit * Hunter_Schmidt_SE_hedges_gs])

        results = {}
        results["Hedges' gs"] = hedgesg
        results["Hedges' gs Pivotal Confidence Interval"] = confidence_intervals_pivotal * correction
        results["Hedges' gs Central Confidence Interval (True)"] = confidence_intervals_true_hedges_gs
        results["Hedges' gs Central Confidence Interval (Morris)"] = confidence_intervals_Morris_hedges_gs
        results["Hedges' gs Central Confidence Interval (Hedges)"] = confidence_intervals_Hedges_hedges_gs
        results["Hedges' gs Central Confidence Interval (Hedges-Olkin Zdist)"] = confidence_intervals_Hedges_Olkin_hedges_gs
        results["Hedges' gs Central Confidence Interval (Hedges-Olkin tdist) (as in pingouin package in python)"] = confidence_intervals_Hedges_Olkin_tcrit_hedges_gs
        results["Hedges' gs Central Confidence Interval (MLE)"] = confidence_intervals_MLE_hedges_gs
        results["Hedges' gs Central Confidence Interval (Large N)"] = confidence_intervals_Large_N_hedges_gs
        results["Hedges' gs Central Confidence Interval (Small N)"] = confidence_intervals_Small_N_hedges_gs
        results["Hedges' gs Central Confidence Interval (Hunter Schmidt)"] = confidence_interval_hunter_schmidt_hedges_g
        return results

    # Thing's to consider

    # 1.Add Confidence Intervals adopted from fitts - it is still not supporting his criticism  
    # 2. Let user put as much as parameters as possible but give a more accurate as CI's as more parameters are provided
    # 3. Find out whether Cousineau suggestions are applicabele for the Hedges gav versions



