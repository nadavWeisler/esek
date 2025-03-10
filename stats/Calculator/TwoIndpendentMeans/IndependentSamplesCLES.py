
#####################################################
## Effect Size for Two Independent Samples CLES  ####
#####################################################

import numpy as np
from scipy.stats import norm, nct, t, rankdata
import math 
from collections import Counter
import pandas as pd

# Relevant Functions for Common Language Effect Size

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

# 2. Central CI's
def calculate_Central_ci_from_cohens_d_two_indpednent_sample_t_test(cohens_d, sample_size1, sample_size2, confidence_level): # This is a function that calculates the Confidence Intervals of the Effect size in One Sample Z_score test (or two dependent samples)
    sample_size = sample_size1+sample_size2
    df = sample_size - 2 # This is the Degrees of Freedom for one sample t-test
    correction_factor =  math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
    harmonic_sample_size = 2 / (1/sample_size1 + 1/sample_size2)
    standard_error_es = np.sqrt((df/(df-2)) * (2 / harmonic_sample_size ) * (1 + cohens_d**2 * harmonic_sample_size / 2)  - (cohens_d**2 / correction_factor**2))# This formula from Hedges, 1981, is the True formula for one sample t-test (The hedges and Olking (1985) formula is for one sampele z-test or where N is large enough)
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = cohens_d - standard_error_es * z_critical_value,  cohens_d + standard_error_es * z_critical_value
    return ci_lower, ci_upper, standard_error_es


class Two_Independent_samples_Common_Language_Effect_Sizes():
    @staticmethod
    def Common_Language_Effect_Sizes_Independent_Samples_from_t_score(params: dict) -> dict: # Note that from Parameters, ONnly the Parametric Effect Sizes Can be Calculated
    
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

        # Parametric Common Language Effect Sizes
        #########################################

        # 1. Based on Cohen's ds
        cohens_U2_ds = (norm.cdf(abs(cohens_ds)/2))
        cohens_U1_ds =  (2* cohens_U2_ds - 1) / cohens_U2_ds
        cohens_U3_ds = norm.cdf(abs(cohens_ds))
        Mcgraw_Wong_CLds = norm.cdf(cohens_ds / np.sqrt(2))
        proportion_of_overlap_ds = 2 * norm.cdf(-abs(cohens_ds) / 2)

        # 2. Based on Cohen's dpop
        cohens_U2_dpop = (norm.cdf(abs(cohens_dpop)/2))
        cohens_U1_dpop =  (2*abs(cohens_U2_dpop) - 1) / cohens_U2_dpop
        cohens_U3_dpop = norm.cdf(abs(cohens_dpop))
        Mcgraw_Wong_CLdpop = norm.cdf(cohens_dpop / np.sqrt(2)) # See the suggestion of Ruscio for this effect size
        proportion_of_overlap_dpop = 2 * norm.cdf(-abs(cohens_dpop) / 2)

        # 3. Based in Hedges' gs
        cohens_U2_gs = (norm.cdf(abs(hedges_gs)/2))
        cohens_U1_gs=  (2*abs(cohens_U2_gs) - 1) / cohens_U2_gs
        cohens_U3_gs = norm.cdf(abs(hedges_gs))            
        Mcgraw_Wong_CLgs = norm.cdf(hedges_gs / np.sqrt(2)) # Probability of Superiority
        proportion_of_overlap_gs = 2 * norm.cdf(-abs(hedges_gs) / 2)

        # 4. Central Confidence Intervals
        ci_lower_cohens_ds_Central, ci_upper_cohens_ds_Central, standard_error_cohens_ds =  calculate_Central_ci_from_cohens_d_two_indpednent_sample_t_test (cohens_ds, sample_size_1, sample_size_2, confidence_level)
        ci_lower_hedges_gs_Central, ci_upper_hedges_gs_Central, standard_error_hedges_gs =  calculate_Central_ci_from_cohens_d_two_indpednent_sample_t_test (hedges_gs, sample_size_1, sample_size_2, confidence_level)
        ci_lower_cohens_dpop_Central, ci_upper_cohens_dpop_Central, standard_error_cohens_dpop =  calculate_Central_ci_from_cohens_d_two_indpednent_sample_t_test (cohens_dpop, sample_size_1, sample_size_2, confidence_level)

        # 5. Non-Central Confidence Intervals
        nct_ci_lower_Pivotal_ds, nct_ci_upper_Pivotal_ds =  Pivotal_ci_t ( t_score, df, sample_size, confidence_level)
        constant = np.sqrt((sample_size_1 + sample_size_2)/(sample_size_1 * sample_size_2))
        ci_lower_Pivotal_ds, ci_upper_Pivotal_ds = nct_ci_lower_Pivotal_ds * constant , nct_ci_upper_Pivotal_ds * constant
        ci_lower_Pivotal_gs, ci_upper_Pivotal_gs = ci_lower_Pivotal_ds * correction, ci_upper_Pivotal_ds * correction       

        t_score_dpop = cohens_dpop / np.sqrt((1/sample_size_1 + 1/sample_size_2))
        nct_ci_lower_Pivotal_dpop, nct_ci_upper_Pivotal_dpop =  Pivotal_ci_t (t_score_dpop, df, sample_size, confidence_level)
        ci_lower_Pivotal_dpop, ci_upper_Pivotal_dpop = nct_ci_lower_Pivotal_dpop * constant, nct_ci_upper_Pivotal_dpop * constant

        
        # Set results
        results = {}
        
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["t-score"] = np.around(t_score, 4)

        # Parametric Effect Sizes Based on the Distribution of Cohen's ds
        results["Cohen's U1_ds"] = np.around(cohens_U1_ds*100, 4) 
        results["Lower Central CI U1_ds"] = ci_lower_cohens_U1_ds_Central = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2)))) > (((2 * (norm.cdf((ci_upper_cohens_ds_Central) / 2)) - 1) / (norm.cdf((ci_upper_cohens_ds_Central) / 2)))) else (((2 * (norm.cdf((ci_lower_cohens_ds_Central) / 2)) - 1) / (norm.cdf((ci_lower_cohens_ds_Central) / 2))))),0),3) # type: ignore
        results["Upper Central CI U1_ds"] = ci_upper_cohens_U1_ds_Central = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_cohens_ds_Central) / 2)) - 1) / (norm.cdf(abs(ci_lower_cohens_ds_Central) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)))))),100),3) # type: ignore
        results["Lower Pivotal CI U1_ds"] = ci_lower_cohens_U1_ds_Pivotal = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)))) > (((2 * (norm.cdf((ci_upper_Pivotal_ds) / 2)) - 1) / (norm.cdf((ci_upper_Pivotal_ds) / 2)))) else (((2 * (norm.cdf((ci_lower_Pivotal_ds) / 2)) - 1) / (norm.cdf((ci_lower_Pivotal_ds) / 2))))),0),3) # type: ignore
        results["Upper Pivotal CI U1_ds"] = ci_upper_cohens_U1_ds_Pivotal = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_Pivotal_ds) / 2)) - 1) / (norm.cdf(abs(ci_lower_Pivotal_ds) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)))))),100),3) # type: ignore
        results["Statistical Line Cohen's U1_ds"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2081d\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U1_ds * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U1_ds_Pivotal, 3),  np.around(ci_upper_cohens_U1_ds_Pivotal, 3))

        results["Cohen's U2_ds"] = np.around(cohens_U2_ds*100, 4)
        results["Lower Central CI U2_ds"] = ci_lower_cohens_U2_ds_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_cohens_ds_Central) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2))  ) )) > ((( (norm.cdf((ci_upper_cohens_ds_Central) / 2))  ) )) else ((( (norm.cdf((ci_lower_cohens_ds_Central) / 2))  ) ))),50),3) # type: ignore
        results["Upper Central CI U2_ds"] = ci_upper_cohens_U2_ds_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_cohens_ds_Central) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U2_ds"] = ci_lower_cohens_U2_ds_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_ds) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_ds) / 2))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_ds) / 2))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_ds) / 2))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U2_ds"] = ci_upper_cohens_U2_ds_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_ds) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_ds) / 2))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U2_ds"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2082d\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U2_ds * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U2_ds_Pivotal, 3),  np.around(ci_upper_cohens_U2_ds_Pivotal, 3))

        results["Cohen's U3_ds"] = np.around(cohens_U3_ds*100, 4)
        results["Lower Central CI U3_ds"] = ci_lower_cohens_U3_ds_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_cohens_ds_Central)))  ) )) if ((( (norm.cdf(abs(ci_upper_cohens_ds_Central)  ))  ) )) > ((( (norm.cdf((ci_upper_cohens_ds_Central)  ))  ) )) else ((( (norm.cdf((ci_lower_cohens_ds_Central)  ))  ) ))),50),3) # type: ignore
        results["Upper Central CI U3_ds"] = ci_upper_cohens_U3_ds_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_cohens_ds_Central) ))  ) )), ((( (norm.cdf(abs(ci_upper_cohens_ds_Central)  ))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U3_ds"] = ci_lower_cohens_U3_ds_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_ds)))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_ds)  ))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_ds)  ))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_ds)  ))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U3_ds"] = ci_upper_cohens_U3_ds_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_ds)))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_ds)  ))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U3_ds"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2083d\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U3_ds * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U3_ds_Pivotal, 3),  np.around(ci_upper_cohens_U3_ds_Pivotal, 3))

        results["Mcgraw and Wong, CLds"] = np.around(100 * (Mcgraw_Wong_CLds),4)
        results["Lower Central Ci Mcgraw and Wong, CLds"] = ci_lower_clds_Central = np.around(100 * (max(norm.cdf(ci_lower_cohens_ds_Central / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Central Ci Mcgraw and Wong, CLds"] = ci_upper_clds_Central = np.around(100 * (min(norm.cdf(ci_upper_cohens_ds_Central / np.sqrt(2)),1)),4) # type: ignore
        results["Lower Pivotal Ci Mcgraw and Wong, CLds"] = ci_lower_clds_Pivotal = np.around(100 * (max(norm.cdf(ci_lower_Pivotal_ds / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Pivotal Ci Mcgraw and Wong, CLds"] = ci_upper_clds_Pivotal = np.around(100 * (min(norm.cdf(ci_upper_Pivotal_ds / np.sqrt(2)),1)),4) # type: ignore
        results["Statistical Line Mcgraw and Wong, CLds"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLd\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(Mcgraw_Wong_CLds * 100),1), confidence_level_percentages , np.around(ci_lower_clds_Pivotal, 3),  np.around(ci_upper_clds_Pivotal, 3))

        results["Proportion Of Overlap ds"] = np.around(proportion_of_overlap_ds, 4)
        results["Lower Central CI Proportion of Overlap_ds"] = ci_lower_pov_Central_ds = np.around((2 * norm.cdf(-abs(ci_lower_cohens_ds_Central) / 2)), 4) # type: ignore
        results["Upper Central CI Proportion of Overlap_ds"] = ci_upper_pov_Central_ds = np.around((2 * norm.cdf(-abs(ci_upper_cohens_ds_Central) / 2)), 4) # type: ignore
        results["Lower Pivotal CI Proportion of Overlap_ds"] = ci_lower_pov_Pivotal_ds = np.around((2 * norm.cdf(-abs(ci_lower_Pivotal_ds) / 2)), 4) # type: ignore
        results["Upper Pivotal CI Proportion of Overlap_ds"] = ci_upper_pov_Pivotal_ds = np.around((2 * norm.cdf(-abs(ci_upper_Pivotal_ds) / 2)), 4) # type: ignore
        results["Statistical Line Proportion of Overlap (POVds) "] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, POVd\u209B = {:.3f}%, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(proportion_of_overlap_ds),3), confidence_level_percentages , np.around(ci_lower_pov_Pivotal_ds, 3),  np.around(ci_upper_pov_Pivotal_ds, 3))


        # Parametric Effect Sizes Based on the Distribution of Hedge's gs
        results["Cohen's U1_gs"] = np.around(cohens_U1_gs*100, 4) 
        results["Lower Central CI U1_gs"] = ci_lower_cohens_U1_gs_Central = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2)))) > (((2 * (norm.cdf((ci_upper_hedges_gs_Central) / 2)) - 1) / (norm.cdf((ci_upper_hedges_gs_Central) / 2)))) else (((2 * (norm.cdf((ci_lower_hedges_gs_Central) / 2)) - 1) / (norm.cdf((ci_lower_hedges_gs_Central) / 2))))),0),3) # type: ignore
        results["Upper Central CI U1_gs"] = ci_upper_cohens_U1_gs_Central = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_hedges_gs_Central) / 2)) - 1) / (norm.cdf(abs(ci_lower_hedges_gs_Central) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)))))),100),3) # type: ignore
        results["Lower Pivotal CI U1_gs"] = ci_lower_cohens_U1_gs_Pivotal = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)))) > (((2 * (norm.cdf((ci_upper_Pivotal_gs) / 2)) - 1) / (norm.cdf((ci_upper_Pivotal_gs) / 2)))) else (((2 * (norm.cdf((ci_lower_Pivotal_gs) / 2)) - 1) / (norm.cdf((ci_lower_Pivotal_gs) / 2))))),0),3) # type: ignore
        results["Upper Pivotal CI U1_gs"] = ci_upper_cohens_U1_gs_Pivotal = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_Pivotal_gs) / 2)) - 1) / (norm.cdf(abs(ci_lower_Pivotal_gs) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)))))),100),3) # type: ignore
        results["Statistical Line Cohen's U1_gs"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2081g\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U1_gs * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U1_gs_Pivotal, 3),  np.around(ci_upper_cohens_U1_gs_Pivotal, 3))

        results["Cohen's U2_gs"] = np.around(cohens_U2_gs*100, 4)
        results["Lower Central CI U2_gs"] = ci_lower_cohens_U2_gs_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_hedges_gs_Central) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2))  ) )) > ((( (norm.cdf((ci_upper_hedges_gs_Central) / 2))  ) )) else ((( (norm.cdf((ci_lower_hedges_gs_Central) / 2))  ) ))),50),3) # type: ignore
        results["Upper Central CI U2_gs"] = ci_upper_cohens_U2_gs_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_hedges_gs_Central) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U2_gs"] = ci_lower_cohens_U2_gs_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_gs) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_gs) / 2))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_gs) / 2))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_gs) / 2))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U2_gs"] = ci_upper_cohens_U2_gs_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_gs) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_gs) / 2))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U2_gs"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2082g\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U2_gs * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U2_gs_Pivotal, 3),  np.around(ci_upper_cohens_U2_gs_Pivotal, 3))

        results["Cohen's U3_gs"] = np.around(cohens_U3_gs*100, 4)
        results["Lower Central CI U3_gs"] = ci_lower_cohens_U3_gs_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_hedges_gs_Central)))  ) )) if ((( (norm.cdf(abs(ci_upper_hedges_gs_Central)  ))  ) )) > ((( (norm.cdf((ci_upper_hedges_gs_Central)  ))  ) )) else ((( (norm.cdf((ci_lower_hedges_gs_Central)  ))  ) ))),50),3) # type: ignore
        results["Upper Central CI U3_gs"] = ci_upper_cohens_U3_gs_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_hedges_gs_Central) ))  ) )), ((( (norm.cdf(abs(ci_upper_hedges_gs_Central)  ))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U3_gs"] = ci_lower_cohens_U3_gs_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_gs)))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_gs)  ))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_gs)  ))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_gs)  ))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U3_gs"] = ci_upper_cohens_U3_gs_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_gs)))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_gs)  ))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U3_gs"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2083g\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U3_gs * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U3_gs_Pivotal, 3),  np.around(ci_upper_cohens_U3_gs_Pivotal, 3))

        results["Mcgraw and Wong, CLgs"] = np.around(100 * (Mcgraw_Wong_CLgs),4)
        results["Lower Central Ci Mcgraw and Wong, CLgs"] = ci_lower_clgs_Central = np.around(100 * (max(norm.cdf(ci_lower_hedges_gs_Central / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Central Ci Mcgraw and Wong, CLgs"] = ci_upper_clgs_Central = np.around(100 * (min(norm.cdf(ci_upper_hedges_gs_Central / np.sqrt(2)),1)),4) # type: ignore
        results["Lower Pivotal Ci Mcgraw and Wong, CLgs"] = ci_lower_clgs_Pivotal = np.around(100 * (max(norm.cdf(ci_lower_Pivotal_gs / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Pivotal Ci Mcgraw and Wong, CLgs"] = ci_upper_clgs_Pivotal = np.around(100 * (min(norm.cdf(ci_upper_Pivotal_gs / np.sqrt(2)),1)),4) # type: ignore
        results["Statistical Line Mcgraw and Wong, CLgs"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLg\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(Mcgraw_Wong_CLgs * 100),1), confidence_level_percentages , np.around(ci_lower_clgs_Pivotal, 3),  np.around(ci_upper_clgs_Pivotal, 3))

        results["Proportion Of Overlap gs"] = np.around(proportion_of_overlap_gs, 4)
        results["Lower Central CI Proportion of Overlap_gs"] = ci_lower_pov_Central_gs = np.around((2 * norm.cdf(-abs(ci_lower_hedges_gs_Central) / 2)), 4) # type: ignore
        results["Upper Central CI Proportion of Overlap_gs"] = ci_upper_pov_Central_gs = np.around((2 * norm.cdf(-abs(ci_upper_hedges_gs_Central) / 2)), 4) # type: ignore
        results["Lower Pivotal CI Proportion of Overlap_gs"] = ci_lower_pov_Pivotal_gs = np.around((2 * norm.cdf(-abs(ci_lower_Pivotal_gs) / 2)), 4) # type: ignore
        results["Upper Pivotal CI Proportion of Overlap_gs"] = ci_upper_pov_Pivotal_gs = np.around((2 * norm.cdf(-abs(ci_upper_Pivotal_gs) / 2)), 4) # type: ignore
        results["Statistical Line Proportion of Overlap (POVgs) "] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, POVg\u209B = {:.3f}%, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(proportion_of_overlap_gs),3), confidence_level_percentages , np.around(ci_lower_pov_Pivotal_gs, 3),  np.around(ci_upper_pov_Pivotal_gs, 3))


        # Parametric Effect Sizes Based on the Distribution of Cohen's dpop
        results["Cohen's U1_dpop"] = np.around(cohens_U1_dpop*100, 4) 
        results["Lower Central CI U1_dpop"] = ci_lower_cohens_U1_dpop_Central = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2)))) > (((2 * (norm.cdf((ci_upper_cohens_dpop_Central) / 2)) - 1) / (norm.cdf((ci_upper_cohens_dpop_Central) / 2)))) else (((2 * (norm.cdf((ci_lower_cohens_dpop_Central) / 2)) - 1) / (norm.cdf((ci_lower_cohens_dpop_Central) / 2))))),0),3) # type: ignore
        results["Upper Central CI U1_dpop"] = ci_upper_cohens_U1_dpop_Central = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_cohens_dpop_Central) / 2)) - 1) / (norm.cdf(abs(ci_lower_cohens_dpop_Central) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)))))),100),3) # type: ignore
        results["Lower Pivotal CI U1_dpop"] = ci_lower_cohens_U1_dpop_Pivotal = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)))) > (((2 * (norm.cdf((ci_upper_Pivotal_dpop) / 2)) - 1) / (norm.cdf((ci_upper_Pivotal_dpop) / 2)))) else (((2 * (norm.cdf((ci_lower_Pivotal_dpop) / 2)) - 1) / (norm.cdf((ci_lower_Pivotal_dpop) / 2))))),0),3) # type: ignore
        results["Upper Pivotal CI U1_dpop"] = ci_upper_cohens_U1_dpop_Pivotal = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_Pivotal_dpop) / 2)) - 1) / (norm.cdf(abs(ci_lower_Pivotal_dpop) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)))))),100),3) # type: ignore
        results["Statistical Line Cohen's U1_dpop"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2081d\u209a\u2092\u209a = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U1_dpop * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U1_dpop_Pivotal, 3),  np.around(ci_upper_cohens_U1_dpop_Pivotal, 3))

        results["Cohen's U2_dpop"] = np.around(cohens_U2_dpop*100, 4)
        results["Lower Central CI U2_dpop"] = ci_lower_cohens_U2_dpop_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2))  ) )) > ((( (norm.cdf((ci_upper_cohens_dpop_Central) / 2))  ) )) else ((( (norm.cdf((ci_lower_cohens_dpop_Central) / 2))  ) ))),50),3) # type: ignore
        results["Upper Central CI U2_dpop"] = ci_upper_cohens_U2_dpop_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_cohens_dpop_Central) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U2_dpop"] = ci_lower_cohens_U2_dpop_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_dpop) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_dpop) / 2))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_dpop) / 2))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U2_dpop"] = ci_upper_cohens_U2_dpop_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_dpop) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U2_dpop"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2082d\u209a\u2092\u209a = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U2_dpop * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U2_dpop_Pivotal, 3),  np.around(ci_upper_cohens_U2_dpop_Pivotal, 3))

        results["Cohen's U3_dpop"] = np.around(cohens_U3_dpop*100, 4)
        results["Lower Central CI U3_dpop"] = ci_lower_cohens_U3_dpop_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_cohens_dpop_Central)))  ) )) if ((( (norm.cdf(abs(ci_upper_cohens_dpop_Central)  ))  ) )) > ((( (norm.cdf((ci_upper_cohens_dpop_Central)  ))  ) )) else ((( (norm.cdf((ci_lower_cohens_dpop_Central)  ))  ) ))),50),3) # type: ignore
        results["Upper Central CI U3_dpop"] = ci_upper_cohens_U3_dpop_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_cohens_dpop_Central) ))  ) )), ((( (norm.cdf(abs(ci_upper_cohens_dpop_Central)  ))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U3_dpop"] = ci_lower_cohens_U3_dpop_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_dpop)))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_dpop)  ))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_dpop)  ))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_dpop)  ))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U3_dpop"] = ci_upper_cohens_U3_dpop_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_dpop)))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_dpop)  ))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U3_dpop"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2083d\u209a\u2092\u209a = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U3_dpop * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U3_dpop_Pivotal, 3),  np.around(ci_upper_cohens_U3_dpop_Pivotal, 3))

        results["Mcgraw and Wong, CLdpop"] = np.around(100 * (Mcgraw_Wong_CLdpop),4)
        results["Lower Central Ci Mcgraw and Wong, CLdpop"] = ci_lower_cldpop_Central = np.around(100 * (max(norm.cdf(ci_lower_cohens_dpop_Central / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Central Ci Mcgraw and Wong, CLdpop"] = ci_upper_cldpop_Central = np.around(100 * (min(norm.cdf(ci_upper_cohens_dpop_Central / np.sqrt(2)),1)),4) # type: ignore
        results["Lower Pivotal Ci Mcgraw and Wong, CLdpop"] = ci_lower_cldpop_Pivotal = np.around(100 * (max(norm.cdf(ci_lower_Pivotal_dpop / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Pivotal Ci Mcgraw and Wong, CLdpop"] = ci_upper_cldpop_Pivotal = np.around(100 * (min(norm.cdf(ci_upper_Pivotal_dpop / np.sqrt(2)),1)),4) # type: ignore
        results["Statistical Line Mcgraw and Wong, CLdpop"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLd\u209a\u2092\u209a = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(Mcgraw_Wong_CLdpop * 100),1), confidence_level_percentages , np.around(ci_lower_cldpop_Pivotal, 3),  np.around(ci_upper_cldpop_Pivotal, 3))

        results["Proportion Of Overlap dpop"] = np.around(proportion_of_overlap_dpop, 4)
        results["Lower Central CI Proportion of Overlap_dpop"] = ci_lower_pov_Central_dpop = np.around((2 * norm.cdf(-abs(ci_lower_cohens_dpop_Central) / 2)), 4) # type: ignore
        results["Upper Central CI Proportion of Overlap_dpop"] = ci_upper_pov_Central_dpop = np.around((2 * norm.cdf(-abs(ci_upper_cohens_dpop_Central) / 2)), 4) # type: ignore
        results["Lower Pivotal CI Proportion of Overlap_dpop"] = ci_lower_pov_Pivotal_dpop = np.around((2 * norm.cdf(-abs(ci_lower_Pivotal_dpop) / 2)), 4) # type: ignore
        results["Upper Pivotal CI Proportion of Overlap_dpop"] = ci_upper_pov_Pivotal_dpop = np.around((2 * norm.cdf(-abs(ci_upper_Pivotal_dpop) / 2)), 4) # type: ignore
        results["Statistical Line Proportion of Overlap (POVdpop) "] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, POVd\u209a\u2092\u209a = {:.3f}%, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(proportion_of_overlap_dpop),3), confidence_level_percentages , np.around(ci_lower_pov_Pivotal_dpop, 3),  np.around(ci_upper_pov_Pivotal_dpop, 3))

        return results



    @staticmethod
    def Common_Language_Effect_Sizes_Independent_Samples_from_Parameters(params: dict) -> dict: # Note that from Parameters, ONnly the Parametric Effect Sizes Can be Calculated

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
        cohens_ds = (sample_mean_1 - sample_mean_2 - population_mean_diff)  / (np.sqrt((((sample_size_1-1)*(sample_sd_1**2)) + ((sample_size_2-1)*(sample_sd_2**2))) / df))
        cohens_dpop = cohens_ds / np.sqrt(df/(df+2))
        cohens_dav = (sample_mean_1 - sample_mean_2 -population_mean_diff) / np.sqrt((sample_sd_1**2 + sample_sd_2**2) / 2)
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
        hedges_gs = cohens_ds * correction
        hedges_gav = cohens_dav * correction
        standardizer_ds = np.sqrt(((((sample_size_1-1)*sample_sd_1**2)) + ((sample_size_2-1)*sample_sd_2**2)) / (sample_size-2)) # this is Spooled
        standard_error = standardizer_ds  * np.sqrt((sample_size_1+sample_size_2)/(sample_size_1*sample_size_2))        
        t_score = ((sample_mean_1 - sample_mean_2 - population_mean_diff) / standard_error)
        t_score_av = ((sample_mean_1 - sample_mean_2) - population_mean_diff ) /   (np.sqrt((sample_sd_1**2 + sample_sd_2**2)/2) * np.sqrt((sample_size_1+sample_size_2)/(sample_size_1*sample_size_2))) 
        t_score_dpop = cohens_dpop / np.sqrt((1/sample_size_1 + 1/sample_size_2))
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)


        # Parametric Common Language Effect Sizes
        #########################################

        # 1. Based on Cohen's ds
        cohens_U2_ds = (norm.cdf(abs(cohens_ds)/2))
        cohens_U1_ds =  (2* cohens_U2_ds - 1) / cohens_U2_ds
        cohens_U3_ds = norm.cdf(abs(cohens_ds))
        Mcgraw_Wong_CLds = norm.cdf(cohens_ds / np.sqrt(2))
        proportion_of_overlap_ds = 2 * norm.cdf(-abs(cohens_ds) / 2)

        # 2. Based on Cohen's dpop
        cohens_U2_dpop = (norm.cdf(abs(cohens_dpop)/2))
        cohens_U1_dpop =  (2*abs(cohens_U2_dpop) - 1) / cohens_U2_dpop
        cohens_U3_dpop = norm.cdf(abs(cohens_dpop))
        Mcgraw_Wong_CLdpop = norm.cdf(cohens_dpop / np.sqrt(2)) # See the suggestion of Ruscio for this effect size
        proportion_of_overlap_dpop = 2 * norm.cdf(-abs(cohens_dpop) / 2)

        # 3. Based on Cohen's dav
        cohens_U2_dav = (norm.cdf(abs(cohens_dav)/2))
        cohens_U1_dav =  (2*abs(cohens_U2_dav) - 1) / cohens_U2_dav
        cohens_U3_dav = norm.cdf(abs(cohens_dav))            
        Mcgraw_Wong_CLdav = norm.cdf(cohens_dav / np.sqrt(2)) # Probability of Superiority
        proportion_of_overlap_dav = 2 * norm.cdf(-abs(cohens_dav) / 2)

        # 4. Based on Hedges' gs
        cohens_U2_gs = (norm.cdf(abs(hedges_gs)/2))
        cohens_U1_gs=  (2*abs(cohens_U2_gs) - 1) / cohens_U2_gs
        cohens_U3_gs = norm.cdf(abs(hedges_gs))            
        Mcgraw_Wong_CLgs = norm.cdf(hedges_gs / np.sqrt(2)) # Probability of Superiority
        proportion_of_overlap_gs = 2 * norm.cdf(-abs(hedges_gs) / 2)

        # 5. Based on Hedges' gav
        cohens_U2_gav = (norm.cdf(abs(hedges_gav)/2))
        cohens_U1_gav =  (2*abs(cohens_U2_gav) - 1) / cohens_U2_gav
        cohens_U3_gav = norm.cdf(abs(hedges_gav))            
        Mcgraw_Wong_CLgav = norm.cdf(hedges_gav / np.sqrt(2)) # Probability of Superiority
        proportion_of_overlap_gav = 2 * norm.cdf(-abs(hedges_gav) / 2)

        # 6. Central CI's
        ci_lower_cohens_ds_Central, ci_upper_cohens_ds_Central, standard_error_cohens_ds =  calculate_Central_ci_from_cohens_d_two_indpednent_sample_t_test (cohens_ds, sample_size_1, sample_size_2, confidence_level)
        ci_lower_cohens_dpop_Central, ci_upper_cohens_dpop_Central, standard_error_cohens_dpop =  calculate_Central_ci_from_cohens_d_two_indpednent_sample_t_test (cohens_dpop, sample_size_1, sample_size_2, confidence_level)
        ci_lower_cohens_dav_Central, ci_upper_cohens_dav_Central, standard_error_cohens_dav =  calculate_Central_ci_from_cohens_d_two_indpednent_sample_t_test (cohens_dav, sample_size_1, sample_size_2, confidence_level)
        ci_lower_hedges_gs_Central, ci_upper_hedges_gs_Central, standard_error_hedges_gs =  calculate_Central_ci_from_cohens_d_two_indpednent_sample_t_test (hedges_gs, sample_size_1, sample_size_2, confidence_level)
        ci_lower_hedges_gav_Central, ci_upper_hedges_gav_Central, standard_error_hedges_gav =  calculate_Central_ci_from_cohens_d_two_indpednent_sample_t_test (hedges_gav, sample_size_1, sample_size_2, confidence_level)

        # 7. Non-Central CI's
        constant = np.sqrt((sample_size_1 + sample_size_2)/(sample_size_1 * sample_size_2))
        NCT_ci_lower_Pivotal_ds, NCT_ci_upper_Pivotal_ds =  Pivotal_ci_t (t_score, df, sample_size, confidence_level)
        ci_lower_Pivotal_ds, ci_upper_Pivotal_ds = NCT_ci_lower_Pivotal_ds * constant,  NCT_ci_upper_Pivotal_ds * constant
        ci_lower_Pivotal_gs, ci_upper_Pivotal_gs = ci_lower_Pivotal_ds * correction, ci_upper_Pivotal_ds * correction

        NCT_ci_lower_Pivotal_dav, NCT_ci_upper_Pivotal_dav =  Pivotal_ci_t (t_score_av, df, sample_size, confidence_level)
        ci_lower_Pivotal_dav, ci_upper_Pivotal_dav =  NCT_ci_lower_Pivotal_dav * constant, NCT_ci_upper_Pivotal_dav * constant
        ci_lower_Pivotal_gav, ci_upper_Pivotal_gav = ci_lower_Pivotal_dav * correction, ci_upper_Pivotal_dav * correction

        NCT_ci_lower_Pivotal_dpop, NCT_ci_upper_Pivotal_dpop = ci_lower_Pivotal_ds * np.sqrt((df/sample_size)), ci_upper_Pivotal_ds * np.sqrt((df/sample_size))
        ci_lower_Pivotal_dpop, ci_upper_Pivotal_dpop =  NCT_ci_lower_Pivotal_dpop * constant, NCT_ci_upper_Pivotal_dpop * constant


        # Set results
        results = {}
        
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["t-score"] = np.around(t_score, 4)

        # Parametric Effect Sizes Based on the Distribution of Cohen's ds
        results["Cohen's U1_ds"] = np.around(cohens_U1_ds*100, 4) 
        results["Lower Central CI U1_ds"] = ci_lower_cohens_U1_ds_Central = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2)))) > (((2 * (norm.cdf((ci_upper_cohens_ds_Central) / 2)) - 1) / (norm.cdf((ci_upper_cohens_ds_Central) / 2)))) else (((2 * (norm.cdf((ci_lower_cohens_ds_Central) / 2)) - 1) / (norm.cdf((ci_lower_cohens_ds_Central) / 2))))),0),3) # type: ignore
        results["Upper Central CI U1_ds"] = ci_upper_cohens_U1_ds_Central = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_cohens_ds_Central) / 2)) - 1) / (norm.cdf(abs(ci_lower_cohens_ds_Central) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)))))),100),3) # type: ignore
        results["Lower Pivotal CI U1_ds"] = ci_lower_cohens_U1_ds_Pivotal = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)))) > (((2 * (norm.cdf((ci_upper_Pivotal_ds) / 2)) - 1) / (norm.cdf((ci_upper_Pivotal_ds) / 2)))) else (((2 * (norm.cdf((ci_lower_Pivotal_ds) / 2)) - 1) / (norm.cdf((ci_lower_Pivotal_ds) / 2))))),0),3) # type: ignore
        results["Upper Pivotal CI U1_ds"] = ci_upper_cohens_U1_ds_Pivotal = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_Pivotal_ds) / 2)) - 1) / (norm.cdf(abs(ci_lower_Pivotal_ds) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)))))),100),3) # type: ignore
        results["Statistical Line Cohen's U1_ds"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2081d\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U1_ds * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U1_ds_Pivotal, 3),  np.around(ci_upper_cohens_U1_ds_Pivotal, 3))

        results["Cohen's U2_ds"] = np.around(cohens_U2_ds*100, 4)
        results["Lower Central CI U2_ds"] = ci_lower_cohens_U2_ds_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_cohens_ds_Central) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2))  ) )) > ((( (norm.cdf((ci_upper_cohens_ds_Central) / 2))  ) )) else ((( (norm.cdf((ci_lower_cohens_ds_Central) / 2))  ) ))),50),3) # type: ignore
        results["Upper Central CI U2_ds"] = ci_upper_cohens_U2_ds_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_cohens_ds_Central) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U2_ds"] = ci_lower_cohens_U2_ds_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_ds) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_ds) / 2))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_ds) / 2))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_ds) / 2))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U2_ds"] = ci_upper_cohens_U2_ds_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_ds) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_ds) / 2))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U2_ds"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2082d\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U2_ds * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U2_ds_Pivotal, 3),  np.around(ci_upper_cohens_U2_ds_Pivotal, 3))

        results["Cohen's U3_ds"] = np.around(cohens_U3_ds*100, 4)
        results["Lower Central CI U3_ds"] = ci_lower_cohens_U3_ds_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_cohens_ds_Central)))  ) )) if ((( (norm.cdf(abs(ci_upper_cohens_ds_Central)  ))  ) )) > ((( (norm.cdf((ci_upper_cohens_ds_Central)  ))  ) )) else ((( (norm.cdf((ci_lower_cohens_ds_Central)  ))  ) ))),50),3) # type: ignore
        results["Upper Central CI U3_ds"] = ci_upper_cohens_U3_ds_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_cohens_ds_Central) ))  ) )), ((( (norm.cdf(abs(ci_upper_cohens_ds_Central)  ))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U3_ds"] = ci_lower_cohens_U3_ds_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_ds)))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_ds)  ))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_ds)  ))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_ds)  ))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U3_ds"] = ci_upper_cohens_U3_ds_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_ds)))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_ds)  ))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U3_ds"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2083d\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U3_ds * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U3_ds_Pivotal, 3),  np.around(ci_upper_cohens_U3_ds_Pivotal, 3))

        results["Mcgraw and Wong, CLds"] = np.around(100 * (Mcgraw_Wong_CLds),4)
        results["Lower Central Ci Mcgraw and Wong, CLds"] = ci_lower_clds_Central = np.around(100 * (max(norm.cdf(ci_lower_cohens_ds_Central / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Central Ci Mcgraw and Wong, CLds"] = ci_upper_clds_Central = np.around(100 * (min(norm.cdf(ci_upper_cohens_ds_Central / np.sqrt(2)),1)),4) # type: ignore
        results["Lower Pivotal Ci Mcgraw and Wong, CLds"] = ci_lower_clds_Pivotal = np.around(100 * (max(norm.cdf(ci_lower_Pivotal_ds / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Pivotal Ci Mcgraw and Wong, CLds"] = ci_upper_clds_Pivotal = np.around(100 * (min(norm.cdf(ci_upper_Pivotal_ds / np.sqrt(2)),1)),4) # type: ignore
        results["Statistical Line Mcgraw and Wong, CLds"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLd\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(Mcgraw_Wong_CLds * 100),1), confidence_level_percentages , np.around(ci_lower_clds_Pivotal, 3),  np.around(ci_upper_clds_Pivotal, 3))

        results["Proportion Of Overlap ds"] = np.around(proportion_of_overlap_ds, 4)
        results["Lower Central CI Proportion of Overlap_ds"] = ci_lower_pov_Central_ds = np.around((2 * norm.cdf(-abs(ci_lower_cohens_ds_Central) / 2)), 4) # type: ignore
        results["Upper Central CI Proportion of Overlap_ds"] = ci_upper_pov_Central_ds = np.around((2 * norm.cdf(-abs(ci_upper_cohens_ds_Central) / 2)), 4) # type: ignore
        results["Lower Pivotal CI Proportion of Overlap_ds"] = ci_lower_pov_Pivotal_ds = np.around((2 * norm.cdf(-abs(ci_lower_Pivotal_ds) / 2)), 4) # type: ignore
        results["Upper Pivotal CI Proportion of Overlap_ds"] = ci_upper_pov_Pivotal_ds = np.around((2 * norm.cdf(-abs(ci_upper_Pivotal_ds) / 2)), 4) # type: ignore
        results["Statistical Line Proportion of Overlap (POVds) "] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, POVd\u209B = {:.3f}%, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(proportion_of_overlap_ds),3), confidence_level_percentages , np.around(ci_lower_pov_Pivotal_ds, 3),  np.around(ci_upper_pov_Pivotal_ds, 3))


        # Parametric Effect Sizes Based on the Distribution of Hedge's gs
        results["Cohen's U1_gs"] = np.around(cohens_U1_gs*100, 4) 
        results["Lower Central CI U1_gs"] = ci_lower_cohens_U1_gs_Central = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2)))) > (((2 * (norm.cdf((ci_upper_hedges_gs_Central) / 2)) - 1) / (norm.cdf((ci_upper_hedges_gs_Central) / 2)))) else (((2 * (norm.cdf((ci_lower_hedges_gs_Central) / 2)) - 1) / (norm.cdf((ci_lower_hedges_gs_Central) / 2))))),0),3) # type: ignore
        results["Upper Central CI U1_gs"] = ci_upper_cohens_U1_gs_Central = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_hedges_gs_Central) / 2)) - 1) / (norm.cdf(abs(ci_lower_hedges_gs_Central) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)))))),100),3) # type: ignore
        results["Lower Pivotal CI U1_gs"] = ci_lower_cohens_U1_gs_Pivotal = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)))) > (((2 * (norm.cdf((ci_upper_Pivotal_gs) / 2)) - 1) / (norm.cdf((ci_upper_Pivotal_gs) / 2)))) else (((2 * (norm.cdf((ci_lower_Pivotal_gs) / 2)) - 1) / (norm.cdf((ci_lower_Pivotal_gs) / 2))))),0),3) # type: ignore
        results["Upper Pivotal CI U1_gs"] = ci_upper_cohens_U1_gs_Pivotal = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_Pivotal_gs) / 2)) - 1) / (norm.cdf(abs(ci_lower_Pivotal_gs) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)))))),100),3) # type: ignore
        results["Statistical Line Cohen's U1_gs"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2081g\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U1_gs * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U1_gs_Pivotal, 3),  np.around(ci_upper_cohens_U1_gs_Pivotal, 3))

        results["Cohen's U2_gs"] = np.around(cohens_U2_gs*100, 4)
        results["Lower Central CI U2_gs"] = ci_lower_cohens_U2_gs_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_hedges_gs_Central) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2))  ) )) > ((( (norm.cdf((ci_upper_hedges_gs_Central) / 2))  ) )) else ((( (norm.cdf((ci_lower_hedges_gs_Central) / 2))  ) ))),50),3) # type: ignore
        results["Upper Central CI U2_gs"] = ci_upper_cohens_U2_gs_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_hedges_gs_Central) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U2_gs"] = ci_lower_cohens_U2_gs_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_gs) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_gs) / 2))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_gs) / 2))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_gs) / 2))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U2_gs"] = ci_upper_cohens_U2_gs_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_gs) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_gs) / 2))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U2_gs"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2082g\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U2_gs * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U2_gs_Pivotal, 3),  np.around(ci_upper_cohens_U2_gs_Pivotal, 3))

        results["Cohen's U3_gs"] = np.around(cohens_U3_gs*100, 4)
        results["Lower Central CI U3_gs"] = ci_lower_cohens_U3_gs_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_hedges_gs_Central)))  ) )) if ((( (norm.cdf(abs(ci_upper_hedges_gs_Central)  ))  ) )) > ((( (norm.cdf((ci_upper_hedges_gs_Central)  ))  ) )) else ((( (norm.cdf((ci_lower_hedges_gs_Central)  ))  ) ))),50),3) # type: ignore
        results["Upper Central CI U3_gs"] = ci_upper_cohens_U3_gs_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_hedges_gs_Central) ))  ) )), ((( (norm.cdf(abs(ci_upper_hedges_gs_Central)  ))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U3_gs"] = ci_lower_cohens_U3_gs_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_gs)))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_gs)  ))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_gs)  ))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_gs)  ))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U3_gs"] = ci_upper_cohens_U3_gs_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_gs)))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_gs)  ))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U3_gs"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2083g\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U3_gs * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U3_gs_Pivotal, 3),  np.around(ci_upper_cohens_U3_gs_Pivotal, 3))

        results["Mcgraw and Wong, CLgs"] = np.around(100 * (Mcgraw_Wong_CLgs),4)
        results["Lower Central Ci Mcgraw and Wong, CLgs"] = ci_lower_clgs_Central = np.around(100 * (max(norm.cdf(ci_lower_hedges_gs_Central / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Central Ci Mcgraw and Wong, CLgs"] = ci_upper_clgs_Central = np.around(100 * (min(norm.cdf(ci_upper_hedges_gs_Central / np.sqrt(2)),1)),4) # type: ignore
        results["Lower Pivotal Ci Mcgraw and Wong, CLgs"] = ci_lower_clgs_Pivotal = np.around(100 * (max(norm.cdf(ci_lower_Pivotal_gs / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Pivotal Ci Mcgraw and Wong, CLgs"] = ci_upper_clgs_Pivotal = np.around(100 * (min(norm.cdf(ci_upper_Pivotal_gs / np.sqrt(2)),1)),4) # type: ignore
        results["Statistical Line Mcgraw and Wong, CLgs"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLg\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(Mcgraw_Wong_CLgs * 100),1), confidence_level_percentages , np.around(ci_lower_clgs_Pivotal, 3),  np.around(ci_upper_clgs_Pivotal, 3))

        results["Proportion Of Overlap gs"] = np.around(proportion_of_overlap_gs, 4)
        results["Lower Central CI Proportion of Overlap_gs"] = ci_lower_pov_Central_gs = np.around((2 * norm.cdf(-abs(ci_lower_hedges_gs_Central) / 2)), 4) # type: ignore
        results["Upper Central CI Proportion of Overlap_gs"] = ci_upper_pov_Central_gs = np.around((2 * norm.cdf(-abs(ci_upper_hedges_gs_Central) / 2)), 4) # type: ignore
        results["Lower Pivotal CI Proportion of Overlap_gs"] = ci_lower_pov_Pivotal_gs = np.around((2 * norm.cdf(-abs(ci_lower_Pivotal_gs) / 2)), 4) # type: ignore
        results["Upper Pivotal CI Proportion of Overlap_gs"] = ci_upper_pov_Pivotal_gs = np.around((2 * norm.cdf(-abs(ci_upper_Pivotal_gs) / 2)), 4) # type: ignore
        results["Statistical Line Proportion of Overlap (POVgs) "] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, POVg\u209B = {:.3f}%, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(proportion_of_overlap_gs),3), confidence_level_percentages , np.around(ci_lower_pov_Pivotal_gs, 3),  np.around(ci_upper_pov_Pivotal_gs, 3))


        # Parametric Effect Sizes Based on the Distribution of Cohen's dav
        results["Cohen's U1_dav"] = np.around(cohens_U1_dav*100, 4) 
        results["Lower Central CI U1_dav"] = ci_lower_cohens_U1_dav_Central = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_cohens_dav_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_cohens_dav_Central) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_cohens_dav_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_cohens_dav_Central) / 2)))) > (((2 * (norm.cdf((ci_upper_cohens_dav_Central) / 2)) - 1) / (norm.cdf((ci_upper_cohens_dav_Central) / 2)))) else (((2 * (norm.cdf((ci_lower_cohens_dav_Central) / 2)) - 1) / (norm.cdf((ci_lower_cohens_dav_Central) / 2))))),0),3) # type: ignore
        results["Upper Central CI U1_dav"] = ci_upper_cohens_U1_dav_Central = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_cohens_dav_Central) / 2)) - 1) / (norm.cdf(abs(ci_lower_cohens_dav_Central) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_dav) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dav) / 2)))))),100),3) # type: ignore
        results["Lower Pivotal CI U1_dav"] = ci_lower_cohens_U1_dav_Pivotal = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_Pivotal_dav) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dav) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_Pivotal_dav) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dav) / 2)))) > (((2 * (norm.cdf((ci_upper_Pivotal_dav) / 2)) - 1) / (norm.cdf((ci_upper_Pivotal_dav) / 2)))) else (((2 * (norm.cdf((ci_lower_Pivotal_dav) / 2)) - 1) / (norm.cdf((ci_lower_Pivotal_dav) / 2))))),0),3) # type: ignore
        results["Upper Pivotal CI U1_dav"] = ci_upper_cohens_U1_dav_Pivotal = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_Pivotal_dav) / 2)) - 1) / (norm.cdf(abs(ci_lower_Pivotal_dav) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_dav) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dav) / 2)))))),100),3) # type: ignore
        results["Statistical Line Cohen's U1_dav"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2081d\u2090\u1d65 = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U1_dav * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U1_dav_Pivotal, 3),  np.around(ci_upper_cohens_U1_dav_Pivotal, 3))

        results["Cohen's U2_dav"] = np.around(cohens_U2_dav*100, 4)
        results["Lower Central CI U2_dav"] = ci_lower_cohens_U2_dav_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_cohens_dav_Central) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_cohens_dav_Central) / 2))  ) )) > ((( (norm.cdf((ci_upper_cohens_dav_Central) / 2))  ) )) else ((( (norm.cdf((ci_lower_cohens_dav_Central) / 2))  ) ))),50),3) # type: ignore
        results["Upper Central CI U2_dav"] = ci_upper_cohens_U2_dav_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_cohens_dav_Central) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_cohens_dav_Central) / 2))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U2_dav"] = ci_lower_cohens_U2_dav_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_dav) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_dav) / 2))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_dav) / 2))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_dav) / 2))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U2_dav"] = ci_upper_cohens_U2_dav_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_dav) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_dav) / 2))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U2_dav"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2082d\u2090\u1d65 = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U2_dav * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U2_dav_Pivotal, 3),  np.around(ci_upper_cohens_U2_dav_Pivotal, 3))

        results["Cohen's U3_dav"] = np.around(cohens_U3_dav*100, 4)
        results["Lower Central CI U3_dav"] = ci_lower_cohens_U3_dav_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_cohens_dav_Central)))  ) )) if ((( (norm.cdf(abs(ci_upper_cohens_dav_Central)  ))  ) )) > ((( (norm.cdf((ci_upper_cohens_dav_Central)  ))  ) )) else ((( (norm.cdf((ci_lower_cohens_dav_Central)  ))  ) ))),50),3) # type: ignore
        results["Upper Central CI U3_dav"] = ci_upper_cohens_U3_dav_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_cohens_dav_Central) ))  ) )), ((( (norm.cdf(abs(ci_upper_cohens_dav_Central)  ))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U3_dav"] = ci_lower_cohens_U3_dav_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_dav)))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_dav)  ))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_dav)  ))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_dav)  ))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U3_dav"] = ci_upper_cohens_U3_dav_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_dav)))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_dav)  ))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U3_dav"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2083d\u2090\u1d65 = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U3_dav * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U3_dav_Pivotal, 3),  np.around(ci_upper_cohens_U3_dav_Pivotal, 3))

        results["Mcgraw and Wong, CLdav"] = np.around(100 * (Mcgraw_Wong_CLdav),4)
        results["Lower Central Ci Mcgraw and Wong, CLdav"] = ci_lower_cldav_Central = np.around(100 * (max(norm.cdf(ci_lower_cohens_dav_Central / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Central Ci Mcgraw and Wong, CLdav"] = ci_upper_cldav_Central = np.around(100 * (min(norm.cdf(ci_upper_cohens_dav_Central / np.sqrt(2)),1)),4) # type: ignore
        results["Lower Pivotal Ci Mcgraw and Wong, CLdav"] = ci_lower_cldav_Pivotal = np.around(100 * (max(norm.cdf(ci_lower_Pivotal_dav / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Pivotal Ci Mcgraw and Wong, CLdav"] = ci_upper_cldav_Pivotal = np.around(100 * (min(norm.cdf(ci_upper_Pivotal_dav / np.sqrt(2)),1)),4) # type: ignore
        results["Statistical Line Mcgraw and Wong, CLdav"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLd\u2090\u1d65 = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(Mcgraw_Wong_CLdav * 100),1), confidence_level_percentages , np.around(ci_lower_cldav_Pivotal, 3),  np.around(ci_upper_cldav_Pivotal, 3))

        results["Proportion Of Overlap dav"] = np.around(proportion_of_overlap_dav, 4)
        results["Lower Central CI Proportion of Overlap_dav"] = ci_lower_pov_Central_dav = np.around((2 * norm.cdf(-abs(ci_lower_cohens_dav_Central) / 2)), 4) # type: ignore
        results["Upper Central CI Proportion of Overlap_dav"] = ci_upper_pov_Central_dav = np.around((2 * norm.cdf(-abs(ci_upper_cohens_dav_Central) / 2)), 4) # type: ignore
        results["Lower Pivotal CI Proportion of Overlap_dav"] = ci_lower_pov_Pivotal_dav = np.around((2 * norm.cdf(-abs(ci_lower_Pivotal_dav) / 2)), 4) # type: ignore
        results["Upper Pivotal CI Proportion of Overlap_dav"] = ci_upper_pov_Pivotal_dav = np.around((2 * norm.cdf(-abs(ci_upper_Pivotal_dav) / 2)), 4) # type: ignore
        results["Statistical Line Proportion of Overlap (POVdav) "] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, POVd\u2090\u1d65 = {:.3f}%, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(proportion_of_overlap_dav),3), confidence_level_percentages , np.around(ci_lower_pov_Pivotal_dav, 3),  np.around(ci_upper_pov_Pivotal_dav, 3))

        # Parametric Effect Sizes Based on the Distribution of Cohen's gav
        results["Cohen's U1_gav"] = np.around(cohens_U1_gav*100, 4) 
        results["Lower Central CI U1_gav"] = ci_lower_cohens_U1_gav_Central = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_hedges_gav_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_hedges_gav_Central) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_hedges_gav_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_hedges_gav_Central) / 2)))) > (((2 * (norm.cdf((ci_upper_hedges_gav_Central) / 2)) - 1) / (norm.cdf((ci_upper_hedges_gav_Central) / 2)))) else (((2 * (norm.cdf((ci_lower_hedges_gav_Central) / 2)) - 1) / (norm.cdf((ci_lower_hedges_gav_Central) / 2))))),0),3) # type: ignore
        results["Upper Central CI U1_gav"] = ci_upper_cohens_U1_gav_Central = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_hedges_gav_Central) / 2)) - 1) / (norm.cdf(abs(ci_lower_hedges_gav_Central) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_gav) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gav) / 2)))))),100),3) # type: ignore
        results["Lower Pivotal CI U1_gav"] = ci_lower_cohens_U1_gav_Pivotal = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_Pivotal_gav) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gav) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_Pivotal_gav) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gav) / 2)))) > (((2 * (norm.cdf((ci_upper_Pivotal_gav) / 2)) - 1) / (norm.cdf((ci_upper_Pivotal_gav) / 2)))) else (((2 * (norm.cdf((ci_lower_Pivotal_gav) / 2)) - 1) / (norm.cdf((ci_lower_Pivotal_gav) / 2))))),0),3) # type: ignore
        results["Upper Pivotal CI U1_gav"] = ci_upper_cohens_U1_gav_Pivotal = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_Pivotal_gav) / 2)) - 1) / (norm.cdf(abs(ci_lower_Pivotal_gav) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_gav) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gav) / 2)))))),100),3) # type: ignore
        results["Statistical Line Cohen's U1_gav"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2081g\u2090\u1d65 = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U1_gav * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U1_gav_Pivotal, 3),  np.around(ci_upper_cohens_U1_gav_Pivotal, 3))

        results["Cohen's U2_gav"] = np.around(cohens_U2_gav*100, 4)
        results["Lower Central CI U2_gav"] = ci_lower_cohens_U2_gav_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_hedges_gav_Central) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_hedges_gav_Central) / 2))  ) )) > ((( (norm.cdf((ci_upper_hedges_gav_Central) / 2))  ) )) else ((( (norm.cdf((ci_lower_hedges_gav_Central) / 2))  ) ))),50),3) # type: ignore
        results["Upper Central CI U2_gav"] = ci_upper_cohens_U2_gav_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_hedges_gav_Central) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_hedges_gav_Central) / 2))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U2_gav"] = ci_lower_cohens_U2_gav_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_gav) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_gav) / 2))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_gav) / 2))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_gav) / 2))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U2_gav"] = ci_upper_cohens_U2_gav_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_gav) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_gav) / 2))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U2_gav"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2082g\u2090\u1d65 = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U2_gav * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U2_gav_Pivotal, 3),  np.around(ci_upper_cohens_U2_gav_Pivotal, 3))

        results["Cohen's U3_gav"] = np.around(cohens_U3_gav*100, 4)
        results["Lower Central CI U3_gav"] = ci_lower_cohens_U3_gav_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_hedges_gav_Central)))  ) )) if ((( (norm.cdf(abs(ci_upper_hedges_gav_Central)  ))  ) )) > ((( (norm.cdf((ci_upper_hedges_gav_Central)  ))  ) )) else ((( (norm.cdf((ci_lower_hedges_gav_Central)  ))  ) ))),50),3) # type: ignore
        results["Upper Central CI U3_gav"] = ci_upper_cohens_U3_gav_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_hedges_gav_Central) ))  ) )), ((( (norm.cdf(abs(ci_upper_hedges_gav_Central)  ))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U3_gav"] = ci_lower_cohens_U3_gav_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_gav)))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_gav)  ))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_gav)  ))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_gav)  ))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U3_gav"] = ci_upper_cohens_U3_gav_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_gav)))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_gav)  ))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U3_gav"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2083g\u2090\u1d65 = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U3_gav * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U3_gav_Pivotal, 3),  np.around(ci_upper_cohens_U3_gav_Pivotal, 3))

        results["Mcgraw and Wong, CLgav"] = np.around(100 * (Mcgraw_Wong_CLgav),4)
        results["Lower Central Ci Mcgraw and Wong, CLgav"] = ci_lower_clgav_Central = np.around(100 * (max(norm.cdf(ci_lower_hedges_gav_Central / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Central Ci Mcgraw and Wong, CLgav"] = ci_upper_clgav_Central = np.around(100 * (min(norm.cdf(ci_upper_hedges_gav_Central / np.sqrt(2)),1)),4) # type: ignore
        results["Lower Pivotal Ci Mcgraw and Wong, CLgav"] = ci_lower_clgav_Pivotal = np.around(100 * (max(norm.cdf(ci_lower_Pivotal_gav / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Pivotal Ci Mcgraw and Wong, CLgav"] = ci_upper_clgav_Pivotal = np.around(100 * (min(norm.cdf(ci_upper_Pivotal_gav / np.sqrt(2)),1)),4) # type: ignore
        results["Statistical Line Mcgraw and Wong, CLgav"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLg\u2090\u1d65 = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(Mcgraw_Wong_CLgav * 100),1), confidence_level_percentages , np.around(ci_lower_clgav_Pivotal, 3),  np.around(ci_upper_clgav_Pivotal, 3))

        results["Proportion Of Overlap gav"] = np.around(proportion_of_overlap_gav, 4)
        results["Lower Central CI Proportion of Overlap_gav"] = ci_lower_pov_Central_gav = np.around((2 * norm.cdf(-abs(ci_lower_hedges_gav_Central) / 2)), 4) # type: ignore
        results["Upper Central CI Proportion of Overlap_gav"] = ci_upper_pov_Central_gav = np.around((2 * norm.cdf(-abs(ci_upper_hedges_gav_Central) / 2)), 4) # type: ignore
        results["Lower Pivotal CI Proportion of Overlap_gav"] = ci_lower_pov_Pivotal_gav = np.around((2 * norm.cdf(-abs(ci_lower_Pivotal_gav) / 2)), 4) # type: ignore
        results["Upper Pivotal CI Proportion of Overlap_gav"] = ci_upper_pov_Pivotal_gav = np.around((2 * norm.cdf(-abs(ci_upper_Pivotal_gav) / 2)), 4) # type: ignore
        results["Statistical Line Proportion of Overlap (POVgav) "] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, POVg\u2090\u1d65 = {:.3f}%, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(proportion_of_overlap_gav),3), confidence_level_percentages , np.around(ci_lower_pov_Pivotal_gav, 3),  np.around(ci_upper_pov_Pivotal_gav, 3))

        # Parametric Effect Sizes Based on the Distribution of Cohen's dpop
        results["Cohen's U1_dpop"] = np.around(cohens_U1_dpop*100, 4) 
        results["Lower Central CI U1_dpop"] = ci_lower_cohens_U1_dpop_Central = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2)))) > (((2 * (norm.cdf((ci_upper_cohens_dpop_Central) / 2)) - 1) / (norm.cdf((ci_upper_cohens_dpop_Central) / 2)))) else (((2 * (norm.cdf((ci_lower_cohens_dpop_Central) / 2)) - 1) / (norm.cdf((ci_lower_cohens_dpop_Central) / 2))))),0),3) # type: ignore
        results["Upper Central CI U1_dpop"] = ci_upper_cohens_U1_dpop_Central = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_cohens_dpop_Central) / 2)) - 1) / (norm.cdf(abs(ci_lower_cohens_dpop_Central) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)))))),100),3) # type: ignore
        results["Lower Pivotal CI U1_dpop"] = ci_lower_cohens_U1_dpop_Pivotal = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)))) > (((2 * (norm.cdf((ci_upper_Pivotal_dpop) / 2)) - 1) / (norm.cdf((ci_upper_Pivotal_dpop) / 2)))) else (((2 * (norm.cdf((ci_lower_Pivotal_dpop) / 2)) - 1) / (norm.cdf((ci_lower_Pivotal_dpop) / 2))))),0),3) # type: ignore
        results["Upper Pivotal CI U1_dpop"] = ci_upper_cohens_U1_dpop_Pivotal = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_Pivotal_dpop) / 2)) - 1) / (norm.cdf(abs(ci_lower_Pivotal_dpop) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)))))),100),3) # type: ignore
        results["Statistical Line Cohen's U1_dpop"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2081d\u209a\u2092\u209a = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U1_dpop * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U1_dpop_Pivotal, 3),  np.around(ci_upper_cohens_U1_dpop_Pivotal, 3))

        results["Cohen's U2_dpop"] = np.around(cohens_U2_dpop*100, 4)
        results["Lower Central CI U2_dpop"] = ci_lower_cohens_U2_dpop_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2))  ) )) > ((( (norm.cdf((ci_upper_cohens_dpop_Central) / 2))  ) )) else ((( (norm.cdf((ci_lower_cohens_dpop_Central) / 2))  ) ))),50),3) # type: ignore
        results["Upper Central CI U2_dpop"] = ci_upper_cohens_U2_dpop_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_cohens_dpop_Central) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U2_dpop"] = ci_lower_cohens_U2_dpop_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_dpop) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_dpop) / 2))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_dpop) / 2))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U2_dpop"] = ci_upper_cohens_U2_dpop_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_dpop) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U2_dpop"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2082d\u209a\u2092\u209a = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U2_dpop * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U2_dpop_Pivotal, 3),  np.around(ci_upper_cohens_U2_dpop_Pivotal, 3))

        results["Cohen's U3_dpop"] = np.around(cohens_U3_dpop*100, 4)
        results["Lower Central CI U3_dpop"] = ci_lower_cohens_U3_dpop_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_cohens_dpop_Central)))  ) )) if ((( (norm.cdf(abs(ci_upper_cohens_dpop_Central)  ))  ) )) > ((( (norm.cdf((ci_upper_cohens_dpop_Central)  ))  ) )) else ((( (norm.cdf((ci_lower_cohens_dpop_Central)  ))  ) ))),50),3) # type: ignore
        results["Upper Central CI U3_dpop"] = ci_upper_cohens_U3_dpop_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_cohens_dpop_Central) ))  ) )), ((( (norm.cdf(abs(ci_upper_cohens_dpop_Central)  ))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U3_dpop"] = ci_lower_cohens_U3_dpop_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_dpop)))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_dpop)  ))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_dpop)  ))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_dpop)  ))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U3_dpop"] = ci_upper_cohens_U3_dpop_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_dpop)))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_dpop)  ))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U3_dpop"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2083d\u209a\u2092\u209a = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U3_dpop * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U3_dpop_Pivotal, 3),  np.around(ci_upper_cohens_U3_dpop_Pivotal, 3))

        results["Mcgraw and Wong, CLdpop"] = np.around(100 * (Mcgraw_Wong_CLdpop),4)
        results["Lower Central Ci Mcgraw and Wong, CLdpop"] = ci_lower_cldpop_Central = np.around(100 * (max(norm.cdf(ci_lower_cohens_dpop_Central / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Central Ci Mcgraw and Wong, CLdpop"] = ci_upper_cldpop_Central = np.around(100 * (min(norm.cdf(ci_upper_cohens_dpop_Central / np.sqrt(2)),1)),4) # type: ignore
        results["Lower Pivotal Ci Mcgraw and Wong, CLdpop"] = ci_lower_cldpop_Pivotal = np.around(100 * (max(norm.cdf(ci_lower_Pivotal_dpop / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Pivotal Ci Mcgraw and Wong, CLdpop"] = ci_upper_cldpop_Pivotal = np.around(100 * (min(norm.cdf(ci_upper_Pivotal_dpop / np.sqrt(2)),1)),4) # type: ignore
        results["Statistical Line Mcgraw and Wong, CLdpop"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLd\u209a\u2092\u209a = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(Mcgraw_Wong_CLdpop * 100),1), confidence_level_percentages , np.around(ci_lower_cldpop_Pivotal, 3),  np.around(ci_upper_cldpop_Pivotal, 3))

        results["Proportion Of Overlap dpop"] = np.around(proportion_of_overlap_dpop, 4)
        results["Lower Central CI Proportion of Overlap_dpop"] = ci_lower_pov_Central_dpop = np.around((2 * norm.cdf(-abs(ci_lower_cohens_dpop_Central) / 2)), 4) # type: ignore
        results["Upper Central CI Proportion of Overlap_dpop"] = ci_upper_pov_Central_dpop = np.around((2 * norm.cdf(-abs(ci_upper_cohens_dpop_Central) / 2)), 4) # type: ignore
        results["Lower Pivotal CI Proportion of Overlap_dpop"] = ci_lower_pov_Pivotal_dpop = np.around((2 * norm.cdf(-abs(ci_lower_Pivotal_dpop) / 2)), 4) # type: ignore
        results["Upper Pivotal CI Proportion of Overlap_dpop"] = ci_upper_pov_Pivotal_dpop = np.around((2 * norm.cdf(-abs(ci_upper_Pivotal_dpop) / 2)), 4) # type: ignore
        results["Statistical Line Proportion of Overlap (POVdpop) "] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, POVd\u209a\u2092\u209a = {:.3f}%, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(proportion_of_overlap_dpop),3), confidence_level_percentages , np.around(ci_lower_pov_Pivotal_dpop, 3),  np.around(ci_upper_pov_Pivotal_dpop, 3))

        return results


    @staticmethod
    def Common_Language_Effect_Sizes_Independent_Samples_from_Data(params: dict) -> dict: # Note that from Parameters, ONnly the Parametric Effect Sizes Can be Calculated

        # Set params
        column_1 = params["column_1"]
        column_2 = params["column_2"]
        population_mean_diff = params["Difference in the Population"]
        confidence_level_percentages = params["Confidence Level"]
        reps = params["Number of Bootstrapping Samples"]
       
        
        # Calculation
        confidence_level = confidence_level_percentages / 100
        sample_size_1 = len(column_1)
        sample_size_2 = len(column_2)
        sample_mean_1 = np.mean(column_1)
        sample_mean_2 = np.mean(column_2)
        Sample_1_median = np.median(column_1)
        Sample_2_median = np.median(column_2)
        sample_sd_1 = np.std(column_1, ddof = 1)
        sample_sd_2 = np.std(column_2, ddof = 1)

        sample_size = sample_size_1 + sample_size_2
        df = sample_size - 2
        cohens_ds = (sample_mean_1 - sample_mean_2 - population_mean_diff)  / (np.sqrt((((sample_size_1-1)*(sample_sd_1**2)) + ((sample_size_2-1)*(sample_sd_2**2))) / df))
        cohens_dpop = cohens_ds / np.sqrt(df/(df+2))
        cohens_dav = (sample_mean_1 - sample_mean_2 -population_mean_diff) / np.sqrt((sample_sd_1**2 + sample_sd_2**2) / 2)
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
        hedges_gs = cohens_ds * correction
        hedges_gav = cohens_dav * correction
        standardizer_ds = np.sqrt(((((sample_size_1-1)*sample_sd_1**2)) + ((sample_size_2-1)*sample_sd_2**2)) / (sample_size-2)) # this is Spooled
        standard_error = standardizer_ds  * np.sqrt((sample_size_1+sample_size_2)/(sample_size_1*sample_size_2))        
        t_score = ((sample_mean_1 - sample_mean_2 - population_mean_diff) / standard_error)
        t_score_av = ((sample_mean_1 - sample_mean_2) - population_mean_diff ) / ((np.sqrt((sample_sd_1**2 + sample_sd_2**2)/2)) * np.sqrt((sample_size_1+sample_size_2)/(sample_size_1*sample_size_2))) 
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)

        # Parametric Common Language Effect Sizes
        #########################################

        # 1. Based on Cohen's ds
        cohens_U2_ds = (norm.cdf(abs(cohens_ds)/2))
        cohens_U1_ds =  (2* cohens_U2_ds - 1) / cohens_U2_ds
        cohens_U3_ds = norm.cdf(abs(cohens_ds))
        Mcgraw_Wong_CLds = norm.cdf(cohens_ds / np.sqrt(2))
        proportion_of_overlap_ds = 2 * norm.cdf(-abs(cohens_ds) / 2)

        # 2. Based on Cohen's dpop
        cohens_U2_dpop = (norm.cdf(abs(cohens_dpop)/2))
        cohens_U1_dpop =  (2*abs(cohens_U2_dpop) - 1) / cohens_U2_dpop
        cohens_U3_dpop = norm.cdf(abs(cohens_dpop))
        Mcgraw_Wong_CLdpop = norm.cdf(cohens_dpop / np.sqrt(2)) # See the suggestion of Ruscio for this effect size
        proportion_of_overlap_dpop = 2 * norm.cdf(-abs(cohens_dpop) / 2)

        # 3. Based on Cohen's dav
        cohens_U2_dav = (norm.cdf(abs(cohens_dav)/2))
        cohens_U1_dav =  (2*abs(cohens_U2_dav) - 1) / cohens_U2_dav
        cohens_U3_dav = norm.cdf(abs(cohens_dav))            
        Mcgraw_Wong_CLdav = norm.cdf(cohens_dav / np.sqrt(2)) # Probability of Superiority
        proportion_of_overlap_dav = 2 * norm.cdf(-abs(cohens_dav) / 2)

        # 4. Based in Hedges' gs
        cohens_U2_gs = (norm.cdf(abs(hedges_gs)/2))
        cohens_U1_gs=  (2*abs(cohens_U2_gs) - 1) / cohens_U2_gs
        cohens_U3_gs = norm.cdf(abs(hedges_gs))            
        Mcgraw_Wong_CLgs = norm.cdf(hedges_gs / np.sqrt(2)) # Probability of Superiority
        proportion_of_overlap_gs = 2 * norm.cdf(-abs(hedges_gs) / 2)

        # 5. Based in Hedges' gav
        cohens_U2_gav = (norm.cdf(abs(hedges_gav)/2))
        cohens_U1_gav =  (2*abs(cohens_U2_gav) - 1) / cohens_U2_gav
        cohens_U3_gav = norm.cdf(abs(hedges_gav))            
        Mcgraw_Wong_CLgav = norm.cdf(hedges_gav / np.sqrt(2)) # Probability of Superiority
        proportion_of_overlap_gav = 2 * norm.cdf(-abs(hedges_gav) / 2)

        # 6. Central CI's

        ci_lower_cohens_ds_Central, ci_upper_cohens_ds_Central, standard_error_cohens_ds =  calculate_Central_ci_from_cohens_d_two_indpednent_sample_t_test (cohens_ds, sample_size_1, sample_size_2, confidence_level)
        ci_lower_cohens_dpop_Central, ci_upper_cohens_dpop_Central, standard_error_cohens_dpop =  calculate_Central_ci_from_cohens_d_two_indpednent_sample_t_test (cohens_dpop, sample_size_1, sample_size_2, confidence_level)
        ci_lower_cohens_dav_Central, ci_upper_cohens_dav_Central, standard_error_cohens_dav =  calculate_Central_ci_from_cohens_d_two_indpednent_sample_t_test (cohens_dav, sample_size_1, sample_size_2, confidence_level)
        ci_lower_hedges_gs_Central, ci_upper_hedges_gs_Central, standard_error_hedges_gs =  calculate_Central_ci_from_cohens_d_two_indpednent_sample_t_test (hedges_gs, sample_size_1, sample_size_2, confidence_level)
        ci_lower_hedges_gav_Central, ci_upper_hedges_gav_Central, standard_error_hedges_gav =  calculate_Central_ci_from_cohens_d_two_indpednent_sample_t_test (hedges_gav, sample_size_1, sample_size_2, confidence_level)

        # 7. Non-Central CI's
        constant = np.sqrt((sample_size_1 + sample_size_2)/(sample_size_1 * sample_size_2))
        NCT_ci_lower_Pivotal_ds, NCT_ci_upper_Pivotal_ds =  Pivotal_ci_t (t_score, df, sample_size, confidence_level)
        ci_lower_Pivotal_ds, ci_upper_Pivotal_ds = NCT_ci_lower_Pivotal_ds * constant,  NCT_ci_upper_Pivotal_ds * constant
        ci_lower_Pivotal_gs, ci_upper_Pivotal_gs = ci_lower_Pivotal_ds * correction, ci_upper_Pivotal_ds * correction

        NCT_ci_lower_Pivotal_dav, NCT_ci_upper_Pivotal_dav =  Pivotal_ci_t (t_score_av, df, sample_size, confidence_level)
        ci_lower_Pivotal_dav, ci_upper_Pivotal_dav =  NCT_ci_lower_Pivotal_dav * constant, NCT_ci_upper_Pivotal_dav * constant
        ci_lower_Pivotal_gav, ci_upper_Pivotal_gav = ci_lower_Pivotal_dav * correction, ci_upper_Pivotal_dav * correction

        NCT_ci_lower_Pivotal_dpop, NCT_ci_upper_Pivotal_dpop = ci_lower_Pivotal_ds * np.sqrt((df/sample_size)), ci_upper_Pivotal_ds * np.sqrt((df/sample_size))
        ci_lower_Pivotal_dpop, ci_upper_Pivotal_dpop =  NCT_ci_lower_Pivotal_dpop * constant, NCT_ci_upper_Pivotal_dpop * constant



        # Aparametric Common Language Effect Sizes
        ##########################################

        # Preperation of Parameters for the Aparametric CLES
        Number_of_comparisons_between_x_and_y = sample_size_1 * sample_size_2
        Matrix = np.subtract.outer(column_1, column_2)
        Poistive_Compariosons = Matrix > 0
        nonties_comparisons = Matrix != 0 
        Superioirty_num = np.count_nonzero(Poistive_Compariosons) ## Proportions of cases group1 is larger
        Number_Of_Ties = (Number_of_comparisons_between_x_and_y - np.count_nonzero(nonties_comparisons)) # Amount of ties in the data
        proportion_of_ties = Number_Of_Ties / Number_of_comparisons_between_x_and_y
        Inferioirtiy_num = Number_of_comparisons_between_x_and_y - Superioirty_num - Number_Of_Ties # Proportions of cases group2 is larger

        # 1. Probability of Superiority not Considering Ties (Grissom & Kim)
        POS_Grissom_Kim = Superioirty_num / Number_of_comparisons_between_x_and_y

        # 2. Probability of Superiority Considerung Ties (Vargha & Delaney) 
        POS_Vargha_Delaney = (Superioirty_num + Number_Of_Ties *0.5) / Number_of_comparisons_between_x_and_y

        # 3. Probability of Superiority Based on Goodman and Krushkal Gamma (*Metsämuuronen 2023)
        x,y = np.concatenate([column_2, column_1]),np.concatenate([np.ones_like(column_2), np.zeros_like(column_1)]).astype(int) # Step 1 - Converting the data into long format (x is a binary variable of group and y is a continous dependent variable)

        Rx, Ry = np.sign(np.subtract.outer(y, y)), np.sign(np.subtract.outer(x, x)) # Step 2 - Calculating Goodman and Krushkal Gama and 
        Goodman_Krushkal_Gamma_Correlation = np.sum(np.multiply(Rx, Ry)) / np.sum(np.abs(np.multiply(Rx, Ry)))
        POS_Gamma_Based_Metsamuuronen = Goodman_Krushkal_Gamma_Correlation * 0.5 + 0.5 # Step 3 - Convert it to Probability of superiority based on Gamma

        # 4. Wilcoxon Mann Whitney Genralized Odds Ratio (Agresti)
        Generalizd_Odds_Ratio_Agresti = Superioirty_num/Inferioirtiy_num

        # 5. Wilcoxon Mann Whitney Generalized Odds Ratio Considering Ties (Obrien or Churilov)
        Generalizd_Odds_Ratio_Churilov = ((Superioirty_num  / Number_of_comparisons_between_x_and_y) + 0.5 * proportion_of_ties) /((Inferioirtiy_num / Number_of_comparisons_between_x_and_y) + 0.5 * proportion_of_ties)

        # 6. Cliffs Delta
        Cliffs_Delta = -1 * (1 - 2 * POS_Vargha_Delaney) # (This is the original version of Cliff's Delta)
        Cliffs_Delta_no_ties = -1 *(1 - 2 * POS_Grissom_Kim)
        
        # Other Common Language Effect Sizes and Bootstrapping CI's
        
        # 7. Kraemer & Andrews Gamma (Degree of Overlap)
        Number_of_cases_x_larger_than_median_y = sum(1 for val in column_1 if val > Sample_2_median)
        Aparametric_Cohens_U3_no_ties = (Number_of_cases_x_larger_than_median_y / sample_size)
        if Aparametric_Cohens_U3_no_ties == 0:   Aparametric_Cohens_U3_no_ties = 1 / (sample_size + 1)
        elif Aparametric_Cohens_U3_no_ties == 1: Aparametric_Cohens_U3_no_ties = sample_size / (sample_size + 1)
        Kraemer_Andrews_Gamma = norm.ppf(Aparametric_Cohens_U3_no_ties)
        
        # 8. Hentschke & Stüttgen U3 (Aparametric Version of U3)
        Number_of_cases_x_equal_to_median_y = sum(1 for val in column_1 if val == Sample_2_median)
        Hentschke_Stüttgen_U3 = ((Number_of_cases_x_larger_than_median_y + Number_of_cases_x_equal_to_median_y * 0.5) / sample_size) #This is a non Parametric U3 as it called by the authors and its the tied consideration version of K & A gamma
        if Sample_1_median == Sample_2_median: Hentschke_Stüttgen_U3 = 0.5

        # 9. Hentschke & Stüttgen U1 (Aparametric Version of U1)
        Number_of_cases_x_larger_than_maximum_y = sum(1 for val in column_1 if val > np.max(column_2))
        Number_of_cases_x_smaller_than_minimum_y = sum(1 for val in column_1 if val < np.min(column_2))
        Hentschke_Stüttgen_U1 = (Number_of_cases_x_larger_than_maximum_y + Number_of_cases_x_smaller_than_minimum_y) / sample_size

        # Wilcox and Musaka's Q
        # Computation of the Constants h
        h1 = max((1.2 * (np.percentile(column_1, 75) - np.percentile(column_1, 25))) / (sample_size_1 ** (1 / 5)), 0.05)
        h2 = max((1.2 * (np.percentile(column_2, 75) - np.percentile(column_2, 25))) / (sample_size_2 ** (1 / 5)), 0.05)

        eta1 = 0
        eta2 = 0

        for Value in column_1: 
            f_x1 = (np.sum(column_1 <= (Value + h1)) - np.sum(column_1 < (Value - h1))) / (2 * sample_size_1 * h1)
            f_x2 = (np.sum(column_2 <= (Value + h2)) - np.sum(column_2 < (Value - h2))) / (2 * sample_size_2 * h2)
            if f_x1 > f_x2: eta1 += 1

        for Value in column_2:
            f_x1 = (np.sum(column_1 <= (Value + h1)) - np.sum(column_1 < (Value - h1))) / (2 * sample_size_1 * h1)
            f_x2 = (np.sum(column_2 <= (Value + h2)) - np.sum(column_2 < (Value - h2))) / (2 * sample_size_2 * h2)
            if f_x2 > f_x1: eta2 += 1

        Wilcox_Musaka_Q = (eta1 + eta2) / (sample_size_1 + sample_size_2)


        # Bootstrapping CI's for other common language effect sizes
        Bootstrap_Samples_x = []
        Bootstrap_Samples_y = []

        for _ in range(reps):
            # Generate bootstrap samples
            sample_1_bootstrap = np.random.choice(column_1, len(column_1), replace=True)
            sample_2_bootstrap = np.random.choice(column_2, len(column_2), replace=True)
            Bootstrap_Samples_x.append(sample_1_bootstrap)

        # Confidence Intervals for K&A Gamma        
        Number_of_cases_x_larger_than_median_y_bootstrapping =(np.array([(np.sum(sample_x > Sample_2_median)) for sample_x in Bootstrap_Samples_x])) 
        Number_of_cases_x_larger_than_median_y_bootstrapping = (Number_of_cases_x_larger_than_median_y_bootstrapping/sample_size)
        Number_of_cases_x_larger_than_median_y_bootstrapping = np.where(Number_of_cases_x_larger_than_median_y_bootstrapping == 0, 1 / (sample_size + 1), Number_of_cases_x_larger_than_median_y_bootstrapping)
        Number_of_cases_x_larger_than_median_y_bootstrapping = np.where(Number_of_cases_x_larger_than_median_y_bootstrapping == 1, sample_size / (sample_size + 1), Number_of_cases_x_larger_than_median_y_bootstrapping)
        Kraemer_Andrews_Gamma_bootstrapping = norm.ppf(Number_of_cases_x_larger_than_median_y_bootstrapping)
        lower_ci_Kraemer_Andrews_Gamma_boot = np.percentile(Kraemer_Andrews_Gamma_bootstrapping, ((1 - confidence_level) - ((1 - confidence_level)/2)) *100)
        upper_ci_Kraemer_Andrews_Gamma_boot = np.percentile(Kraemer_Andrews_Gamma_bootstrapping, ((confidence_level) + ((1 - confidence_level)/2))*100)

        # Confidence Intervals for Hentschke & Stüttgen U3
        Hentschke_Stüttgen_U3_boot = []
        for sample_x in Bootstrap_Samples_x:
            Hentschke_Stüttgen_U3_boot.append((np.sum(sample_x > Sample_2_median) + np.sum(sample_x == Sample_2_median) * 0.5) / sample_size)
            if np.median(sample_x) == Sample_2_median: Hentschke_Stüttgen_U3_boot.append(0.5)
        lower_ci_Hentschke_Stüttgen_U3 = np.percentile(Hentschke_Stüttgen_U3_boot, ((1 - confidence_level) - ((1 - confidence_level)/2)) * 100)
        upper_ci_Hentschke_Stüttgen_U3 = np.percentile(Hentschke_Stüttgen_U3_boot, ((confidence_level) + ((1 - confidence_level)/2)) * 100)

        # Confidence Intervals for Hentschke & Stüttgen U1
        Number_of_cases_x_larger_than_max_y_bootstrapping =(np.array([(np.sum(sample_x > np.max(column_2))) for sample_x in Bootstrap_Samples_x])) 
        Number_of_cases_x_smaller_than_min_y_bootstrapping =(np.array([(np.sum(sample_x < np.min(column_2))) for sample_x in Bootstrap_Samples_x])) 
        Hentschke_Stüttgen_U1_boot = (Number_of_cases_x_larger_than_max_y_bootstrapping + Number_of_cases_x_smaller_than_min_y_bootstrapping) /sample_size
        lower_ci_Hentschke_Stüttgen_U1 = np.percentile(Hentschke_Stüttgen_U1_boot, ((1 - confidence_level) - ((1 - confidence_level)/2)) * 100)
        upper_ci_Hentschke_Stüttgen_U1 = np.percentile(Hentschke_Stüttgen_U1_boot, ((confidence_level) + ((1 - confidence_level)/2)) * 100)
 

        # Central Confidence Intervals for Aparmetric Effect Sizes
        ##############################
        #The t_value_for the CI's
        Critical_t_value = t.ppf(1 - ((1-confidence_level)/2) , sample_size-2)
        critical_z_value = norm.ppf(1 - ((1-confidence_level)/2))

        # 1. Probabilty of Superiority not considering ties CI (Grissom and Kim)
        # A. Traditional-Wald Type CI's: this is a wald type CI's reported in Grissom and Kim, 2001 p.141 or in Ruscio - called them the tradiational CI's
        Standard_Error_Ruscio = np.sqrt( (1/12) * ( (1/sample_size_1) + (1/sample_size_2) + (1/ (sample_size_1*sample_size_2))  ))
        ci_lower_Grissom_Kim_ruscio = POS_Grissom_Kim - (Standard_Error_Ruscio * Critical_t_value)
        ci_upper_Grissom_Kim_ruscio = POS_Grissom_Kim + (Standard_Error_Ruscio * Critical_t_value)

        # B. Cliff's Method (this method calcualte the CI for cliffs Delta and Converts them to PS)
        Cliffs_Delta_Binary_Matrix = np.subtract.outer(column_1, column_2)
        Cliffs_Delta_Binary_Matrix = np.sign(Cliffs_Delta_Binary_Matrix)
        sample_size_1 = len(column_1)
        sample_size_2 = len(column_2)
        Sigma = np.sum((Cliffs_Delta_Binary_Matrix - Cliffs_Delta) ** 2) / ((sample_size_1 * sample_size_2) - 1)
        X_Larger_Var = np.var(np.array([np.sum(xi > column_2) / sample_size_2 - np.sum(xi < column_2) / sample_size_2 for xi in column_1]), ddof = 1)
        X_Smaller_Var = np.var(np.array([np.sum(yi > column_1) / sample_size_1 - np.sum(yi < column_1) / sample_size_1 for yi in column_2]), ddof = 1)
        Cliffs_Variance = ((sample_size_2 - 1) * X_Larger_Var + (sample_size_1 - 1) * X_Smaller_Var + Sigma) / (sample_size_1 * sample_size_2)
        critical_z_value = norm.ppf(0.05 / 2)
        upper_ci_GK = (Cliffs_Delta_no_ties - Cliffs_Delta_no_ties**3 - critical_z_value * np.sqrt(Cliffs_Variance) * np.sqrt((1 - Cliffs_Delta_no_ties**2)**2 + critical_z_value**2 * Cliffs_Variance)) / (1 - Cliffs_Delta_no_ties**2 + critical_z_value**2 * Cliffs_Variance)
        lower_ci_GK = (Cliffs_Delta_no_ties - Cliffs_Delta_no_ties**3 + critical_z_value * np.sqrt(Cliffs_Variance) * np.sqrt((1 - Cliffs_Delta_no_ties**2)**2 + critical_z_value**2 * Cliffs_Variance)) / (1 - Cliffs_Delta_no_ties**2 + critical_z_value**2 * Cliffs_Variance)
        upper_ci_GK_cliff = 1 - (1 - upper_ci_GK) / 2
        lower_ci_GK_cliff = 1 - (1 - lower_ci_GK) / 2
                
        # C. Brunner & Munzel Method (2000)
        ranked_data = rankdata(np.concatenate([column_1, column_2]))
        Mean_Ranks_1 = np.mean(ranked_data[np.arange(1, sample_size_1 + 1) - 1])
        Mean_Ranks_2 = np.mean(ranked_data[np.arange(sample_size_1 + 1, sample_size + 1) - 1])
        S1sq = np.sum((ranked_data[np.arange(1, sample_size_1 + 1) - 1] - rankdata(column_1) - Mean_Ranks_1 + (sample_size_1 + 1) / 2)**2) / (sample_size_1 - 1)
        S2sq = np.sum((ranked_data[np.arange(sample_size_1 + 1, sample_size + 1) - 1] - rankdata(column_2) - Mean_Ranks_2 + (sample_size_2 + 1) / 2)**2) / (sample_size_2 - 1)       
        sig1 = S1sq / sample_size_2**2
        sig2 = S2sq / sample_size_1**2        
        Standard_Error_Brunner_Munzel = np.sqrt(sample_size) * np.sqrt(sample_size * (sig1 / sample_size_1 + sig2 / sample_size_2))       
        Degrees_of_Freedom = (S1sq / sample_size_2 + S2sq / sample_size_1)**2 / ((S1sq / sample_size_2)**2 / (sample_size_1 - 1) + (S2sq / sample_size_1)**2 / (sample_size_2 - 1))
        Critical_Value_Brunner_Munzel = t.ppf(1 - (1-confidence_level) / 2, Degrees_of_Freedom)
        LowerCi_GK_Brunner_Munzel = POS_Grissom_Kim - Critical_Value_Brunner_Munzel * Standard_Error_Brunner_Munzel / sample_size
        UpperCi_GK_Brunner_Munzel = POS_Grissom_Kim + Critical_Value_Brunner_Munzel * Standard_Error_Brunner_Munzel / sample_size

        
        # 2. Probability of Superiority with ties CI (Vargha-Delaney)       
        # A. Traditional-Wald Type CI's
        ci_lower_Vargha_Delaney_Ruscio = POS_Vargha_Delaney - (Standard_Error_Ruscio * Critical_t_value)
        ci_upper_Vargha_Delaney_Ruscio = POS_Vargha_Delaney + (Standard_Error_Ruscio * Critical_t_value)

        # B. Cliffs Method         
        Upper_VDA_Cliff = (Cliffs_Delta - Cliffs_Delta**3 - critical_z_value * np.sqrt(Cliffs_Variance) * np.sqrt((1 - Cliffs_Delta**2)**2 + critical_z_value**2 * Cliffs_Variance)) / (1 - Cliffs_Delta**2 + critical_z_value**2 * Cliffs_Variance)
        Lower_VDA_Cliff= (Cliffs_Delta - Cliffs_Delta**3 + critical_z_value * np.sqrt(Cliffs_Variance) * np.sqrt((1 - Cliffs_Delta**2)**2 + critical_z_value**2 * Cliffs_Variance)) / (1 - Cliffs_Delta**2 + critical_z_value**2 * Cliffs_Variance)
        upper_ci_VDA_cliff = 1 - (1 - Upper_VDA_Cliff) / 2
        lower_ci_VDA_cliff = 1 - (1 - Lower_VDA_Cliff) / 2

        # C. Brunner-Munzel Method                
        LowerCi_VD_Brunner_Munzel = POS_Vargha_Delaney - Critical_Value_Brunner_Munzel * Standard_Error_Brunner_Munzel / sample_size
        UpperCi_VD_Brunner_Munzel = POS_Vargha_Delaney + Critical_Value_Brunner_Munzel * Standard_Error_Brunner_Munzel / sample_size

        # D + E. Hanley & McNaeil Method
        merged_values = np.unique(np.concatenate((column_1, column_2)))
        sorted_freq_vector_x = np.array([np.sum(column_1 == value) for value in merged_values])
        sorted_freq_vector_y = np.array([np.sum(column_2 == value) for value in merged_values])

        count_values_larger_in_column_2 = np.zeros_like(merged_values)
        count_values_smaller_in_column_1 = np.zeros_like(merged_values)

        for i, value in enumerate(merged_values):
            if value in column_1:count_values_larger_in_column_2[i] = np.sum(column_2 > value)

        for i, value in enumerate(merged_values):
            if value in column_2: count_values_smaller_in_column_1[i] = np.sum(column_1 < value)

        term1 = np.sum(sorted_freq_vector_x * count_values_larger_in_column_2 + 0.5 *sorted_freq_vector_x * sorted_freq_vector_y)
        term2 = np.sum(sorted_freq_vector_y * ((count_values_smaller_in_column_1**2 + count_values_smaller_in_column_1  * sorted_freq_vector_x + (1/3) * sorted_freq_vector_x**2)))
        term3 = np.sum(sorted_freq_vector_x * ((count_values_larger_in_column_2**2+ count_values_larger_in_column_2 * sorted_freq_vector_y + (1/3) * sorted_freq_vector_y**2)))
        AUC = term1 / (len(column_1) * (len(column_2)))
        Q1_Formula_1 = term3 / (len(column_1)*(len(column_2)**2))
        Q2_Formula_1 = term2 / (len(column_2)*(len(column_1)**2))
        Q1_Formula_2 = (AUC) / (2-AUC)
        Q2_Formula_2 = (2* AUC**2) / (1+AUC)
        Standard_Error_Hanley_McNeil_1 = np.sqrt(((AUC*(1-AUC)) + ((len(column_1) - 1)* (Q2_Formula_1 - AUC**2)) + ((len(column_2) - 1)* (Q1_Formula_1 - AUC**2))) / (len(column_1)*(len(column_2))))
        Standard_Error_Hanley_McNeil_2 = np.sqrt(((AUC*(1-AUC)) + ((len(column_1) - 1)* (Q2_Formula_2 - AUC**2)) + ((len(column_2) - 1)* (Q1_Formula_2 - AUC**2))) / (len(column_1)*(len(column_2))))
        CI_VDA_Hanely_Mcneil_1_upper= POS_Vargha_Delaney - critical_z_value * Standard_Error_Hanley_McNeil_1
        CI_VDA_Hanely_Mcneil_1_lower= POS_Vargha_Delaney + critical_z_value * Standard_Error_Hanley_McNeil_1

        CI_VDA_Hanely_Mcneil_2_upper = POS_Vargha_Delaney - critical_z_value * Standard_Error_Hanley_McNeil_2 
        CI_VDA_Hanely_Mcneil_2_lower = POS_Vargha_Delaney + critical_z_value * Standard_Error_Hanley_McNeil_2

        # F. Fisher tranformation of the rank biserial Correlation (Note that one can use the confidence intervals of cliff's delta since it is equal to rank biserial correlation in the two samples case)
        Rank_Biserial_Transformed = math.atanh(Cliffs_Delta)
        Standard_Error_Fisher = np.sqrt((sample_size_1+sample_size_2 +1) / (3*sample_size_1*sample_size_2)) #The Wikipedia Version

        Lower_ci_cliff_delta_Fisher = math.tanh(Rank_Biserial_Transformed - critical_z_value * Standard_Error_Fisher)
        Upper_ci_cliff_delta_Fisher = math.tanh(Rank_Biserial_Transformed + critical_z_value * Standard_Error_Fisher)
        Ci_PS_Vargha_Delaney_Fisher_upper = 1 - (1 - Lower_ci_cliff_delta_Fisher) / 2
        Ci_PS_Vargha_Delaney_Fisher_lower = 1 - (1 - Upper_ci_cliff_delta_Fisher) / 2

        # G. Metsamuuronen, Converted Sommers Deltas CI (PHD)
        # Step 1 - build and convert data to a contingency table in a long format and then present the data in long format with all the relevant variables
        column_1_long = np.concatenate([params["column_1"], params["column_2"]])
        column_2_long = np.concatenate([np.ones_like(params["column_1"]), 2 * np.ones_like(params["column_2"])])
        Data_Frame = pd.DataFrame({'column_1_long': column_1_long, 'column_2_long': column_2_long})
        Contingency_Table = pd.crosstab(Data_Frame['column_1_long'], Data_Frame['column_2_long'])
        Final_Data = Contingency_Table.reset_index().melt(id_vars='column_1_long', var_name='column_2_long', value_name='Nij')
        Final_Data['Ni'] = Final_Data['column_1_long'].map(Data_Frame['column_1_long'].value_counts())
        Final_Data['Nj'] = Final_Data['column_2_long'].map(Data_Frame['column_2_long'].value_counts())
        Final_Data['P'] = Final_Data['Concordant Pairs'] = 0
        Final_Data['Q'] = Final_Data['Disconcordant Pairs'] = 0

        # Step 2 - Calculation of Concordant and Disconcordant Pairs
        for i, row in Final_Data.iterrows():
            x_val = row['column_1_long']
            y_val = row['column_2_long']
            Concordant = 0
            Disconcordant = 0
            
            for index, other_row in Final_Data[Final_Data['column_1_long'] != x_val].iterrows():
                count1 = other_row['Nij']  # Frequency of the other values group
                if (x_val > other_row['column_1_long'] and y_val > other_row['column_2_long']) or (x_val < other_row['column_1_long'] and y_val < other_row['column_2_long']):
                    Concordant += count1
            for index, other_row in Final_Data[Final_Data['column_1_long'] != x_val].iterrows():
                count2 = other_row['Nij']  # Frequency of the other values group
                if (x_val > other_row['column_1_long'] and y_val < other_row['column_2_long']) or (x_val < other_row['column_1_long'] and y_val > other_row['column_2_long']):
                    Disconcordant += count2
            
            Final_Data.at[i, 'Concordant Pairs'] = Concordant
            Final_Data.at[i, 'Disconcordant Pairs'] = Disconcordant
            Final_Data.at[i, '(C-D)^2*Nij'] = (Concordant - Disconcordant)**2 * row['Nij']
            Final_Data.at[i, 'Concordant Pairs * Frequency'] = Concordant * row['Nij']
            Final_Data.at[i, 'Disconcordant Pairs * Frequency'] = Disconcordant * row['Nij']
            Final_Data.at[i, 'P'] = Final_Data.at[i, 'Concordant Pairs * Frequency']
            Final_Data.at[i, 'Q'] = Final_Data.at[i, 'Disconcordant Pairs * Frequency']

        # Step 3 = Calculation of Measures of Association and thier Standrd Errors
        P = Final_Data['P'].sum()
        Q = Final_Data['Q'].sum()
        Dg = sample_size**2 - np.sum(np.sum(Contingency_Table, axis = 0)**2)
               
        Sommers_Delta_Dgx = (P - Q) / Dg
        Standard_Error_Delta_Dgx_Metsamuuronen = (2 / Dg**2) * np.sqrt(np.sum(Final_Data['Nij'] * (Dg * (Final_Data['Concordant Pairs'] - Final_Data['Disconcordant Pairs']) - (P - Q) *(sample_size - Final_Data['Nj']))**2))
        Standard_Error_Delta_Dgx_H0 = 2/ Dg *np.sqrt((np.sum(Final_Data['Nij'] * (Final_Data['Disconcordant Pairs'] - Final_Data['Concordant Pairs'])**2)) - (1 / sample_size * (P-Q)**2))
        ci_lower_Delta_Metsamuuronen = ((Sommers_Delta_Dgx) - (Standard_Error_Delta_Dgx_Metsamuuronen * Critical_t_value))
        ci_upper_Delta_Metsamuuronen = ((Sommers_Delta_Dgx) + (Standard_Error_Delta_Dgx_Metsamuuronen * Critical_t_value))      
        ci_lower_VDA_Metsamuuronen = max(0, min(1 - (0.5 * ci_upper_Delta_Metsamuuronen + 0.5), 1 - (0.5 * ci_lower_Delta_Metsamuuronen + 0.5)))
        ci_upper_VDA_Metsamuuronen = min(1, max(1 - (0.5 * ci_upper_Delta_Metsamuuronen + 0.5), 1 - (0.5 * ci_lower_Delta_Metsamuuronen + 0.5)))

        # 3. PS Gamma-Based (Metsamuuronen) CI's
        
        # A. Traditional-Wald Type CI's
        ci_lower_Gamma = (1 - POS_Gamma_Based_Metsamuuronen) - (Standard_Error_Ruscio * Critical_t_value)
        ci_upper_Gamma = (1 - POS_Gamma_Based_Metsamuuronen) + (Standard_Error_Ruscio * Critical_t_value)
        ci_lower_Gamma = max(ci_lower_Gamma, 0)
        ci_upper_Gamma = min(ci_upper_Gamma, 1)

        # B. Metsammuuronen - this is based on the same code as sommer's delta converted to VDA above
        Standard_Error_Gamma_Metsamuuronen = (4 / (P + Q)**2) * np.sqrt(np.sum(Final_Data['Nij'] * (P * Final_Data['Disconcordant Pairs'] - Q * Final_Data['Concordant Pairs'])**2))
        Standard_Error_Gamma_H0 = (2/ (P+Q)) * np.sqrt((np.sum(Final_Data['Nij'] * (Final_Data['Disconcordant Pairs'] - Final_Data['Concordant Pairs'])**2)) - (1 / sample_size * (P-Q)**2))
        GHD_SE_HO = 0.5 * Standard_Error_Gamma_H0 + 0.5
        ci_lower_Gamma_Metsamuuronen = ((Goodman_Krushkal_Gamma_Correlation) - (Standard_Error_Gamma_Metsamuuronen * Critical_t_value))
        ci_upper_Gamma_Metsamuuronen = ((Goodman_Krushkal_Gamma_Correlation) + (Standard_Error_Gamma_Metsamuuronen * Critical_t_value))      
        ci_lower_Gamma_Metsamuuronen = 1 - (0.5 * ci_lower_Gamma_Metsamuuronen + 0.5)
        ci_upper_Gamma_Metsamuuronen = 1 - (0.5 * ci_upper_Gamma_Metsamuuronen + 0.5)

        # 4. Cliffs Delta   - A simple linear conversion of the VDA CI's - directly in the output  
    

        # 5. Genralized Odds Ratio (Agresti and Churilov/Obrien for considering ties) # I borrowed this idea form genodds package. maybe invistigate more options...
        Contingency_Table = pd.crosstab(Data_Frame['column_1_long'], Data_Frame['column_2_long'])
        Contingency_Table['p(x)'] = Contingency_Table[1].astype(float) / sample_size
        Contingency_Table['p(y)'] = Contingency_Table[2].astype(float) / sample_size
        Contingency_Table['p(x2)'] = ((Contingency_Table['p(x)'] + Contingency_Table['p(y)']))* np.sum(Contingency_Table['p(x)']) 
        Contingency_Table['p(y2)'] = ((Contingency_Table['p(x)'] + Contingency_Table['p(y)']))* np.sum(Contingency_Table['p(y)'])

        Pi = np.concatenate([Contingency_Table['p(x)'].values, Contingency_Table['p(y)'].values]) # type: ignore
        RT = np.concatenate([Contingency_Table['p(y)'].values, Contingency_Table['p(x)'].values]) # type: ignore
        Pi2 = np.concatenate([Contingency_Table['p(x2)'].values, Contingency_Table['p(y2)'].values]) # type: ignore
        RT2 = np.concatenate([Contingency_Table['p(y2)'].values, Contingency_Table['p(x2)'].values]) # type: ignore

        Contingency_Table['Rd(x)'] = [float(Contingency_Table.loc[:current_index -1, 'p(y)'].sum()) for current_index in Contingency_Table.index]
        Contingency_Table['Rd(y)'] = [float(Contingency_Table.loc[current_index + 1:, 'p(x)'].sum()) for current_index in Contingency_Table.index]        
        Contingency_Table['Rd(x2)'] = [float(Contingency_Table.loc[:current_index -1, 'p(y2)'].sum()) for current_index in Contingency_Table.index]
        Contingency_Table['Rd(y2)'] = [float(Contingency_Table.loc[current_index + 1:, 'p(x2)'].sum()) for current_index in Contingency_Table.index]
        Contingency_Table['Rs(x)'] = [float(Contingency_Table.loc[current_index + 1:, 'p(y)'].sum()) for current_index in Contingency_Table.index]
        Contingency_Table['Rs(y)'] = [float(Contingency_Table.loc[:current_index -1, 'p(x)'].sum()) for current_index in Contingency_Table.index]
        Contingency_Table['Rs(x2)'] = [float(Contingency_Table.loc[current_index + 1:, 'p(y2)'].sum()) for current_index in Contingency_Table.index]
        Contingency_Table['Rs(y2)'] = [float(Contingency_Table.loc[:current_index -1, 'p(x2)'].sum()) for current_index in Contingency_Table.index]

        RS = np.concatenate([Contingency_Table['Rs(x)'].values, Contingency_Table['Rs(y)'].values]) # type: ignore
        RD = np.concatenate([Contingency_Table['Rd(x)'].values, Contingency_Table['Rd(y)'].values]) # type: ignore
        RS_ties_corrected = np.concatenate([Contingency_Table['Rs(x)'].values, Contingency_Table['Rs(y)'].values]) + 0.5 * RT # type: ignore
        RD_ties_corrected = np.concatenate([Contingency_Table['Rd(x)'].values, Contingency_Table['Rd(y)'].values]) + 0.5 * RT # type: ignore

        RS2 = np.concatenate([Contingency_Table['Rs(x2)'].values, Contingency_Table['Rs(y2)'].values]) # type: ignore
        RD2 = np.concatenate([Contingency_Table['Rd(x2)'].values, Contingency_Table['Rd(y2)'].values]) # type: ignore
        RS2_ties_corrected = np.concatenate([Contingency_Table['Rs(x2)'].values, Contingency_Table['Rs(y2)'].values]) + 0.5 * RT2 # type: ignore
        RD2_ties_corrected = np.concatenate([Contingency_Table['Rd(x2)'].values, Contingency_Table['Rd(y2)'].values]) + 0.5 * RT2 # type: ignore
        
        Standard_Error_Odds_Ratio_Agresti = ((2 / sum(Pi * RD))* (sum(Pi*((RD * (1/Generalizd_Odds_Ratio_Agresti)) - RS)**2)/sample_size)**0.5) / (1/Generalizd_Odds_Ratio_Agresti)
        Standard_Error_Null_Agresti = ((2 / sum(Pi2 * RD2))* (sum(Pi2*((RD2) - RS2)**2)/sample_size)**0.5)
        Standard_Error_Odds_Ratio_Churilov = ((2 / sum(Pi * RD_ties_corrected))* (sum(Pi*((RD_ties_corrected * (1/Generalizd_Odds_Ratio_Churilov)) - RS_ties_corrected)**2)/sample_size)**0.5) / (1/Generalizd_Odds_Ratio_Churilov)       
        Standard_Error_Null_Churilov = ((2 / sum(Pi2 * RD2_ties_corrected))* (sum(Pi2*((RD2_ties_corrected) - RS2_ties_corrected)**2)/sample_size)**0.5) 
    
        CI_genodds_Agresti_lower = np.exp(norm.ppf((1-confidence_level)/2, loc=np.log(Generalizd_Odds_Ratio_Agresti), scale=Standard_Error_Odds_Ratio_Agresti))
        CI_genodds_Agresti_upper = np.exp(norm.ppf(1 - (1-confidence_level)/2, loc=np.log(Generalizd_Odds_Ratio_Agresti), scale=Standard_Error_Odds_Ratio_Agresti))
        CI_genodds_Null_Agresti_lower = np.exp(norm.ppf((1-confidence_level)/2, loc=np.log(Generalizd_Odds_Ratio_Agresti), scale=Standard_Error_Null_Agresti))
        CI_genodds_Null_Agresti_Upper = np.exp(norm.ppf(1 - (1-confidence_level)/2, loc=np.log(Generalizd_Odds_Ratio_Agresti), scale=Standard_Error_Null_Agresti))
        CI_genodds_Churilov_lower = np.exp(norm.ppf((1-confidence_level)/2, loc=np.log(Generalizd_Odds_Ratio_Churilov), scale=Standard_Error_Odds_Ratio_Churilov))
        CI_genodds_Churilov_upper = np.exp(norm.ppf(1 - (1-confidence_level)/2, loc=np.log(Generalizd_Odds_Ratio_Churilov), scale=Standard_Error_Odds_Ratio_Churilov))
        CI_genodds_Null_Churilov_lower = np.exp(norm.ppf((1-confidence_level)/2, loc=np.log(Generalizd_Odds_Ratio_Churilov), scale=Standard_Error_Null_Churilov))
        CI_genodds_Null_Churilov_Upper = np.exp(norm.ppf(1 - (1-confidence_level)/2, loc=np.log(Generalizd_Odds_Ratio_Churilov), scale=Standard_Error_Null_Churilov))


        # Set results
        results = {}
        
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"

        # Parametric Effect Sizes Based on the Distribution of Cohen's ds
        results["Cohen's U1_ds"] = np.around(cohens_U1_ds*100, 4) 
        results["Lower Central CI U1_ds"] = ci_lower_cohens_U1_ds_Central = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2)))) > (((2 * (norm.cdf((ci_upper_cohens_ds_Central) / 2)) - 1) / (norm.cdf((ci_upper_cohens_ds_Central) / 2)))) else (((2 * (norm.cdf((ci_lower_cohens_ds_Central) / 2)) - 1) / (norm.cdf((ci_lower_cohens_ds_Central) / 2))))),0),3) # type: ignore
        results["Upper Central CI U1_ds"] = ci_upper_cohens_U1_ds_Central = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_cohens_ds_Central) / 2)) - 1) / (norm.cdf(abs(ci_lower_cohens_ds_Central) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)))))),100),3) # type: ignore
        results["Lower Pivotal CI U1_ds"] = ci_lower_cohens_U1_ds_Pivotal = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)))) > (((2 * (norm.cdf((ci_upper_Pivotal_ds) / 2)) - 1) / (norm.cdf((ci_upper_Pivotal_ds) / 2)))) else (((2 * (norm.cdf((ci_lower_Pivotal_ds) / 2)) - 1) / (norm.cdf((ci_lower_Pivotal_ds) / 2))))),0),3) # type: ignore
        results["Upper Pivotal CI U1_ds"] = ci_upper_cohens_U1_ds_Pivotal = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_Pivotal_ds) / 2)) - 1) / (norm.cdf(abs(ci_lower_Pivotal_ds) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_ds) / 2)))))),100),3) # type: ignore
        results["Statistical Line Cohen's U1_ds"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2081d\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U1_ds * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U1_ds_Pivotal, 3),  np.around(ci_upper_cohens_U1_ds_Pivotal, 3))

        results["Cohen's U2_ds"] = np.around(cohens_U2_ds*100, 4)
        results["Lower Central CI U2_ds"] = ci_lower_cohens_U2_ds_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_cohens_ds_Central) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2))  ) )) > ((( (norm.cdf((ci_upper_cohens_ds_Central) / 2))  ) )) else ((( (norm.cdf((ci_lower_cohens_ds_Central) / 2))  ) ))),50),3) # type: ignore
        results["Upper Central CI U2_ds"] = ci_upper_cohens_U2_ds_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_cohens_ds_Central) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_cohens_ds_Central) / 2))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U2_ds"] = ci_lower_cohens_U2_ds_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_ds) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_ds) / 2))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_ds) / 2))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_ds) / 2))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U2_ds"] = ci_upper_cohens_U2_ds_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_ds) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_ds) / 2))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U2_ds"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2082d\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U2_ds * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U2_ds_Pivotal, 3),  np.around(ci_upper_cohens_U2_ds_Pivotal, 3))

        results["Cohen's U3_ds"] = np.around(cohens_U3_ds*100, 4)
        results["Lower Central CI U3_ds"] = ci_lower_cohens_U3_ds_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_cohens_ds_Central)))  ) )) if ((( (norm.cdf(abs(ci_upper_cohens_ds_Central)  ))  ) )) > ((( (norm.cdf((ci_upper_cohens_ds_Central)  ))  ) )) else ((( (norm.cdf((ci_lower_cohens_ds_Central)  ))  ) ))),50),3) # type: ignore
        results["Upper Central CI U3_ds"] = ci_upper_cohens_U3_ds_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_cohens_ds_Central) ))  ) )), ((( (norm.cdf(abs(ci_upper_cohens_ds_Central)  ))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U3_ds"] = ci_lower_cohens_U3_ds_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_ds)))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_ds)  ))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_ds)  ))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_ds)  ))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U3_ds"] = ci_upper_cohens_U3_ds_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_ds)))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_ds)  ))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U3_ds"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2083d\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U3_ds * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U3_ds_Pivotal, 3),  np.around(ci_upper_cohens_U3_ds_Pivotal, 3))

        results["Mcgraw and Wong, CLds"] = np.around(100 * (Mcgraw_Wong_CLds),4)
        results["Lower Central Ci Mcgraw and Wong, CLds"] = ci_lower_clds_Central = np.around(100 * (max(norm.cdf(ci_lower_cohens_ds_Central / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Central Ci Mcgraw and Wong, CLds"] = ci_upper_clds_Central = np.around(100 * (min(norm.cdf(ci_upper_cohens_ds_Central / np.sqrt(2)),1)),4) # type: ignore
        results["Lower Pivotal Ci Mcgraw and Wong, CLds"] = ci_lower_clds_Pivotal = np.around(100 * (max(norm.cdf(ci_lower_Pivotal_ds / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Pivotal Ci Mcgraw and Wong, CLds"] = ci_upper_clds_Pivotal = np.around(100 * (min(norm.cdf(ci_upper_Pivotal_ds / np.sqrt(2)),1)),4) # type: ignore
        results["Statistical Line Mcgraw and Wong, CLds"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLd\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(Mcgraw_Wong_CLds * 100),1), confidence_level_percentages , np.around(ci_lower_clds_Pivotal, 3),  np.around(ci_upper_clds_Pivotal, 3))

        results["Proportion Of Overlap ds"] = np.around(proportion_of_overlap_ds, 4)
        results["Lower Central CI Proportion of Overlap_ds"] = ci_lower_pov_Central_ds = np.around((2 * norm.cdf(-abs(ci_lower_cohens_ds_Central) / 2)), 4) # type: ignore
        results["Upper Central CI Proportion of Overlap_ds"] = ci_upper_pov_Central_ds = np.around((2 * norm.cdf(-abs(ci_upper_cohens_ds_Central) / 2)), 4) # type: ignore
        results["Lower Pivotal CI Proportion of Overlap_ds"] = ci_lower_pov_Pivotal_ds = np.around((2 * norm.cdf(-abs(ci_lower_Pivotal_ds) / 2)), 4) # type: ignore
        results["Upper Pivotal CI Proportion of Overlap_ds"] = ci_upper_pov_Pivotal_ds = np.around((2 * norm.cdf(-abs(ci_upper_Pivotal_ds) / 2)), 4) # type: ignore
        results["Statistical Line Proportion of Overlap (POVds) "] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, POVd\u209B = {:.3f}%, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(proportion_of_overlap_ds),3), confidence_level_percentages , np.around(ci_lower_pov_Pivotal_ds, 3),  np.around(ci_upper_pov_Pivotal_ds, 3))


        # Parametric Effect Sizes Based on the Distribution of Hedge's gs
        results["Cohen's U1_gs"] = np.around(cohens_U1_gs*100, 4) 
        results["Lower Central CI U1_gs"] = ci_lower_cohens_U1_gs_Central = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2)))) > (((2 * (norm.cdf((ci_upper_hedges_gs_Central) / 2)) - 1) / (norm.cdf((ci_upper_hedges_gs_Central) / 2)))) else (((2 * (norm.cdf((ci_lower_hedges_gs_Central) / 2)) - 1) / (norm.cdf((ci_lower_hedges_gs_Central) / 2))))),0),3) # type: ignore
        results["Upper Central CI U1_gs"] = ci_upper_cohens_U1_gs_Central = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_hedges_gs_Central) / 2)) - 1) / (norm.cdf(abs(ci_lower_hedges_gs_Central) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)))))),100),3) # type: ignore
        results["Lower Pivotal CI U1_gs"] = ci_lower_cohens_U1_gs_Pivotal = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)))) > (((2 * (norm.cdf((ci_upper_Pivotal_gs) / 2)) - 1) / (norm.cdf((ci_upper_Pivotal_gs) / 2)))) else (((2 * (norm.cdf((ci_lower_Pivotal_gs) / 2)) - 1) / (norm.cdf((ci_lower_Pivotal_gs) / 2))))),0),3) # type: ignore
        results["Upper Pivotal CI U1_gs"] = ci_upper_cohens_U1_gs_Pivotal = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_Pivotal_gs) / 2)) - 1) / (norm.cdf(abs(ci_lower_Pivotal_gs) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gs) / 2)))))),100),3) # type: ignore
        results["Statistical Line Cohen's U1_gs"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2081g\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U1_gs * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U1_gs_Pivotal, 3),  np.around(ci_upper_cohens_U1_gs_Pivotal, 3))

        results["Cohen's U2_gs"] = np.around(cohens_U2_gs*100, 4)
        results["Lower Central CI U2_gs"] = ci_lower_cohens_U2_gs_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_hedges_gs_Central) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2))  ) )) > ((( (norm.cdf((ci_upper_hedges_gs_Central) / 2))  ) )) else ((( (norm.cdf((ci_lower_hedges_gs_Central) / 2))  ) ))),50),3) # type: ignore
        results["Upper Central CI U2_gs"] = ci_upper_cohens_U2_gs_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_hedges_gs_Central) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_hedges_gs_Central) / 2))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U2_gs"] = ci_lower_cohens_U2_gs_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_gs) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_gs) / 2))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_gs) / 2))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_gs) / 2))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U2_gs"] = ci_upper_cohens_U2_gs_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_gs) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_gs) / 2))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U2_gs"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2082g\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U2_gs * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U2_gs_Pivotal, 3),  np.around(ci_upper_cohens_U2_gs_Pivotal, 3))

        results["Cohen's U3_gs"] = np.around(cohens_U3_gs*100, 4)
        results["Lower Central CI U3_gs"] = ci_lower_cohens_U3_gs_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_hedges_gs_Central)))  ) )) if ((( (norm.cdf(abs(ci_upper_hedges_gs_Central)  ))  ) )) > ((( (norm.cdf((ci_upper_hedges_gs_Central)  ))  ) )) else ((( (norm.cdf((ci_lower_hedges_gs_Central)  ))  ) ))),50),3) # type: ignore
        results["Upper Central CI U3_gs"] = ci_upper_cohens_U3_gs_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_hedges_gs_Central) ))  ) )), ((( (norm.cdf(abs(ci_upper_hedges_gs_Central)  ))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U3_gs"] = ci_lower_cohens_U3_gs_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_gs)))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_gs)  ))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_gs)  ))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_gs)  ))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U3_gs"] = ci_upper_cohens_U3_gs_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_gs)))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_gs)  ))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U3_gs"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2083g\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U3_gs * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U3_gs_Pivotal, 3),  np.around(ci_upper_cohens_U3_gs_Pivotal, 3))

        results["Mcgraw and Wong, CLgs"] = np.around(100 * (Mcgraw_Wong_CLgs),4)
        results["Lower Central Ci Mcgraw and Wong, CLgs"] = ci_lower_clgs_Central = np.around(100 * (max(norm.cdf(ci_lower_hedges_gs_Central / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Central Ci Mcgraw and Wong, CLgs"] = ci_upper_clgs_Central = np.around(100 * (min(norm.cdf(ci_upper_hedges_gs_Central / np.sqrt(2)),1)),4) # type: ignore
        results["Lower Pivotal Ci Mcgraw and Wong, CLgs"] = ci_lower_clgs_Pivotal = np.around(100 * (max(norm.cdf(ci_lower_Pivotal_gs / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Pivotal Ci Mcgraw and Wong, CLgs"] = ci_upper_clgs_Pivotal = np.around(100 * (min(norm.cdf(ci_upper_Pivotal_gs / np.sqrt(2)),1)),4) # type: ignore
        results["Statistical Line Mcgraw and Wong, CLgs"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLg\u209B = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(Mcgraw_Wong_CLgs * 100),1), confidence_level_percentages , np.around(ci_lower_clgs_Pivotal, 3),  np.around(ci_upper_clgs_Pivotal, 3))

        results["Proportion Of Overlap gs"] = np.around(proportion_of_overlap_gs, 4)
        results["Lower Central CI Proportion of Overlap_gs"] = ci_lower_pov_Central_gs = np.around((2 * norm.cdf(-abs(ci_lower_hedges_gs_Central) / 2)), 4) # type: ignore
        results["Upper Central CI Proportion of Overlap_gs"] = ci_upper_pov_Central_gs = np.around((2 * norm.cdf(-abs(ci_upper_hedges_gs_Central) / 2)), 4) # type: ignore
        results["Lower Pivotal CI Proportion of Overlap_gs"] = ci_lower_pov_Pivotal_gs = np.around((2 * norm.cdf(-abs(ci_lower_Pivotal_gs) / 2)), 4) # type: ignore
        results["Upper Pivotal CI Proportion of Overlap_gs"] = ci_upper_pov_Pivotal_gs = np.around((2 * norm.cdf(-abs(ci_upper_Pivotal_gs) / 2)), 4) # type: ignore
        results["Statistical Line Proportion of Overlap (POVgs) "] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, POVg\u209B = {:.3f}%, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(proportion_of_overlap_gs),3), confidence_level_percentages , np.around(ci_lower_pov_Pivotal_gs, 3),  np.around(ci_upper_pov_Pivotal_gs, 3))


        # Parametric Effect Sizes Based on the Distribution of Cohen's dav
        results["Cohen's U1_dav"] = np.around(cohens_U1_dav*100, 4) 
        results["Lower Central CI U1_dav"] = ci_lower_cohens_U1_dav_Central = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_cohens_dav_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_cohens_dav_Central) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_cohens_dav_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_cohens_dav_Central) / 2)))) > (((2 * (norm.cdf((ci_upper_cohens_dav_Central) / 2)) - 1) / (norm.cdf((ci_upper_cohens_dav_Central) / 2)))) else (((2 * (norm.cdf((ci_lower_cohens_dav_Central) / 2)) - 1) / (norm.cdf((ci_lower_cohens_dav_Central) / 2))))),0),3) # type: ignore
        results["Upper Central CI U1_dav"] = ci_upper_cohens_U1_dav_Central = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_cohens_dav_Central) / 2)) - 1) / (norm.cdf(abs(ci_lower_cohens_dav_Central) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_dav) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dav) / 2)))))),100),3) # type: ignore
        results["Lower Pivotal CI U1_dav"] = ci_lower_cohens_U1_dav_Pivotal = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_Pivotal_dav) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dav) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_Pivotal_dav) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dav) / 2)))) > (((2 * (norm.cdf((ci_upper_Pivotal_dav) / 2)) - 1) / (norm.cdf((ci_upper_Pivotal_dav) / 2)))) else (((2 * (norm.cdf((ci_lower_Pivotal_dav) / 2)) - 1) / (norm.cdf((ci_lower_Pivotal_dav) / 2))))),0),3) # type: ignore
        results["Upper Pivotal CI U1_dav"] = ci_upper_cohens_U1_dav_Pivotal = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_Pivotal_dav) / 2)) - 1) / (norm.cdf(abs(ci_lower_Pivotal_dav) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_dav) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dav) / 2)))))),100),3) # type: ignore
        results["Statistical Line Cohen's U1_dav"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2081d\u2090\u1d65 = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U1_dav * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U1_dav_Pivotal, 3),  np.around(ci_upper_cohens_U1_dav_Pivotal, 3))

        results["Cohen's U2_dav"] = np.around(cohens_U2_dav*100, 4)
        results["Lower Central CI U2_dav"] = ci_lower_cohens_U2_dav_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_cohens_dav_Central) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_cohens_dav_Central) / 2))  ) )) > ((( (norm.cdf((ci_upper_cohens_dav_Central) / 2))  ) )) else ((( (norm.cdf((ci_lower_cohens_dav_Central) / 2))  ) ))),50),3) # type: ignore
        results["Upper Central CI U2_dav"] = ci_upper_cohens_U2_dav_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_cohens_dav_Central) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_cohens_dav_Central) / 2))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U2_dav"] = ci_lower_cohens_U2_dav_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_dav) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_dav) / 2))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_dav) / 2))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_dav) / 2))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U2_dav"] = ci_upper_cohens_U2_dav_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_dav) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_dav) / 2))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U2_dav"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2082d\u2090\u1d65 = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U2_dav * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U2_dav_Pivotal, 3),  np.around(ci_upper_cohens_U2_dav_Pivotal, 3))

        results["Cohen's U3_dav"] = np.around(cohens_U3_dav*100, 4)
        results["Lower Central CI U3_dav"] = ci_lower_cohens_U3_dav_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_cohens_dav_Central)))  ) )) if ((( (norm.cdf(abs(ci_upper_cohens_dav_Central)  ))  ) )) > ((( (norm.cdf((ci_upper_cohens_dav_Central)  ))  ) )) else ((( (norm.cdf((ci_lower_cohens_dav_Central)  ))  ) ))),50),3) # type: ignore
        results["Upper Central CI U3_dav"] = ci_upper_cohens_U3_dav_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_cohens_dav_Central) ))  ) )), ((( (norm.cdf(abs(ci_upper_cohens_dav_Central)  ))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U3_dav"] = ci_lower_cohens_U3_dav_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_dav)))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_dav)  ))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_dav)  ))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_dav)  ))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U3_dav"] = ci_upper_cohens_U3_dav_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_dav)))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_dav)  ))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U3_dav"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2083d\u2090\u1d65 = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U3_dav * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U3_dav_Pivotal, 3),  np.around(ci_upper_cohens_U3_dav_Pivotal, 3))

        results["Mcgraw and Wong, CLdav"] = np.around(100 * (Mcgraw_Wong_CLdav),4)
        results["Lower Central Ci Mcgraw and Wong, CLdav"] = ci_lower_cldav_Central = np.around(100 * (max(norm.cdf(ci_lower_cohens_dav_Central / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Central Ci Mcgraw and Wong, CLdav"] = ci_upper_cldav_Central = np.around(100 * (min(norm.cdf(ci_upper_cohens_dav_Central / np.sqrt(2)),1)),4) # type: ignore
        results["Lower Pivotal Ci Mcgraw and Wong, CLdav"] = ci_lower_cldav_Pivotal = np.around(100 * (max(norm.cdf(ci_lower_Pivotal_dav / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Pivotal Ci Mcgraw and Wong, CLdav"] = ci_upper_cldav_Pivotal = np.around(100 * (min(norm.cdf(ci_upper_Pivotal_dav / np.sqrt(2)),1)),4) # type: ignore
        results["Statistical Line Mcgraw and Wong, CLdav"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLd\u2090\u1d65 = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(Mcgraw_Wong_CLdav * 100),1), confidence_level_percentages , np.around(ci_lower_cldav_Pivotal, 3),  np.around(ci_upper_cldav_Pivotal, 3))

        results["Proportion Of Overlap dav"] = np.around(proportion_of_overlap_dav, 4)
        results["Lower Central CI Proportion of Overlap_dav"] = ci_lower_pov_Central_dav = np.around((2 * norm.cdf(-abs(ci_lower_cohens_dav_Central) / 2)), 4) # type: ignore
        results["Upper Central CI Proportion of Overlap_dav"] = ci_upper_pov_Central_dav = np.around((2 * norm.cdf(-abs(ci_upper_cohens_dav_Central) / 2)), 4) # type: ignore
        results["Lower Pivotal CI Proportion of Overlap_dav"] = ci_lower_pov_Pivotal_dav = np.around((2 * norm.cdf(-abs(ci_lower_Pivotal_dav) / 2)), 4) # type: ignore
        results["Upper Pivotal CI Proportion of Overlap_dav"] = ci_upper_pov_Pivotal_dav = np.around((2 * norm.cdf(-abs(ci_upper_Pivotal_dav) / 2)), 4) # type: ignore
        results["Statistical Line Proportion of Overlap (POVdav) "] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, POVd\u2090\u1d65 = {:.3f}%, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(proportion_of_overlap_dav),3), confidence_level_percentages , np.around(ci_lower_pov_Pivotal_dav, 3),  np.around(ci_upper_pov_Pivotal_dav, 3))

        # Parametric Effect Sizes Based on the Distribution of Cohen's gav
        results["Cohen's U1_gav"] = np.around(cohens_U1_gav*100, 4) 
        results["Lower Central CI U1_gav"] = ci_lower_cohens_U1_gav_Central = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_hedges_gav_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_hedges_gav_Central) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_hedges_gav_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_hedges_gav_Central) / 2)))) > (((2 * (norm.cdf((ci_upper_hedges_gav_Central) / 2)) - 1) / (norm.cdf((ci_upper_hedges_gav_Central) / 2)))) else (((2 * (norm.cdf((ci_lower_hedges_gav_Central) / 2)) - 1) / (norm.cdf((ci_lower_hedges_gav_Central) / 2))))),0),3) # type: ignore
        results["Upper Central CI U1_gav"] = ci_upper_cohens_U1_gav_Central = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_hedges_gav_Central) / 2)) - 1) / (norm.cdf(abs(ci_lower_hedges_gav_Central) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_gav) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gav) / 2)))))),100),3) # type: ignore
        results["Lower Pivotal CI U1_gav"] = ci_lower_cohens_U1_gav_Pivotal = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_Pivotal_gav) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gav) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_Pivotal_gav) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gav) / 2)))) > (((2 * (norm.cdf((ci_upper_Pivotal_gav) / 2)) - 1) / (norm.cdf((ci_upper_Pivotal_gav) / 2)))) else (((2 * (norm.cdf((ci_lower_Pivotal_gav) / 2)) - 1) / (norm.cdf((ci_lower_Pivotal_gav) / 2))))),0),3) # type: ignore
        results["Upper Pivotal CI U1_gav"] = ci_upper_cohens_U1_gav_Pivotal = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_Pivotal_gav) / 2)) - 1) / (norm.cdf(abs(ci_lower_Pivotal_gav) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_gav) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_gav) / 2)))))),100),3) # type: ignore
        results["Statistical Line Cohen's U1_gav"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2081g\u2090\u1d65 = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U1_gav * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U1_gav_Pivotal, 3),  np.around(ci_upper_cohens_U1_gav_Pivotal, 3))

        results["Cohen's U2_gav"] = np.around(cohens_U2_gav*100, 4)
        results["Lower Central CI U2_gav"] = ci_lower_cohens_U2_gav_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_hedges_gav_Central) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_hedges_gav_Central) / 2))  ) )) > ((( (norm.cdf((ci_upper_hedges_gav_Central) / 2))  ) )) else ((( (norm.cdf((ci_lower_hedges_gav_Central) / 2))  ) ))),50),3) # type: ignore
        results["Upper Central CI U2_gav"] = ci_upper_cohens_U2_gav_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_hedges_gav_Central) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_hedges_gav_Central) / 2))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U2_gav"] = ci_lower_cohens_U2_gav_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_gav) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_gav) / 2))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_gav) / 2))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_gav) / 2))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U2_gav"] = ci_upper_cohens_U2_gav_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_gav) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_gav) / 2))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U2_gav"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2082g\u2090\u1d65 = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U2_gav * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U2_gav_Pivotal, 3),  np.around(ci_upper_cohens_U2_gav_Pivotal, 3))

        results["Cohen's U3_gav"] = np.around(cohens_U3_gav*100, 4)
        results["Lower Central CI U3_gav"] = ci_lower_cohens_U3_gav_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_hedges_gav_Central)))  ) )) if ((( (norm.cdf(abs(ci_upper_hedges_gav_Central)  ))  ) )) > ((( (norm.cdf((ci_upper_hedges_gav_Central)  ))  ) )) else ((( (norm.cdf((ci_lower_hedges_gav_Central)  ))  ) ))),50),3) # type: ignore
        results["Upper Central CI U3_gav"] = ci_upper_cohens_U3_gav_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_hedges_gav_Central) ))  ) )), ((( (norm.cdf(abs(ci_upper_hedges_gav_Central)  ))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U3_gav"] = ci_lower_cohens_U3_gav_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_gav)))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_gav)  ))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_gav)  ))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_gav)  ))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U3_gav"] = ci_upper_cohens_U3_gav_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_gav)))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_gav)  ))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U3_gav"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2083g\u2090\u1d65 = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U3_gav * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U3_gav_Pivotal, 3),  np.around(ci_upper_cohens_U3_gav_Pivotal, 3))

        results["Mcgraw and Wong, CLgav"] = np.around(100 * (Mcgraw_Wong_CLgav),4)
        results["Lower Central Ci Mcgraw and Wong, CLgav"] = ci_lower_clgav_Central = np.around(100 * (max(norm.cdf(ci_lower_hedges_gav_Central / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Central Ci Mcgraw and Wong, CLgav"] = ci_upper_clgav_Central = np.around(100 * (min(norm.cdf(ci_upper_hedges_gav_Central / np.sqrt(2)),1)),4) # type: ignore
        results["Lower Pivotal Ci Mcgraw and Wong, CLgav"] = ci_lower_clgav_Pivotal = np.around(100 * (max(norm.cdf(ci_lower_Pivotal_gav / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Pivotal Ci Mcgraw and Wong, CLgav"] = ci_upper_clgav_Pivotal = np.around(100 * (min(norm.cdf(ci_upper_Pivotal_gav / np.sqrt(2)),1)),4) # type: ignore
        results["Statistical Line Mcgraw and Wong, CLgav"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLg\u2090\u1d65 = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(Mcgraw_Wong_CLgav * 100),1), confidence_level_percentages , np.around(ci_lower_clgav_Pivotal, 3),  np.around(ci_upper_clgav_Pivotal, 3))

        results["Proportion Of Overlap gav"] = np.around(proportion_of_overlap_gav, 4)
        results["Lower Central CI Proportion of Overlap_gav"] = ci_lower_pov_Central_gav = np.around((2 * norm.cdf(-abs(ci_lower_hedges_gav_Central) / 2)), 4) # type: ignore
        results["Upper Central CI Proportion of Overlap_gav"] = ci_upper_pov_Central_gav = np.around((2 * norm.cdf(-abs(ci_upper_hedges_gav_Central) / 2)), 4) # type: ignore
        results["Lower Pivotal CI Proportion of Overlap_gav"] = ci_lower_pov_Pivotal_gav = np.around((2 * norm.cdf(-abs(ci_lower_Pivotal_gav) / 2)), 4) # type: ignore
        results["Upper Pivotal CI Proportion of Overlap_gav"] = ci_upper_pov_Pivotal_gav = np.around((2 * norm.cdf(-abs(ci_upper_Pivotal_gav) / 2)), 4) # type: ignore
        results["Statistical Line Proportion of Overlap (POVgav) "] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, POVg\u2090\u1d65 = {:.3f}%, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(proportion_of_overlap_gav),3), confidence_level_percentages , np.around(ci_lower_pov_Pivotal_gav, 3),  np.around(ci_upper_pov_Pivotal_gav, 3))

        # Parametric Effect Sizes Based on the Distribution of Cohen's dpop
        results["Cohen's U1_dpop"] = np.around(cohens_U1_dpop*100, 4) 
        results["Lower Central CI U1_dpop"] = ci_lower_cohens_U1_dpop_Central = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2)) - 1) / (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2)))) > (((2 * (norm.cdf((ci_upper_cohens_dpop_Central) / 2)) - 1) / (norm.cdf((ci_upper_cohens_dpop_Central) / 2)))) else (((2 * (norm.cdf((ci_lower_cohens_dpop_Central) / 2)) - 1) / (norm.cdf((ci_lower_cohens_dpop_Central) / 2))))),0),3) # type: ignore
        results["Upper Central CI U1_dpop"] = ci_upper_cohens_U1_dpop_Central = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_cohens_dpop_Central) / 2)) - 1) / (norm.cdf(abs(ci_lower_cohens_dpop_Central) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)))))),100),3) # type: ignore
        results["Lower Pivotal CI U1_dpop"] = ci_lower_cohens_U1_dpop_Pivotal = np.round(max((100 * (((2 * (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)))) if (((2 * (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)))) > (((2 * (norm.cdf((ci_upper_Pivotal_dpop) / 2)) - 1) / (norm.cdf((ci_upper_Pivotal_dpop) / 2)))) else (((2 * (norm.cdf((ci_lower_Pivotal_dpop) / 2)) - 1) / (norm.cdf((ci_lower_Pivotal_dpop) / 2))))),0),3) # type: ignore
        results["Upper Pivotal CI U1_dpop"] = ci_upper_cohens_U1_dpop_Pivotal = np.round(min((100 * max((((2 * (norm.cdf(abs(ci_lower_Pivotal_dpop) / 2)) - 1) / (norm.cdf(abs(ci_lower_Pivotal_dpop) / 2)))), (((2 * (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)) - 1) / (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2)))))),100),3) # type: ignore
        results["Statistical Line Cohen's U1_dpop"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2081d\u209a\u2092\u209a = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U1_dpop * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U1_dpop_Pivotal, 3),  np.around(ci_upper_cohens_U1_dpop_Pivotal, 3))

        results["Cohen's U2_dpop"] = np.around(cohens_U2_dpop*100, 4)
        results["Lower Central CI U2_dpop"] = ci_lower_cohens_U2_dpop_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2))  ) )) > ((( (norm.cdf((ci_upper_cohens_dpop_Central) / 2))  ) )) else ((( (norm.cdf((ci_lower_cohens_dpop_Central) / 2))  ) ))),50),3) # type: ignore
        results["Upper Central CI U2_dpop"] = ci_upper_cohens_U2_dpop_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_cohens_dpop_Central) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_cohens_dpop_Central) / 2))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U2_dpop"] = ci_lower_cohens_U2_dpop_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_dpop) / 2))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_dpop) / 2))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_dpop) / 2))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U2_dpop"] = ci_upper_cohens_U2_dpop_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_dpop) / 2))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_dpop) / 2))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U2_dpop"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2082d\u209a\u2092\u209a = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U2_dpop * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U2_dpop_Pivotal, 3),  np.around(ci_upper_cohens_U2_dpop_Pivotal, 3))

        results["Cohen's U3_dpop"] = np.around(cohens_U3_dpop*100, 4)
        results["Lower Central CI U3_dpop"] = ci_lower_cohens_U3_dpop_Central = np.round(max((100 * ((((norm.cdf(abs(ci_upper_cohens_dpop_Central)))  ) )) if ((( (norm.cdf(abs(ci_upper_cohens_dpop_Central)  ))  ) )) > ((( (norm.cdf((ci_upper_cohens_dpop_Central)  ))  ) )) else ((( (norm.cdf((ci_lower_cohens_dpop_Central)  ))  ) ))),50),3) # type: ignore
        results["Upper Central CI U3_dpop"] = ci_upper_cohens_U3_dpop_Central = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_cohens_dpop_Central) ))  ) )), ((( (norm.cdf(abs(ci_upper_cohens_dpop_Central)  ))  ) )))),100),3) # type: ignore
        results["Lower Pivotal CI U3_dpop"] = ci_lower_cohens_U3_dpop_Pivotal = np.round(max((100 * ((((norm.cdf(abs(ci_upper_Pivotal_dpop)))  ) )) if ((( (norm.cdf(abs(ci_upper_Pivotal_dpop)  ))  ) )) > ((( (norm.cdf((ci_upper_Pivotal_dpop)  ))  ) )) else ((( (norm.cdf((ci_lower_Pivotal_dpop)  ))  ) ))),50),3) # type: ignore
        results["Upper Pivotal CI U3_dpop"] = ci_upper_cohens_U3_dpop_Pivotal = np.round(min((100 * max(((((norm.cdf(abs(ci_lower_Pivotal_dpop)))  ) )), ((( (norm.cdf(abs(ci_upper_Pivotal_dpop)  ))  ) )))),100),3) # type: ignore
        results["Statistical Line Cohen's U3_dpop"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's U\u2083d\u209a\u2092\u209a = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(cohens_U3_dpop * 100),1), confidence_level_percentages , np.around(ci_lower_cohens_U3_dpop_Pivotal, 3),  np.around(ci_upper_cohens_U3_dpop_Pivotal, 3))

        results["Mcgraw and Wong, CLdpop"] = np.around(100 * (Mcgraw_Wong_CLdpop),4)
        results["Lower Central Ci Mcgraw and Wong, CLdpop"] = ci_lower_cldpop_Central = np.around(100 * (max(norm.cdf(ci_lower_cohens_dpop_Central / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Central Ci Mcgraw and Wong, CLdpop"] = ci_upper_cldpop_Central = np.around(100 * (min(norm.cdf(ci_upper_cohens_dpop_Central / np.sqrt(2)),1)),4) # type: ignore
        results["Lower Pivotal Ci Mcgraw and Wong, CLdpop"] = ci_lower_cldpop_Pivotal = np.around(100 * (max(norm.cdf(ci_lower_Pivotal_dpop / np.sqrt(2)),0)),4) # type: ignore
        results["Upper Pivotal Ci Mcgraw and Wong, CLdpop"] = ci_upper_cldpop_Pivotal = np.around(100 * (min(norm.cdf(ci_upper_Pivotal_dpop / np.sqrt(2)),1)),4) # type: ignore
        results["Statistical Line Mcgraw and Wong, CLdpop"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLd\u209a\u2092\u209a = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(Mcgraw_Wong_CLdpop * 100),1), confidence_level_percentages , np.around(ci_lower_cldpop_Pivotal, 3),  np.around(ci_upper_cldpop_Pivotal, 3))

        results["Proportion Of Overlap dpop"] = np.around(proportion_of_overlap_dpop, 4)
        results["Lower Central CI Proportion of Overlap_dpop"] = ci_lower_pov_Central_dpop = np.around((2 * norm.cdf(-abs(ci_lower_cohens_dpop_Central) / 2)), 4) # type: ignore
        results["Upper Central CI Proportion of Overlap_dpop"] = ci_upper_pov_Central_dpop = np.around((2 * norm.cdf(-abs(ci_upper_cohens_dpop_Central) / 2)), 4) # type: ignore
        results["Lower Pivotal CI Proportion of Overlap_dpop"] = ci_lower_pov_Pivotal_dpop = np.around((2 * norm.cdf(-abs(ci_lower_Pivotal_dpop) / 2)), 4) # type: ignore
        results["Upper Pivotal CI Proportion of Overlap_dpop"] = ci_upper_pov_Pivotal_dpop = np.around((2 * norm.cdf(-abs(ci_upper_Pivotal_dpop) / 2)), 4) # type: ignore
        results["Statistical Line Proportion of Overlap (POVdpop) "] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, POV\u209a\u2092\u209a = {:.3f}%, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, np.round(float(proportion_of_overlap_dpop),3), confidence_level_percentages , np.around(ci_lower_pov_Pivotal_dpop, 3),  np.around(ci_upper_pov_Pivotal_dpop, 3))

        # Aparametric Effect Sizes for Independent Samples
        results["_______________________________________"] = ""

        results["Proabability of Superiority Considering Ties (Vargha & Delaney)"] = round(POS_Vargha_Delaney, 4)
        results["Proabability of Superiority With Ties CI (Cliff)"] = [round(max(0, lower_ci_VDA_cliff), 4), round(min(1, upper_ci_VDA_cliff), 4)]
        results["Proabability of Superiority With Ties CI (Brunner-Munzel)"] = [round(max(0, LowerCi_VD_Brunner_Munzel), 4), round(min(1, UpperCi_VD_Brunner_Munzel), 4)]
        results["Proabability of Superiority With Ties CI (Traditional)"] = [round(max(0, ci_lower_Vargha_Delaney_Ruscio), 4), round(min(1, ci_upper_Vargha_Delaney_Ruscio), 4)]
        results["Proabability of Superiority With Ties CI (Hanley & McNeil 1)"] = [round(max(0, CI_VDA_Hanely_Mcneil_1_lower), 4), round(min(1, CI_VDA_Hanely_Mcneil_1_upper), 4)]
        results["Proabability of Superiority With Ties CI (Hanley & McNeil 2)"] = [round(max(0, CI_VDA_Hanely_Mcneil_2_lower), 4), round(min(1, CI_VDA_Hanely_Mcneil_2_upper), 4)]
        results["Proabability of Superiority With Ties CI (Fisher Transformed)"] = [round(max(0, Ci_PS_Vargha_Delaney_Fisher_lower), 4), round(min(1, Ci_PS_Vargha_Delaney_Fisher_upper), 4)]
        results["Proabability of Superiority With Ties CI (Metsamuuronen, 2021)"] = [round(max(0, min(ci_lower_VDA_Metsamuuronen, ci_upper_VDA_Metsamuuronen)), 4), round(min(1, max(ci_lower_VDA_Metsamuuronen, ci_upper_VDA_Metsamuuronen)), 4)]

        results["________________________________________"] = ""

        results["Proabability of Superiority Based on Gamma Correlation (Metsamuuronen)"] = round((1 - POS_Gamma_Based_Metsamuuronen), 4)
        results["Proabability of Superiority Based on Gamma Correlation CI (Metsamuuronen)"] = [round(max(0, min(ci_lower_Gamma_Metsamuuronen, ci_upper_Gamma_Metsamuuronen)), 4), round(min(1, max(ci_lower_Gamma_Metsamuuronen, ci_upper_Gamma_Metsamuuronen)), 4)]
        results["Proabability of Superiority Based on Gamma Correlation CI (Traditional)"] = [round(max(0, ci_lower_Gamma), 4), round(min(ci_upper_Gamma, 1), 4)]

        results["_________________________________________"] = ""

        results["Cliff's Delta"] = round(Cliffs_Delta, 4)
        results["Cliff's Delta Standard Error"] = round(np.sqrt(Cliffs_Variance), 4)
        results["Cliff's Delta CI (Cliff)"] = [round(max(2*(lower_ci_VDA_cliff-0.5), -1), 4), round(min(2*(upper_ci_VDA_cliff-0.5), 1), 4)]
        results["Cliff's Delta CI (Brunner-Munzel)"] = [round(max(2*(LowerCi_VD_Brunner_Munzel-0.5), -1), 4), round(min(2*(UpperCi_VD_Brunner_Munzel-0.5), 1), 4)]
        results["Cliff's Delta CI (Traditional)"] = [round(max(2*(ci_lower_Vargha_Delaney_Ruscio-0.5), -1), 4), round(min(2*(ci_upper_Vargha_Delaney_Ruscio-0.5), 1), 4)]
        results["Cliff's Delta CI (Hanley & McNeil 1)"] = [round(max(2*(CI_VDA_Hanely_Mcneil_1_lower-0.5), -1), 4), round(min(2*(CI_VDA_Hanely_Mcneil_1_upper-0.5), 1), 4)]
        results["Cliff's Delta CI (Hanley & McNeil 2)"] = [round(max(2*(CI_VDA_Hanely_Mcneil_2_lower-0.5), -1), 4), round(min(2*(CI_VDA_Hanely_Mcneil_2_upper-0.5), 1), 4)]
        results["Cliff's Delta CI (Fisher Transformed)"] = [round(max(2*(Ci_PS_Vargha_Delaney_Fisher_lower-0.5), -1), 4), round(min(2*(Ci_PS_Vargha_Delaney_Fisher_upper-0.5), 1), 4)]
        results["Cliff's Delta CI (Metsamuuronen, 2021)"] = [round(max(min(2*(ci_lower_VDA_Metsamuuronen-0.5), 2*(ci_upper_VDA_Metsamuuronen-0.5)), -1), 4), round(min(1, max(2*(ci_lower_VDA_Metsamuuronen-0.5), 2*(ci_upper_VDA_Metsamuuronen-0.5))), 4)]

        results["_______________________________________________"] = ""

        results["Genralized Odds Ratio (Agresti)"] = Generalizd_Odds_Ratio_Agresti
        results["Genralized Odds Ratio Considering Ties (Obrien & Castelloe)"] = Generalizd_Odds_Ratio_Churilov
        results["Lower CI Generalized Odds Ratio (Considering Ties)"] = CI_genodds_Churilov_lower
        results["Upper CI Generalized Odds Ratio (Considering Ties)"] = CI_genodds_Churilov_upper
        results["Lower CI Generalized Odds Ratio (Ignoring Ties)"] = CI_genodds_Agresti_lower
        results["Upper CI Generalized Odds Ratio (Ignoring Ties)"] = CI_genodds_Agresti_upper
        results["O'brien and Castelloe Lower CI Generalized Odds Ratio (Considering Ties)"] = CI_genodds_Null_Churilov_lower
        results["O'brien and Castelloe Upper CI Generalized Odds Ratio (Considering Ties)"] = CI_genodds_Null_Churilov_Upper
        results["O'brien and Castelloe Lower CI Generalized Odds Ratio (Ignoring Ties)"] = CI_genodds_Null_Agresti_lower
        results["O'brien and Castelloe Upper CI Generalized Odds Ratio (Ignoring Ties)"] = CI_genodds_Null_Agresti_Upper

        # Other Non-Parametric Effect Sizes
        results["Kraemer & Andrews Gamma"] = Kraemer_Andrews_Gamma
        results["Lower CI Kraemer & Andrews Gamma (Bootstrapping)"] = round(lower_ci_Kraemer_Andrews_Gamma_boot, 4)
        results["Upper CI Kraemer & Andrews Gamma (Bootstrapping)"] = round(upper_ci_Kraemer_Andrews_Gamma_boot, 4)
        results["Non Parametric Cohen's U3 (Hentschke & Stüttgen)"] = round(Hentschke_Stüttgen_U3,4)
        results["Lower CI Hentschke_Stüttgen_U3 (Bootstrapping)"] = round(lower_ci_Hentschke_Stüttgen_U3, 4)
        results["Upper CI Hentschke_Stüttgen_U3 (Bootstrapping)"] = round(upper_ci_Hentschke_Stüttgen_U3, 4)
        results["Non Parametric Cohen's U1 (Hentschke & Stüttgen)"] = round(Hentschke_Stüttgen_U1,4)
        results["Lower CI Hentschke_Stüttgen_U1 (Bootstrapping)"] = round(lower_ci_Hentschke_Stüttgen_U1, 4)
        results["Upper CI Hentschke_Stüttgen_U1 (Bootstrapping)"] = round(upper_ci_Hentschke_Stüttgen_U1, 4)
        results["Wilcox and Musaka's Q"] = Wilcox_Musaka_Q
        
        # Statistical Lines
        results["Statistical Line Kreamer And Andrew's Gamma"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Kreamer & Andrew's γ = {:.3f}, {}% CI(Boostrapping) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, Kraemer_Andrews_Gamma, confidence_level_percentages, lower_ci_Kraemer_Andrews_Gamma_boot, upper_ci_Kraemer_Andrews_Gamma_boot)
        results["Statistical Line Cohen's U1 (Non-Parametric)"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Hentschke Stüttgen U1 = {}, {}% CI [{}, {}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, str(round(Hentschke_Stüttgen_U1,3)).lstrip("0"),confidence_level_percentages,  str(round(((lower_ci_Hentschke_Stüttgen_U1)),3)).lstrip("0"), str(round(((upper_ci_Hentschke_Stüttgen_U1)),3)).lstrip("0"))
        results["Statistical Line Cohen's U3 (Non-Parametric)"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Hentschke Stüttgen U3 = {}, {}% CI [{}, {}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, str(round(Hentschke_Stüttgen_U3,3)).lstrip("0"),confidence_level_percentages,  str(round(((lower_ci_Hentschke_Stüttgen_U3)),3)).lstrip("0"), str(round(((upper_ci_Hentschke_Stüttgen_U3)),3)).lstrip("0"))





        return results

    
    # Things to Consider
    # 1. Dont forget not to allow negative values when Converting NCT to CI's
    # 2. Test What i Get for large difference between sample sizes
    # 3. Note that Sheskin himslef reported an outcome which is not consistent with softwares (also real sttiscs report this formulae for CI which is not recommended)
    # 4. For the probability of superiority for very large Samples (n > 10^6) wilcox suggested using the bp method for constructin CI to phat 
    # 5. For many effect sizes here the interval is between 0 and 1 - dont forget to adapt this
    # 7. Wilcox use Pratt method for Cliffs CI when p is equal to 0 or 1
    # 9. Which CI does not include here? (Fligner and policello (wilcox splus 1997), mee (only for grissom and kim with no ties), newcomb CI's and Rank-Welch (the original VDA paper CI's)) 
    # 10. Check out whether the traditional wald type Ruscio CI should be use the Z critical or the t critical value
    # 11. Consider adding confidence intervals based on the both the t-score and the z-score

