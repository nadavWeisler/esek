
#####################################################
## Effect Size for Two Paired Samples CLES  #########
#####################################################

import numpy as np
from scipy.stats import norm, nct, beta, t
import math 
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

# 2. Central CI Based on the Difference Paired Samples SE (see Fitts, 2020)
def calculate_central_ci_from_cohens_d_two_paired_sample_t_test(cohens_d, sample_size, confidence_level): # This is a function that calculates the Confidence Intervals of the Effect size in One Sample Z_score test (or two dependent samples)
    df = sample_size - 1 # This is the Degrees of Freedom for one sample t-test
    correction_factor =  math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
    standard_error_es = np.sqrt((df/(df-2)) * (1 / sample_size ) * (1 + cohens_d**2 * sample_size)  - (cohens_d**2 / correction_factor**2)) #This formula from Hedges, 1981, is the True formula for one sample t-test or depdnent samples based on the paired differences. Sample Size here is the number of pairs
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = cohens_d - standard_error_es * z_critical_value,  cohens_d + standard_error_es * z_critical_value
    return ci_lower, ci_upper, standard_error_es


class Two_Paired_Common_Language_Effect_Sizes():

    @staticmethod
    def Common_Language_Effect_Sizes_Paird_Samples_from_t_score(params: dict) -> dict:
        
        # Set Parameters
        t_score = params["t Score"]
        sample_size = params["Number of Pairs"]
        confidence_level_percentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentages  / 100

        # Mcgraw and Wong common language Effect size (based on the calculated dz)
        cohens_dz = t_score/np.sqrt(sample_size)
        cles_dz = norm.cdf(cohens_dz) * 100
        df = sample_size - 1
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
        hedges_gz = correction*cohens_dz
        cles_gz = norm.cdf(hedges_gz) * 100
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)

        # Confidence Intervals
        ci_lower_cohens_dz_central, ci_upper_cohens_dz_central, standard_error_cohens_dz =  calculate_central_ci_from_cohens_d_two_paired_sample_t_test (cohens_dz, sample_size, confidence_level)
        ci_lower_hedges_gz_central, ci_upper_hedges_gz_central, standard_error_hedges_gz =  calculate_central_ci_from_cohens_d_two_paired_sample_t_test (hedges_gz, sample_size, confidence_level)
        ci_lower_cohens_dz_Pivotal, ci_upper_cohens_dz_Pivotal =  Pivotal_ci_t (t_score, df, sample_size, confidence_level)
        ci_lower_hedges_gz_Pivotal, ci_upper_hedges_gz_Pivotal =  Pivotal_ci_t (t_score, df, sample_size, confidence_level)
  
        # Set Results
        results = {}
        results["Mcgraw & Wong, Common Language Effect Size (CLdz)"] = np.around(cles_dz, 4)
        results["Mcgraw & Wong, Unbiased Common Language Effect Size (CLgz)"] = np.around(cles_gz, 4)
        results["Lower Central CI's CLdz"] = np.around(norm.cdf(ci_lower_cohens_dz_central) * 100, 4)
        results["Upper Central CI's CLdz"] = np.around(norm.cdf(ci_upper_cohens_dz_central) *100, 4)
        results["Lower Central CI's CLgz"] = np.around(norm.cdf(ci_lower_hedges_gz_central) *100, 4)
        results["Upper Central CI's CLgz"] = np.around(norm.cdf(ci_upper_hedges_gz_central) *100, 4)
        results["Lower Pivotal CI's CLdz"] = np.around(norm.cdf(ci_lower_cohens_dz_Pivotal) *100, 4)
        results["Upper Pivotal CI's CLdz"] = np.around(norm.cdf(ci_upper_cohens_dz_Pivotal) *100, 4)
        results["Lower Pivotal CI's CLgz"] = np.around(norm.cdf(ci_lower_hedges_gz_Pivotal*correction) *100, 4)
        results["Upper Pivotal CI's CLgz"] = np.around(norm.cdf(ci_upper_hedges_gz_Pivotal*correction) *100, 4)
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"

        results["Statistical Line CLd"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLd = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cles_dz,confidence_level_percentages,  np.around(norm.cdf(ci_lower_cohens_dz_Pivotal) *100, 3),  np.around(norm.cdf(ci_upper_cohens_dz_Pivotal) *100, 3))
        results["Statistical Line CLg"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLg = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cles_gz,confidence_level_percentages,  np.around(norm.cdf(ci_lower_hedges_gz_Pivotal*correction) *100, 3), np.around(norm.cdf(ci_upper_hedges_gz_Pivotal*correction) *100, 3))
    
   
        return results


    @staticmethod
    def Common_Language_Effect_Sizes_Paird_Samples_from_parameters(params: dict) -> dict:
            
         # Set params
        sample_mean_1 = params["Sample Mean 1"]
        sample_mean_2 = params["Sample Mean 2"]
        sample_sd_1 = params["Standard Deviation Sample 1"]
        sample_sd_2 = params["Standard Deviation Sample 2"]
        sample_size = params["Number of Pairs"]
        population_mean_diff = params["Difference in the Population"] # The default value should be 0
        correlation = params ["Pearson Correlation"] # This one is crucial to calcualte the dz (otherwise return only dav)
        confidence_level_percentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentages  / 100
        difference = (sample_mean_1- sample_mean_2)
        mean_difference = np.mean(difference - population_mean_diff)

        # 1. Mcgraw and Wong common language Effect size (based on the calculated dz)
        standardizer_dz = np.sqrt(sample_sd_1**2 + sample_sd_2**2 - 2*correlation * sample_sd_1 * sample_sd_2)
        cohens_dz = mean_difference/standardizer_dz
        cles_dz = norm.cdf(cohens_dz) * 100
        t_score = cohens_dz* np.sqrt(sample_size)
        df = sample_size - 1
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
        hedges_gz = correction*cohens_dz
        cles_gz = norm.cdf(hedges_gz) * 100
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)

        # Confidence Intervals
        ci_lower_cohens_dz_central, ci_upper_cohens_dz_central, standard_error_cohens_dz =  calculate_central_ci_from_cohens_d_two_paired_sample_t_test (cohens_dz, sample_size, confidence_level)
        ci_lower_hedges_gz_central, ci_upper_hedges_gz_central, standard_error_hedges_gz =  calculate_central_ci_from_cohens_d_two_paired_sample_t_test (hedges_gz, sample_size, confidence_level)
        ci_lower_cohens_dz_Pivotal, ci_upper_cohens_dz_Pivotal =  Pivotal_ci_t (t_score, df, sample_size, confidence_level)
        ci_lower_hedges_gz_Pivotal, ci_upper_hedges_gz_Pivotal =  Pivotal_ci_t (t_score, df, sample_size, confidence_level)
   
        # Set Results
        results = {}
        results["Mcgraw & Wong, Common Language Effect Size (CLdz)"] = np.around(cles_dz, 4)
        results["Mcgraw & Wong, Unbiased Common Language Effect Size (CLgz)"] = np.around(cles_gz, 4)
        results["Lower Central CI's CLdz"] = np.around(norm.cdf(ci_lower_cohens_dz_central) * 100, 4)
        results["Upper Central CI's CLdz"] = np.around(norm.cdf(ci_upper_cohens_dz_central) *100, 4)
        results["Lower Central CI's CLgz"] = np.around(norm.cdf(ci_lower_hedges_gz_central) *100, 4)
        results["Upper Central CI's CLgz"] = np.around(norm.cdf(ci_upper_hedges_gz_central) *100, 4)
        results["Lower Pivotal CI's CLdz"] = np.around(norm.cdf(ci_lower_cohens_dz_Pivotal) *100, 4)
        results["Upper Pivotal CI's CLdz"] = np.around(norm.cdf(ci_upper_cohens_dz_Pivotal) *100, 4)
        results["Lower Pivotal CI's CLgz"] = np.around(norm.cdf(ci_lower_hedges_gz_Pivotal*correction) *100, 4)
        results["Upper Pivotal CI's CLgz"] = np.around(norm.cdf(ci_upper_hedges_gz_Pivotal*correction) *100, 4)
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line CLd"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLd = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cles_dz,confidence_level_percentages,  np.around(norm.cdf(ci_lower_cohens_dz_Pivotal) *100, 3),  np.around(norm.cdf(ci_upper_cohens_dz_Pivotal) *100, 3))
        results["Statistical Line CLg"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLg = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cles_gz,confidence_level_percentages,  np.around(norm.cdf(ci_lower_hedges_gz_Pivotal*correction) *100, 3), np.around(norm.cdf(ci_upper_hedges_gz_Pivotal*correction) *100, 3))
        
        return results


    @staticmethod
    def Common_Language_Effect_Sizes_Paird_Samples_from_data(params: dict) -> dict:
        
        # Set Parameters
        column_1 = params["Column 1"]
        column_2 = params["Column 2"]
        confidence_level_percentages = params["Confidence Level"]
        reps = params["Number of Bootstapping Samples"]

        # Calculation
        Sample_1_mean = np.mean(column_1)
        Sample_2_mean = np.mean(column_2)
        Sample_1_median = np.median(column_1)
        Sample_2_median = np.median(column_2)
        Sample_1_standard_deviation = np.std(column_1, ddof = 1)
        Sample_2_standard_deviation = np.std(column_2, ddof = 1)
        Sample_size_1 = len(column_1)
        Sample_size_2 =len(column_2)

        confidence_level = confidence_level_percentages  / 100
        difference = (column_1- column_2)
        mean_difference = np.mean(difference)
        standard_deviation_of_the_differnece = np.std(difference, ddof = 1)
        sample_size = len(difference)
        
        # 1. Mcgraw and Wong common language Effect size (based on the calculated dz)
        cohens_dz = mean_difference/standard_deviation_of_the_differnece
        cles_dz = norm.cdf(cohens_dz) * 100
        t_score = cohens_dz* np.sqrt(sample_size)
        df = sample_size - 1
        correction = math.exp(math.lgamma(df/2) - math.log(math.sqrt(df/2)) - math.lgamma((df-1)/2))
        hedges_gz = correction*cohens_dz
        cles_gz = norm.cdf(hedges_gz) * 100
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)

        # Confidence Intervals
        ci_lower_cohens_dz_central, ci_upper_cohens_dz_central, standard_error_cohens_dz =  calculate_central_ci_from_cohens_d_two_paired_sample_t_test (cohens_dz, sample_size, confidence_level)
        ci_lower_hedges_gz_central, ci_upper_hedges_gz_central, standard_error_hedges_gz =  calculate_central_ci_from_cohens_d_two_paired_sample_t_test (hedges_gz, sample_size, confidence_level)
        ci_lower_cohens_dz_Pivotal, ci_upper_cohens_dz_Pivotal =  Pivotal_ci_t (t_score, df, sample_size, confidence_level)
        ci_lower_hedges_gz_Pivotal, ci_upper_hedges_gz_Pivotal =  Pivotal_ci_t (t_score, df, sample_size, confidence_level)

        # Aparametric Common Language Effect Sizes
        ###########################################

        # 1. Probabilty of superiority PSdep (Grissom and Kim)
        Count_Group1_Larger = sum(np.where(difference > 0, 1, 0))
        Count_Group2_Larger = sum(np.where(difference < 0, 1, 0))
        Count_ties = sum(np.where(difference == 0, 1, 0))
        PSdep = Count_Group1_Larger/sample_size
        
        # 2. Vargha and Delany Axy, 2000 Considering ties: (the dependent groups version)
        superiority_counts = np.where(column_1 > column_2, 1, np.where(column_1 < column_2, 0, 0.5)) #This line  gives 1 for superior values, 0 to inferior values and 0.5 to ties. 
        VDA_xy = sum(superiority_counts)/sample_size

        # 3. Cliffs Delta dependent version (Feng ordinal delta, 2007)
        cliffs_delta = (sum(np.where(column_1>column_2,1,0)) - sum(np.where(column_2>column_1,1,0)))/sample_size

        # Confidcen Intervals for Nonparametric Measures
        # 1. Confidence Intervals for PSdep using the Pratt Method
        ctagsquare = norm.ppf(1 - confidence_level)**2
        ctag = norm.ppf(1 - confidence_level)
        A = ((Count_Group1_Larger + 1)/(sample_size-Count_Group1_Larger))**2
        B = 81 * (Count_Group1_Larger+1) * (sample_size-Count_Group1_Larger) - 9*sample_size - 8
        C = -3 * ctag * np.sqrt(9*(Count_Group1_Larger+1)*(sample_size - Count_Group1_Larger) * (9*sample_size + 5 - ctagsquare) + sample_size + 1)
        D = 81 * (Count_Group1_Larger + 1)**2 - 9 *(Count_Group1_Larger+1)* (2+ctagsquare) + 1
        E = 1 + A * ((B+C)/D)**3
        A2 = (Count_Group1_Larger/ (sample_size-Count_Group1_Larger-1)) **2
        B2 = 81 * (Count_Group1_Larger) * (sample_size-Count_Group1_Larger-1) - 9*sample_size - 8
        C2 = 3 * ctag * np.sqrt(9*Count_Group1_Larger*(sample_size-Count_Group1_Larger-1) * (9*sample_size + 5 - ctagsquare) + sample_size + 1)
        D2 = 81 * Count_Group1_Larger**2 - (9 *Count_Group1_Larger) * (2+ctagsquare) + 1
        E2 = 1 + A2 * ((B2+C2)/D2)**3
        
        upper_ci_psdep_Pratt = 1/E2
        lower_ci_psdep_Pratt = 1/E    

        if Count_Group1_Larger == 1:
            lower_ci_psdep_Pratt = 1 - (1-confidence_level)**(1/sample_size)
            upper_ci_psdep_Pratt = 1 - (confidence_level)**(1/sample_size)

        if Count_Group1_Larger == 0:
            lower_ci_psdep_Pratt = 0
            upper_ci_psdep_Pratt = beta.ppf(1-confidence_level, Count_Group1_Larger+1, sample_size-Count_Group1_Larger)

        if Count_Group1_Larger == sample_size-1:
            lower_ci_psdep_Pratt = (confidence_level)**(1/sample_size)
            upper_ci_psdep_Pratt = (1-confidence_level)**(1/sample_size)
        
        if Count_Group1_Larger == sample_size:
            lower_ci_psdep_Pratt = (confidence_level*2)**(1/sample_size)
            upper_ci_psdep_Pratt = 1

        if lower_ci_psdep_Pratt < 0:
            lower_ci_psdep_Pratt = 0
        if upper_ci_psdep_Pratt > 1:
            upper_ci_psdep_Pratt = 1
     
        # Confidence Interval for Cliff's Delta Within
        critical_z_value = norm.ppf(0.05 / 2)
        critical_t_value = t.ppf(0.05 / 2, (sample_size-1))

        feng_standard_error = np.sqrt(np.sum((np.sign(difference) - cliffs_delta)**2) / (sample_size*(sample_size - 1)))
        upper_ci_cliff = (cliffs_delta - cliffs_delta**3 - critical_t_value * feng_standard_error * np.sqrt((1 - cliffs_delta**2)**2 + critical_t_value**2 * feng_standard_error**2)) / (1 - cliffs_delta**2 + critical_t_value**2 * feng_standard_error**2)
        lower_ci_cliff = (cliffs_delta - cliffs_delta**3 + critical_t_value * feng_standard_error * np.sqrt((1 - cliffs_delta**2)**2 + critical_t_value**2 * feng_standard_error**2)) / (1 - cliffs_delta**2 + critical_t_value**2 * feng_standard_error**2)

        # Other Common Language Effect Sizes and Bootstrapping CI's
        
        # 4. Kraemer & Andrews Gamma (Degree of Overlap)
        Number_of_cases_x_larger_than_median_y = sum(1 for val in column_1 if val > Sample_2_median)
        Aparametric_Cohens_U3_no_ties = (Number_of_cases_x_larger_than_median_y / sample_size)
        if Aparametric_Cohens_U3_no_ties == 0:   Aparametric_Cohens_U3_no_ties = 1 / (sample_size + 1)
        elif Aparametric_Cohens_U3_no_ties == 1: Aparametric_Cohens_U3_no_ties = sample_size / (sample_size + 1)
        Kraemer_Andrews_Gamma = norm.ppf(Aparametric_Cohens_U3_no_ties)
        
        # 5. Hentschke & Stüttgen U3 (Aparametric Version of U3)
        Number_of_cases_x_equal_to_median_y = sum(1 for val in column_1 if val == Sample_2_median)
        Hentschke_Stüttgen_U3 = ((Number_of_cases_x_larger_than_median_y + Number_of_cases_x_equal_to_median_y * 0.5) / sample_size) #This is a non Parametric U3 as it called by the authors and its the tied consideration version of K & A gamma
        if Sample_1_median == Sample_2_median: Hentschke_Stüttgen_U3 = 0.5

        # 6. Hentschke & Stüttgen U1 (Aparametric Version of U1)
        Number_of_cases_x_larger_than_maximum_y = sum(1 for val in column_1 if val > np.max(column_2))
        Number_of_cases_x_smaller_than_minimum_y = sum(1 for val in column_1 if val < np.min(column_2))
        Hentschke_Stüttgen_U1 = (Number_of_cases_x_larger_than_maximum_y + Number_of_cases_x_smaller_than_minimum_y) / sample_size

        # 7. Wilcox and Musaka's Q
        eta = 0

        h1 = max((1.2 * (np.percentile(column_1, 75) - np.percentile(column_1, 25))) / (sample_size ** (1 / 5)), 0.05)
        h2 = max((1.2 * (np.percentile(column_2, 75) - np.percentile(column_2, 25))) / (sample_size ** (1 / 5)), 0.05)


        for Value in column_1: 
            f_x1 = (np.sum(column_1 <= (Value + h1)) - np.sum(column_1 < (Value - h1))) / (2 * sample_size * h1)
            f_x2 = (np.sum(column_2 <= (Value + h2)) - np.sum(column_2 < (Value - h2))) / (2 * sample_size * h2)
            if f_x1 > f_x2: eta += 1

        Wilcox_Musaka_Q_dep = eta / sample_size




        # Bootstrapping CI's for other common language effect sizes
        Bootstrap_Samples_x = []
        for _ in range(reps):
            # Generate bootstrap samples
            sample_1_bootstrap = np.random.choice(column_1, len(column_1), replace=True)
            difference_bootstrapping = np.random.choice(column_1-column_2, len(column_1), replace=True)
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
        
        # Set Results
        results = {}

        # Descreptive Statistics
        results["Median"] = (Number_of_cases_x_larger_than_median_y)
        results["Median 1"] = np.round(Sample_1_median,4)
        results["Median 2"] = np.round(Sample_2_median,4)
        results["Mean 1"] = np.round(Sample_1_mean,4)
        results["Mean 2"] = np.round(Sample_2_mean,4)
        results["Standard Deviation 1"] = np.round(Sample_1_standard_deviation,4)
        results["Standard Deviation 2"] = np.round(Sample_2_standard_deviation,4)
        results["Range 1"] = np.array(np.min(column_1), np.max(column_1))
        results["Range 2"] =  np.array(np.min(column_2), np.max(column_2))

        results["Mcgraw & Wong, Common Language Effect Size (CLdz)"] = np.around(cles_dz, 4)
        results["Mcgraw & Wong, Unbiased Common Language Effect Size (CLgz)"] = np.around(cles_gz, 4)
        results["Lower Central CI's CLdz"] = np.around(norm.cdf(ci_lower_cohens_dz_central) * 100, 4)
        results["Upper Central CI's CLdz"] = np.around(norm.cdf(ci_upper_cohens_dz_central) *100, 4)
        results["Lower Central CI's CLgz"] = np.around(norm.cdf(ci_lower_hedges_gz_central) *100, 4)
        results["Upper Central CI's CLgz"] = np.around(norm.cdf(ci_upper_hedges_gz_central) *100, 4)
        results["Lower Pivotal CI's CLdz"] = np.around(norm.cdf(ci_lower_cohens_dz_Pivotal) *100, 4)
        results["Upper Pivotal CI's CLdz"] = np.around(norm.cdf(ci_upper_cohens_dz_Pivotal) *100, 4)
        results["Lower Pivotal CI's CLgz"] = np.around(norm.cdf(ci_lower_hedges_gz_Pivotal*correction) *100, 4)
        results["Upper Pivotal CI's CLgz"] = np.around(norm.cdf(ci_upper_hedges_gz_Pivotal*correction) *100, 4)

        results["---------------------------------------------------------------------------------------------"] = ""
        results[" "] = ""
        results["Non Parametric CLES"] = ""
        results["-------------------------------------------------------------------------------------------"] = ""
        results["Number of Pairs where Group 1 is Larger"] = Count_Group1_Larger
        results["Number of Pairs where Group 2 is Larger"] = Count_Group2_Larger
        results["Number of Ties "] = Count_ties
        results["Probability of Superiority (PSdep - Grissom & Kim)"] = round(PSdep, 4)
        results["Lower CI PSdep (Pratt Method)"] = lower_psdep = np.round(min(lower_ci_psdep_Pratt,upper_ci_psdep_Pratt),4) # type: ignore
        results["Upper CI Psdep (Pratt Method)"] = upper_psdep = np.round(max(lower_ci_psdep_Pratt,upper_ci_psdep_Pratt),4) # type: ignore
        results["Vargha & Delany A"] = round(VDA_xy, 4)
        results["Lower CI VDAdep (Feng's Method)"] = round((lower_ci_cliff + 1)/2 ,4)
        results["Upper CI VDAdep (Feng's Method)"] = round((upper_ci_cliff + 1)/2 ,4)
        results["Cliff's Delta (for Dependent Groups)"] = round(cliffs_delta, 4)
        results["Lower CI Cliff's Delta (Feng's Method)"] = round(lower_ci_cliff,4)
        results["Upper CI Cliff's Delta (Feng's Method)"] = round(upper_ci_cliff,4)

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
        results["Wilcox and Musaka's Q"] = Wilcox_Musaka_Q_dep

        # Statistical Lines
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line CLd"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLd = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cles_dz,confidence_level_percentages,  np.around(norm.cdf(ci_lower_cohens_dz_Pivotal) *100, 3),  np.around(norm.cdf(ci_upper_cohens_dz_Pivotal) *100, 3))
        results["Statistical Line CLg"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, CLg = {:.1f}%, {}% CI(Pivotal) [{:.1f}%, {:.1f}%]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, cles_gz,confidence_level_percentages,  np.around(norm.cdf(ci_lower_hedges_gz_Pivotal*correction) *100, 3), np.around(norm.cdf(ci_upper_hedges_gz_Pivotal*correction) *100, 3))
        results["Cliff's Δdep"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Cliff's Δdep = {}, {}% CI [{}, {}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, (('-' if str(cliffs_delta).startswith('-') else '') + str(round(cliffs_delta,3)).lstrip('-').lstrip('0') or '0'),confidence_level_percentages, (('-' if str(lower_ci_cliff).startswith('-') else '') + str(round(lower_ci_cliff,3)).lstrip('-').lstrip('0') or '0'),  (('-' if str(upper_ci_cliff).startswith('-') else '') + str(round(upper_ci_cliff,3)).lstrip('-').lstrip('0') or '0'))
        results["Statistical Line VDAdep"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, VDAdep = {}, {}% CI [{}, {}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, str(round(VDA_xy,3)).lstrip("0"),confidence_level_percentages,  str(round(((lower_ci_cliff + 1)/2),3)).lstrip("0"), str(round(((upper_ci_cliff + 1)/2),3)).lstrip("0"))
        results["Statistical Line PSdep"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, PSdep = {}, {}% CI [{}, {}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, str(round(PSdep,3)).lstrip("0"),confidence_level_percentages,  str(round((float(lower_psdep)),3)).lstrip("0"), str(round((float(upper_psdep)),3)).lstrip("0"))
        results["Statistical Line Kreamer And Andrew's Gamma"] = " \033[3mt\033[0m({}) = {:.3f}, {}{}, Kreamer & Andrew's γ = {:.3f}, {}% CI(Boostrapping) [{:.3f}, {:.3f}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, Kraemer_Andrews_Gamma, confidence_level_percentages, lower_ci_Kraemer_Andrews_Gamma_boot, upper_ci_Kraemer_Andrews_Gamma_boot)
        results["Statistical Line Cohen's U1 (Non-Parametric)"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Hentschke Stüttgen U1 = {}, {}% CI [{}, {}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, str(round(Hentschke_Stüttgen_U1,3)).lstrip("0"),confidence_level_percentages,  str(round(((lower_ci_Hentschke_Stüttgen_U1)),3)).lstrip("0"), str(round(((upper_ci_Hentschke_Stüttgen_U1)),3)).lstrip("0"))
        results["Statistical Line Cohen's U3 (Non-Parametric)"] = "\033[3mt\033[0m({}) = {:.3f}, {}{}, Hentschke Stüttgen U3 = {}, {}% CI [{}, {}]".format(df, t_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, str(round(Hentschke_Stüttgen_U3,3)).lstrip("0"),confidence_level_percentages,  str(round(((lower_ci_Hentschke_Stüttgen_U3)),3)).lstrip("0"), str(round(((upper_ci_Hentschke_Stüttgen_U3)),3)).lstrip("0"))

        return results



#######
    
# Things to Consider

# 1. Note that for the Parametric Effect Sizes one can use either Z or t-distribution to calculate CLES - maybe consider to allow both
# 2. Consider using a parametric version that will be based on dav and dmle
# 3. Verify the connection here between vda and cliff's delta when there are ties in the data
# 4. Few important things about the cliff's delta CI (also can apply it to indpednent samples): i need to choose whether to claclate the CI on the t or z distibrution or to give two options..also i need to decide wehther to allow or not symmetric CI's and not the corrected version suggested by feng long and cliff. the Package Orddom is by far the best one so far for Cliff's delta...
# 5. For More confidence Intervals one can use the Ruscio and Mullen (2012) BCA method (see the relvant package by ortelli and ruscio).

# Long, J. D., Feng, D., & Cliff, N. (2003). Ordinal analysis of behavioral data. In J. A. Schinka & W. F. Velicer (Eds.), Handbook of psychology: Research methods in psychology, Vol. 2, pp. 635–661). John Wiley & Sons, Inc.. https://doi.org/10.1002/0471264385.wei0225
# Feng, D. (2007). Robustness and power of ordinal d for paired data. Real data analysis, 163-183.