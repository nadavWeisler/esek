
###############################################
##### Robust Effect Size for Paired Samples ###
###############################################

import numpy as np
from scipy.stats import t, norm, trim_mean

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


#This function caculates the Yuen Statistics for the Trimmed Means
# See Wilcox Solution when epow effect size is greater than 1 i just equated it to 1. 
 
class Robust_Paired():
    @staticmethod    
    def Robust_Paired_from_data(params: dict) -> dict:
        
        # Set Parameters
        column_1 = params["column_1"]
        column_2 = params["column_2"]
        confidence_level_percentages = params["Confidence Level"]
        trimming_level = params["Trimming Level"] #The default should be 0.2
        Population_Difference = params["Difference in the Population"] #The default should be 0.2
        reps = params["Number of Bootstrap Samples"]
        
        # Calculation
        confidence_level = confidence_level_percentages / 100
        difference = column_1 - column_2
        sample_size = len(column_1)
        correction = np.sqrt(area_under_function(density, norm.ppf(trimming_level), norm.ppf(1 - trimming_level))+ 2 * (norm.ppf(trimming_level) ** 2) * trimming_level)
        trimmed_mean_1 = trim_mean(column_1, trimming_level)
        trimmed_mean_2 = trim_mean(column_2, trimming_level)
        Winsorized_Standard_Deviation_1 = np.sqrt(WinsorizedVariance(column_1))
        Winsorized_Standard_Deviation_2 = np.sqrt(WinsorizedVariance(column_2))
        winsorized_correlation = WinsorizedCorrelation(column_1, column_2, trimming_level)['cor']
        winsorized_correlation_pvalue = WinsorizedCorrelation(column_1, column_2, trimming_level)['p.value']
        
        # Algina, Penfield, Kesselman robust effect size (AKP)
        Standardizer = np.sqrt(WinsorizedVariance(difference, trimming_level))
        trimmed_mean = trim_mean(difference, trimming_level)
        akp_effect_size = correction * (trimmed_mean - Population_Difference) / Standardizer

        # Confidence Intervals for AKP effect size using Bootstrapping
        Bootstrap_Samples_difference = []
        for _ in range(reps):
            # Generate bootstrap samples
            difference_bootstrap = np.random.choice(difference, len(difference), replace=True)
            Bootstrap_Samples_difference.append(difference_bootstrap)

        Trimmed_means_of_Bootstrap = (trim_mean(Bootstrap_Samples_difference, trimming_level, axis=1))
        Standardizers_of_Bootstrap = np.sqrt([WinsorizedVariance(array, trimming_level) for array in Bootstrap_Samples_difference])
        AKP_effect_size_Bootstrap = (correction * (Trimmed_means_of_Bootstrap - Population_Difference) / Standardizers_of_Bootstrap)
        lower_ci_akp_boot = np.percentile(AKP_effect_size_Bootstrap, ((1 - confidence_level) - ((1 - confidence_level)/2)) * 100)
        upper_ci_akp_boot = np.percentile(AKP_effect_size_Bootstrap, ((confidence_level) + ((1 - confidence_level)/2)) * 100)

        # Explanatory Measure of Effect Size
        sort_values = np.concatenate((column_1, column_2))
        Variance_Between_Trimmed_Means = (np.std(np.array([trimmed_mean_1, trimmed_mean_2]), ddof=1))**2 # This is the equivalence for SSeffect
        Winzorized_Variance = WinsorizedVariance(sort_values, trimming_level) # This is the equivalence for SStotal
        Explained_variance = Variance_Between_Trimmed_Means/(Winzorized_Variance/correction**2)
        Explanatory_Power_Effect_Size = min(np.sqrt(Explained_variance),1)

        # Bootstrapp Confidence Intervals for the Explanatory Power Effect Size
        Bootstrap_Samples_x = []
        Bootstrap_Samples_y = []
        for _ in range(reps):
            # Generate bootstrap samples
            sample_1_bootstrap = np.random.choice(column_1, len(column_1), replace=True)
            sample_2_bootstrap = np.random.choice(column_2, len(column_2), replace=True)
            Bootstrap_Samples_x.append(sample_1_bootstrap)
            Bootstrap_Samples_y.append(sample_2_bootstrap)
        
        concatenated_samples = [np.concatenate((x, y)) for x, y in zip(Bootstrap_Samples_x, Bootstrap_Samples_y)]
        Trimmed_means_of_Bootstrap_sample_1 = np.array((trim_mean(Bootstrap_Samples_x, trimming_level, axis=1)))
        Trimmed_means_of_Bootstrap_sample_2 = np.array((trim_mean(Bootstrap_Samples_y, trimming_level, axis=1)))
        Variance_Between_Trimmed_Means_Bootstrap = [ (np.std(np.array([x, y]), ddof=1))**2 for x, y in zip(Trimmed_means_of_Bootstrap_sample_1, Trimmed_means_of_Bootstrap_sample_2)]
        Winsorized_Variances_bootstrapp = ([WinsorizedVariance(arr, trimming_level) for arr in concatenated_samples])
        Explained_Variance_Bootstrapping = np.array(Variance_Between_Trimmed_Means_Bootstrap/(Winsorized_Variances_bootstrapp/correction**2))
        Explanatory_Power_Effect_Size_Bootstrap = [array**0.5 for array in Explained_Variance_Bootstrapping]
        lower_ci_epow_boot = np.percentile(Explanatory_Power_Effect_Size_Bootstrap, ((1 - confidence_level) - ((1 - confidence_level)/2)) * 100)
        upper_ci_epow_boot = np.percentile(Explanatory_Power_Effect_Size_Bootstrap, ((confidence_level) + ((1 - confidence_level)/2)) * 100)

        # Yuen Test Statistics 
        q1 = (len(column_1) - 1) * Winsorized_Standard_Deviation_1**2
        q2 = (len(column_2) - 1) * Winsorized_Standard_Deviation_2**2
        q3 = (len(column_1) - 1) * WinsorizedCorrelation(column_1, column_2, trimming_level)['cov']
        non_winsorized_sample_size = len(column_1) - 2 * np.floor(trimming_level * len(column_1))
        df = non_winsorized_sample_size - 1
        Yuen_Standrd_Error = np.sqrt((q1 + q2 - 2 * q3) / (non_winsorized_sample_size * (non_winsorized_sample_size - 1)))    
        difference_trimmed_means = trimmed_mean_1 - trimmed_mean_2
        Yuen_Statistic = difference_trimmed_means / Yuen_Standrd_Error
        Yuen_p_value = 2 * (1 - t.cdf(np.abs(Yuen_Statistic), df))
        

        # Set results
        results = {}
        
        results["Explanatory Power Effect Size (Wilcox & Tian, 2011)"] = round(min(Explanatory_Power_Effect_Size,1.0),4)
        results["Lower CI (Bootstrapping) Explanatory Power Effect Size"] = round(lower_ci_epow_boot,4)
        results["Upper CI (Bootstrapping) Explanatory Power Effect Size"] = round(min(upper_ci_epow_boot,1.0),4)
        results["Robust Effect Size AKP"] = round(akp_effect_size, 4)
        results["Lower Confidence Interval Robust AKP"] = round(lower_ci_akp_boot, 4)
        results["Upper Confidence Interval Robust AKP"] = round(upper_ci_akp_boot, 4)
        
        # Descreptive Statistics
        results["Trimmed Mean 1"] = round(trimmed_mean_1, 4)
        results["Trimmed Mean 2"] = round(trimmed_mean_2, 4)
        results["Winsorized Standard Deviation 1"] = round(Winsorized_Standard_Deviation_1, 4)
        results["Winsorized Standard Deviation 2"] = round(Winsorized_Standard_Deviation_2, 4)
        results["Number of Pairs"] = round(sample_size, 4)
        
        #Inferntial Statistic Table
        results["Yuen's T statistic"] = round(Yuen_Statistic, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["p-value"] = np.around(Yuen_p_value, 4)
        results["Difference Between Means"] = round(difference_trimmed_means, 4)
        results["Standard Error"] = round(Yuen_Standrd_Error, 4)
        results["Winosrized Correlation"] = round(winsorized_correlation, 4)
        results["p-value"] = round(winsorized_correlation_pvalue, 4)
        results["Correction"] = round(winsorized_correlation_pvalue, 4)

        results["Explained Variance"] = round(Explained_variance, 4)
        results["Lower Bootstrap CI's"] = (lower_ci_epow_boot)
        results["Upper Bootstrap CI's"] = (upper_ci_epow_boot)

        formatted_p_value = "{:.3f}".format(Yuen_p_value).lstrip('0') if Yuen_p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Robust AKP effect size"] = "Yuen's \033[3mt\033[0m({}) = {:.3f}, {}{}, \033[3mAKP\033[0m = {:.3f}, {}% CI(Bootstrap) [{:.3f}, {:.3f}]".format(((int(df) if float(df).is_integer() else df)), Yuen_Statistic, '\033[3mp = \033[0m' if Yuen_p_value >= 0.001 else '', formatted_p_value, akp_effect_size, confidence_level_percentages, round(lower_ci_akp_boot,3), round(upper_ci_akp_boot,3))
        results["Statistical Line Robust Explanatory Effect Size"] = "Yuen's \033[3mt\033[0m({}) = {:.3f}, {}{}, Wilcox & Tian's ξ = {}, {}% CI(Bootstrap) [{}, {}]".format(((int(df) if float(df).is_integer() else df)), Yuen_Statistic, '\033[3mp = \033[0m' if Yuen_p_value >= 0.001 else '', formatted_p_value, (str(round(Explanatory_Power_Effect_Size,3)).lstrip("0")), confidence_level_percentages, (str(round(lower_ci_epow_boot,3)).lstrip("0")), (str(round(upper_ci_epow_boot,3)).lstrip("0")))

        return results
 






