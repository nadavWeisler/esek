import numpy as np
import math
import itertools
from scipy.stats import binom, t, wilcoxon, iqr, norm, median_abs_deviation
from scipy.stats.mstats import hdmedian
from astropy.stats import biweight_midvariance
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from arch.bootstrap import IndependentSamplesBootstrap
import rpy2.robjects as ro

# Relevant Functions for two indepdendent Medians
# R imports
desctools = importr("DescTools")
rigr = importr("rigr")

def effect_sizes_for_Indpednent_medians(Column_1, Column_2, population_differnece):
        
        Median_sample_1 = np.median(Column_1)
        Median_sample_2 = np.median(Column_2)
        Sample_Size_1 = len(Column_1)
        Sample_Size_2 = len(Column_2)
        Standard_Deviation_sample_1 = np.std(Column_1, ddof = 1)
        Standard_Deviation_sample_2 = np.std(Column_2, ddof = 1)
        median_absolute_deviation_sample_1 = median_abs_deviation(Column_1)
        median_absolute_deviation_sample_2 = median_abs_deviation(Column_2)
        normal_corrected_median_absoloute_deviation_sample_1 = median_absolute_deviation_sample_1 * 1.4826
        normal_corrected_median_absoloute_deviation_sample_2 = median_absolute_deviation_sample_2 * 1.4826
        
        # Calculation of Effect Sizes for one Sample Median
        
        Median_Difference = (Median_sample_1 - Median_sample_2) - population_differnece

        # 1. Based on SD's (Thompson, 2007)
        d_mdns = Median_Difference / np.sqrt((Standard_Deviation_sample_1**2 + Standard_Deviation_sample_2**2) / 2)

        # 2. Based on Pooled MAD Ricca & Blaine
        Median_Absolute_Deviations_Pooled = ((Sample_Size_1-1)*median_absolute_deviation_sample_1 + (Sample_Size_2-1)*median_absolute_deviation_sample_2) / (Sample_Size_1 + Sample_Size_2 - 2)
        Median_Absolute_Deviations_Pooled_Corrected = ((Sample_Size_1-1)*normal_corrected_median_absoloute_deviation_sample_1 + (Sample_Size_2-1)*normal_corrected_median_absoloute_deviation_sample_2) / (Sample_Size_2 + Sample_Size_2 - 2)
        d_mad_pooled = Median_Difference / Median_Absolute_Deviations_Pooled
        d_mad_pooled_corrected = Median_Difference / Median_Absolute_Deviations_Pooled

        # 3. Quantile Shift of typical differences **Check why did Rand limits the function to n>10
        Pairwise_Comparisons = np.subtract.outer(Sample_Size_1, Sample_Size_2)
        Median_of_Comparisons = np.median(Pairwise_Comparisons)
        Quantile_Symmetric_Measure_Effect_Size = np.mean(Pairwise_Comparisons - Median_of_Comparisons <= Median_of_Comparisons)

        return d_mdns, d_mad_pooled, d_mad_pooled_corrected, Quantile_Symmetric_Measure_Effect_Size


class TwoPairedMedians():
    @staticmethod
    def Two_Paired_Medians_From_Data(params: dict) -> dict:
        
        # Set params
        population_median = params["Difference in the Population"]
        Column_1 = params["Column 1"]
        Column_2 = params["Column 2"]
        confidence_level_percentage = params["Confidence Level"]

        # Calculation

        # A - Descreptive Statistics
        ############################
        confidence_level = confidence_level_percentage / 100

        # Smaple 1 - Descreptive Statistics and Dispersion Measures

        # Descreptive Statistics sample 1
        Median_sample_1 = np.median(Column_1)
        Mean_sample_1 = np.mean(Column_1)
        Sample_Size_1 = len(Column_1)
        Standard_Deviation_sample_1 = np.std(Column_1, ddof = 1)
        pairwise_averages_sample_1 = [(Column_1[i] + Column_1[j]) / 2.0 for i in range(len(Column_1)) for j in range(i, len(Column_1))]
        Pseudo_Median_sample_1 = np.median(pairwise_averages_sample_1) # This is the Hodghes lhemann estimator, note that other packages use different algorithm to estimate the median of the population (Minitab uses Johnson and T. Mizoguchi and stats package, Desc tools uses Monhan which is the same as the default stats package) 
        Harrell_Davis_Estimator_sample_1 = hdmedian(Column_1) 

        # Dispersion Measures sample 1  
        absolute_deviations_from_median_sample_1 = abs(Column_1-Median_sample_1)
        mean_absolute_deviation_sample_1 = np.mean(absolute_deviations_from_median_sample_1)
        normal_corrected_mean_absolout_deviation_sample_1 = mean_absolute_deviation_sample_1 * 1.2533
        median_absolute_deviation_sample_1 = np.median(absolute_deviations_from_median_sample_1)
        normal_corrected_median_absoloute_deviation_sample_1 = 1.4826*median_absolute_deviation_sample_1
        max_value_Sample_1 = np.max(Column_1)
        min_value_Sample_1 = np.min(Column_1)
        Range_Sample_1 = max_value_Sample_1 - min_value_Sample_1
        Inter_Quartile_Range_sample_1 = iqr(Column_1)

        def pairwise_differences(x):      # a function to calcualte all pairwise comparison
            return [b - a for a, b in itertools.combinations(x, 2)]
        Pairwise_Vector_sample_1 = pairwise_differences(Column_1)
        Qn_sample_1 = 2.2219 * (np.quantile(abs(np.array(Pairwise_Vector_sample_1)),0.25))


        # Smaple 2 - Descreptive Statistics and Dispersion Measures

        # Descreptive Statistics sample 2
        Median_sample_2 = np.median(Column_2)
        Mean_sample_2 = np.mean(Column_2)
        Sample_Size_2 = len(Column_2)
        Standard_Deviation_sample_2 = np.std(Column_2, ddof = 1)
        pairwise_averages_sample_2 = [(Column_2[i] + Column_2[j]) / 2.0 for i in range(len(Column_2)) for j in range(i, len(Column_2))]
        Pseudo_Median_sample_2 = np.median(pairwise_averages_sample_2) # This is the Hodghes lhemann estimator, note that other packages use different algorithm to estimate the median of the population (Minitab uses Johnson and T. Mizoguchi and stats package, Desc tools uses Monhan which is the same as the default stats package) 
        Harrell_Davis_Estimator_sample_2 = hdmedian(Column_2) 

        # Dispersion Measures sample 2  
        absolute_deviations_from_median_sample_2 = abs(Column_2-Median_sample_2)
        mean_absolute_deviation_sample_2 = np.mean(absolute_deviations_from_median_sample_2)
        normal_corrected_mean_absolout_deviation_sample_2 = mean_absolute_deviation_sample_2 * 1.2533
        median_absolute_deviation_sample_2 = np.median(absolute_deviations_from_median_sample_2)
        normal_corrected_median_absoloute_deviation_sample_2 = 1.4826*median_absolute_deviation_sample_2
        max_value_Sample_2 = np.max(Column_2)
        min_value_Sample_2 = np.min(Column_2)
        Range_Sample_2 = max_value_Sample_2 - min_value_Sample_2
        Inter_Quartile_Range_sample_2 = iqr(Column_2)

        Pairwise_Vector_sample_2 = pairwise_differences(Column_2)
        Qn_sample_2 = 2.2219 * (np.quantile(abs(np.array(Pairwise_Vector_sample_2)),0.25))

        # Effect Sizes
        Median_Difference = (Median_sample_1 - Median_sample_2) - population_median
        # 1. Based on SD's (Thompson, 2007)
        d_mdns = Median_Difference / np.sqrt((Standard_Deviation_sample_1**2 + Standard_Deviation_sample_2**2) / 2)

        # 2. Based on Pooled MAD Ricca & Blaine
        Median_Absolute_Deviations_Pooled = ((Sample_Size_1-1)*median_absolute_deviation_sample_1 + (Sample_Size_2-1)*median_absolute_deviation_sample_2) / (Sample_Size_1 + Sample_Size_2 - 2)
        Median_Absolute_Deviations_Pooled_Corrected = ((Sample_Size_1-1)*normal_corrected_median_absoloute_deviation_sample_1 + (Sample_Size_2-1)*normal_corrected_median_absoloute_deviation_sample_2) / (Sample_Size_2 + Sample_Size_2 - 2)
        d_mad_pooled = Median_Difference / Median_Absolute_Deviations_Pooled
        d_mad_pooled_corrected = Median_Difference / Median_Absolute_Deviations_Pooled

        # 3. Quantile Shift of typical differences **Check why did Rand limits the function to n>10
        Pairwise_Comparisons = np.subtract.outer(Sample_Size_1, Sample_Size_2)
        Median_of_Comparisons = np.median(Pairwise_Comparisons)
        Quantile_Symmetric_Measure_Effect_Size = np.mean(Pairwise_Comparisons - Median_of_Comparisons <= Median_of_Comparisons)

        
        # Inferential Statisics (# one also can suggest medpb2gen function of wilcox (check out the guide 2023) 
        #######################

        # Mood's Median Test 
        Chi_Square_Statistic, p_value = median_test(Column_1, Column_2)[0:2]

        

        # Confidecne Intervals for the difference between indpednent medians
        #######################################################################

 

        # 2. Price and Bonette
        zcrit = norm.ppf(1 - (1 - confidence_level))
        
        sorted_x = np.sort(Column_1)
        c1 = np.round((Sample_Size_1 + 1) / 2 - Sample_Size_1**0.5)
        X1 = sorted_x[int(Sample_Size_1 - c1)]
        X2 = sorted_x[int(c1 - 1)]
        Z1 = norm.ppf(1 - binom.cdf(c1 - 1, Sample_Size_1, 0.5))
        Variance_Sample_1 = (((X1 - X2) / (2 * Z1))**2)

        sorted_y = np.sort(Column_2)
        c2 = np.round((Sample_Size_2 + 1) / 2 - Sample_Size_2**0.5)
        Y1 = (sorted_y[int(Sample_Size_2 - c2)])
        Y2 = (sorted_y[int(c2 - 1)])
        Z2 = norm.ppf(1 - binom.cdf(c2 - 1, Sample_Size_2, 0.5))
        Variance_Sample_2 = ((Y1 - Y2) / (2 * Z2))**2

        Standard_Error_Indpednent_Differnece_Price_Bonett = np.sqrt(Variance_Sample_1 + Variance_Sample_2)
        Lower_CI_Price_Bonett_between_difference = (Median_sample_1 - Median_sample_2) - zcrit * Standard_Error_Indpednent_Differnece_Price_Bonett
        Upper_CI_Price_Bonett_between_difference = (Median_sample_1 - Median_sample_2) + zcrit * Standard_Error_Indpednent_Differnece_Price_Bonett


        # Ratio of Medians
        log_X1 = np.log(sorted_x[int(Sample_Size_1 - c1)])
        log_X2 = np.log(sorted_x[int(c1 - 1)])
        log_Y1 = np.log(sorted_y[int(Sample_Size_2 - c2)])
        log_Y2 = np.log(sorted_y[int(c2 - 1)])
        log_Z1 = norm.ppf(1 - binom.cdf(c1 - 1, Sample_Size_1, 0.5))
        log_Variance_Sample_1 = (((log_X1 - log_X2) / (2 * log_Z1))**2)
        log_Z2 = norm.ppf(1 - binom.cdf(c2 - 1, Sample_Size_2, 0.5))
        log_Variance_Sample_2 = ((log_Y1 - log_Y2) / (2 * log_Z2))**2
        ratio_of_medians = Median_sample_1 / Median_sample_2
        log_Standard_Error_Indpednent_Differnece_Price_Bonett = np.sqrt(log_Variance_Sample_1+log_Variance_Sample_2)
        Lower_CI_Price_Bonett_between_ratio = ratio_of_medians * math.exp(-zcrit * log_Standard_Error_Indpednent_Differnece_Price_Bonett)
        Upper_CI_Price_Bonett_between_ratio = ratio_of_medians * math.exp(zcrit * log_Standard_Error_Indpednent_Differnece_Price_Bonett)



        results = {}

        results["Sample 1 - Median"] = round(Median_sample_1, 4)
        results["Sample 1 - Mean"] = round(Mean_sample_1, 4)
        results["Sample 1 - Standard Deviation"] = round(Standard_Deviation_sample_1, 4)
        results["Sample 1 - Range"] = (f"R = [{min_value_Sample_1}-{max_value_Sample_1}]")
        results["Sample 1 - Inter Quartile Range"] = np.around(Inter_Quartile_Range_sample_1, 4)
        results["Sample 1 - Mean Absolute Deviation"] = round(mean_absolute_deviation_sample_1, 4)
        results["Sample 1 - Median Absolute Deviation"] = round(median_absolute_deviation_sample_1, 4)

        results["                                                                                                                              "] = ""


        # Descriptive Statistics for Sample 2
        results["Sample 2 - Median"] = round(Median_sample_2, 4)
        results["Sample 2 - Mean"] = round(Mean_sample_2, 4)
        results["Sample 2 - Standard Deviation"] = round(Standard_Deviation_sample_2, 4)
        results["Sample 2 - Range"] = (f"R = [{min_value_Sample_2}-{max_value_Sample_2}]")
        results["Sample 2 - Inter Quartile Range"] = np.around(Inter_Quartile_Range_sample_2, 4)
        results["Sample 2 - Mean Absolute Deviation"] = round(mean_absolute_deviation_sample_2, 4)
        results["Sample 2 - Median Absolute Deviation"] = round(median_absolute_deviation_sample_2, 4)

        results["                                                                                                                                "] = ""
        
        # Inferntial Statistics
        results["Standard Error (Median of Difference)"] = round(Standrd_Error_Median_of_difference, 4)
        results["Standard Error (Median AD)"] = round(Standrd_Error_Median_AD, 4)
        results["Standard Error (IQR)"] = round(Standrd_Error_IQR, 4)
        results["p-value Mood's Median Test"] = np.around(p_value, 4)
        results["Hodges-Lehmann Estimator (Indpendent Samples)"] = round(Hodges_Lehmann_estimator_Paired_samples, 4)
        results["Confidence Interval (HL Estimator)"] = (round(HL_CI[0], 4), round(HL_CI[1], 4))

        results["                                                                                                                                 "] = ""
        
        # Effect Sizes
        results["Effect Size ΔMADp "] = round(d_mad_pooled, 4)
        results["Effect Size ΔMADp Corrected "] = round(d_mad_pooled_corrected, 4)
        results["Effect Size Δsp "] = round(d_mdns, 4)
        results["Median Shift Effect Size "] = round(Quantile_Symmetric_Measure_Effect_Size, 4)
       
        results["                                                                                                                                  "] = ""

        # Mean Difference and Confidence Intervals
        results["Difference Between Medians"] = round(Difference_Between_Medians, 4)
        results["Lower CI"] = round(Lower_CI_difference, 4)
        results["Upper CI"] = round(Upper_CI_difference, 4)
        results["Lower CI (Price-Bonett)"] = round(Lower_CI_Price_Bonett_between_difference, 4)
        results["Upper CI (Price-Bonett)"] = round(Upper_CI_Price_Bonett_between_difference, 4)

        results["                                                                                                                                    "] = ""


        # Ratio of Medians and Confidecne Intervals 
        results["Ratio of Medians"] = round(ratio_of_medians, 4)
        results["Lower CI Ratio of Medians"] = round(Lower_CI_Price_Bonett_between_ratio, 4)
        results["Upper CI Ratio of Medians"] = round(Upper_CI_Price_Bonett_between_ratio, 4)







        return results
        