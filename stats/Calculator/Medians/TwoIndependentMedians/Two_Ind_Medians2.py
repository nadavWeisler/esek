
import numpy as np
import math
import itertools
from scipy.stats import binom, t, iqr, norm, median_abs_deviation, median_test, fisher_exact, mood
from scipy.stats.mstats import hdmedian
from astropy.stats import biweight_midvariance
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
        Inter_Quartile_Range_sample_1 = iqr(Column_1)
        Inter_Quartile_Range_sample_2 = iqr(Column_2)
        def pairwise_differences(x):      # a function to calcualte all pairwise comparison
            return [b - a for a, b in itertools.combinations(x, 2)]
        Pairwise_Vector_sample_1 = pairwise_differences(Column_1)
        Pairwise_Vector_sample_2 = pairwise_differences(Column_2)
        Qn_sample_1 = 2.2219 * (np.quantile(abs(np.array(Pairwise_Vector_sample_1)),0.25))
        Qn_sample_2 = 2.2219 * (np.quantile(abs(np.array(Pairwise_Vector_sample_2)),0.25))
        

        # Calculation of Effect Sizes for one Sample Median
        Median_Difference = (Median_sample_1 - Median_sample_2) - population_differnece

        # 1. Based on SD's (Thompson, 2007)
        Avarage_SD = np.sqrt((Standard_Deviation_sample_1**2 + Standard_Deviation_sample_2**2) / 2)
        d_mdns = Median_Difference / Avarage_SD

        # 2. Based on Pooled MAD Ricca & Blaine
        Median_Absolute_Deviations_Pooled = ((Sample_Size_1-1)*median_absolute_deviation_sample_1 + (Sample_Size_2-1)*median_absolute_deviation_sample_2) / (Sample_Size_1 + Sample_Size_2 - 2)
        Median_Absolute_Deviations_Pooled_Corrected = ((Sample_Size_1-1)*normal_corrected_median_absoloute_deviation_sample_1 + (Sample_Size_2-1)*normal_corrected_median_absoloute_deviation_sample_2) / (Sample_Size_2 + Sample_Size_2 - 2)
        d_mad_pooled = Median_Difference / Median_Absolute_Deviations_Pooled
        d_mad_pooled_corrected = Median_Difference / Median_Absolute_Deviations_Pooled_Corrected

        # 3. Quantile Shift of typical differences **Check why did Rand limits the function to n>10
        Pairwise_Comparisons = np.subtract.outer(Sample_Size_1, Sample_Size_2)
        Median_of_Comparisons = np.median(Pairwise_Comparisons)
        Quantile_Symmetric_Measure_Effect_Size = np.mean(Pairwise_Comparisons - Median_of_Comparisons <= Median_of_Comparisons)

        # 4. Mangiafico's d
        Avarage_MAD = np.sqrt( (median_absolute_deviation_sample_1**2 + median_absolute_deviation_sample_2**2) / 2)
        Avarage_MAD_Corrected = np.sqrt( (normal_corrected_median_absoloute_deviation_sample_1**2 + normal_corrected_median_absoloute_deviation_sample_2**2) / 2)
        Mangiaficos_d = ((Median_sample_1 - Median_sample_2) - population_differnece) / Avarage_MAD
        Mangiaficos_d_corrected = ((Median_sample_1 - Median_sample_2) - population_differnece) / Avarage_MAD_Corrected

        #5. Iqr Based
        Avarage_IQR =  np.sqrt(((Inter_Quartile_Range_sample_1*0.75)**2 + (Inter_Quartile_Range_sample_2*0.75)**2) /2 )       
        d_iqr = ((Median_sample_1 - Median_sample_2) - population_differnece) / Avarage_IQR

        #6. biweight midvariance based effect size
        Avarge_BW = np.sqrt(((biweight_midvariance(Column_1)**0.5)**2 + (biweight_midvariance(Column_2)**0.5)**2) /2 )        
        d_bw = ((Median_sample_1 - Median_sample_2) - population_differnece) / Avarge_BW


        return d_mdns, d_mad_pooled, d_mad_pooled_corrected, Quantile_Symmetric_Measure_Effect_Size, Mangiaficos_d, Mangiaficos_d_corrected, d_iqr, d_bw


class TwoPairedMedians():
    @staticmethod
    def Two_Paired_Medians_From_Data(params: dict) -> dict:
        
        # Set params
        population_difference = params["Difference in the Population"]
        number_of_bootstrap_samples = params["Number of Bootstrapping Smaples"]
        Column_1 = params["Column 1"]
        Column_2 = params["Column 2"]
        confidence_level_percentage = params["Confidence Level"]
        confidence_level = confidence_level_percentage / 100

        # Calculation

        ##############################
        # 1 - Descreptive Statistics #
        ##############################
        
        # 1A. Descreptive Statistics sample 1
        Median_sample_1 = np.median(Column_1)
        Mean_sample_1 = np.mean(Column_1)
        Sample_Size_1 = len(Column_1)
        Standard_Deviation_sample_1 = np.std(Column_1, ddof = 1)
        pairwise_averages_sample_1 = [(Column_1[i] + Column_1[j]) / 2.0 for i in range(len(Column_1)) for j in range(i, len(Column_1))]
        Pseudo_Median_sample_1 = np.median(pairwise_averages_sample_1) # This is the Hodghes lhemann estimator, note that other packages use different algorithm to estimate the median of the population (Minitab uses Johnson and T. Mizoguchi and stats package, Desc tools uses Monhan which is the same as the default stats package) 
        Harrell_Davis_Estimator_sample_1 = hdmedian(Column_1) 

        # 1B. Dispersion Measures sample 1  
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

        # 1C. Descreptive Statistics sample 2
        Median_sample_2 = np.median(Column_2)
        Mean_sample_2 = np.mean(Column_2)
        Sample_Size_2 = len(Column_2)
        Standard_Deviation_sample_2 = np.std(Column_2, ddof = 1)
        pairwise_averages_sample_2 = [(Column_2[i] + Column_2[j]) / 2.0 for i in range(len(Column_2)) for j in range(i, len(Column_2))]
        Pseudo_Median_sample_2 = np.median(pairwise_averages_sample_2) # This is the Hodghes lhemann estimator, note that other packages use different algorithm to estimate the median of the population (Minitab uses Johnson and T. Mizoguchi and stats package, Desc tools uses Monhan which is the same as the default stats package) 
        Harrell_Davis_Estimator_sample_2 = hdmedian(Column_2) 

        # 1D. Dispersion Measures sample 2  
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

        # 1E. Dispersion Measures of the Differences between Medians
        Avarage_SD = np.sqrt((Standard_Deviation_sample_1**2 + Standard_Deviation_sample_2**2) / 2)
        Median_Absolute_Deviations_Pooled = ((Sample_Size_1-1)*median_absolute_deviation_sample_1 + (Sample_Size_2-1)*median_absolute_deviation_sample_2) / (Sample_Size_1 + Sample_Size_2 - 2)
        Median_Absolute_Deviations_Pooled_Corrected = ((Sample_Size_1-1)*normal_corrected_median_absoloute_deviation_sample_1 + (Sample_Size_2-1)*normal_corrected_median_absoloute_deviation_sample_2) / (Sample_Size_2 + Sample_Size_2 - 2)
        Avarage_MAD = np.sqrt( (median_absolute_deviation_sample_1**2 + median_absolute_deviation_sample_2**2) / 2)
        Avarage_MAD_Corrected = np.sqrt( (normal_corrected_median_absoloute_deviation_sample_1**2 + normal_corrected_median_absoloute_deviation_sample_2**2) / 2)
        Avarage_IQR =  np.sqrt(((Inter_Quartile_Range_sample_1*0.75)**2 + (Inter_Quartile_Range_sample_2*0.75)**2) /2 )       
        Avarge_BW = np.sqrt(((biweight_midvariance(Column_1)**0.5)**2 + (biweight_midvariance(Column_2)**0.5)**2) /2 )        


        ##############################################
        # 2. Effect Sizes for two Independent Medians#
        ##############################################
        d_mdns, d_mad_pooled, d_mad_pooled_corrected, Quantile_Symmetric_Measure_Effect_Size, Mangiaficos_d, Mangiaficos_d_corrected, d_iqr, d_bw = effect_sizes_for_Indpednent_medians(Column_1, Column_2, population_difference)
        
        
        ###########################################################
        # 3. Median of the Difference and Its Confidence Intervals#
        ###########################################################
        Median_of_the_Difference = Median_sample_1 - Median_sample_2
        
        
        # 3A. Price Bonett
        zcrit = norm.ppf(1 - (1 - confidence_level)/2)
        
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

        # 3B. Hodges-Lehamnn Estimator CI's (Note Andrey's Suggested this problems check out at: https://aakinshin.net/posts/r-hodges-lehmann-problems/) 
        Hodges_Lehmann = desctools.MedianCI(ro.FloatVector(Column_1), ro.FloatVector(Column_2), conf_level = confidence_level)

        # 3C. Non-Parametric CI's based on the Mann Whitney Test
        Nonparametric_Exact_WMW_Based_CIs = rigr.wilcoxon(ro.FloatVector(Column_1), ro.FloatVector(Column_2), exact = True, correct = True,  conf_int = True, null_hypoth = population_difference, conf_level = confidence_level)[10][2:3]
        Nonparametric_WMW_Based_CIs = rigr.wilcoxon(ro.FloatVector(Column_1), ro.FloatVector(Column_2), conf_int = True, correct = True, null_hypoth = population_difference, conf_level = confidence_level)[10][2:3]
        Nonparametric_WMW_Corrected_Based_CIs  = rigr.wilcoxon(ro.FloatVector(Column_1), ro.FloatVector(Column_2), conf_int = True, correct = False, null_hypoth = population_difference, conf_level = confidence_level)[10][2:3]     # See Hollander and Wolfe, 1999 (Page 133-134)

        # 3D. Bootstrapping Confidence Intervals
        

        ############################
        # 4. Inferential Statisics #
        ############################

        # 4A. Mood's Median Test 
        Median_test_Chi_Square_Statistic, Meidan_test_p_value, grand_median, Contingency_Table = median_test(Column_1, Column_2)
        
        # 4B. Fisher Exact Test for 2X2 table
        Fisher_Stat, Fisher_p_value = fisher_exact(Contingency_Table)

        # 4C. Mood's Brown Test
        Mood_Median_test_Chi_Square_Statistic, Mood_Meidan_test_p_value = mood(Column_1, Column_2)

        # 4d. T based t-value
        Standard_Error_of_MAD = Median_Absolute_Deviations_Pooled_Corrected * np.sqrt(1/Sample_Size_1+ 1/Sample_Size_2)
        t_MAD_based = (Median_of_the_Difference - population_difference) / Standard_Error_of_MAD
        p_value_t_mad_based = t.sf(t_MAD_based, Sample_Size_1+Sample_Size_2-2) 

        # *** one can also suggest medpb2gen function of wilcox (check out the guide 2023) 


        ######################
        # 5. Ratio of Medians#
        ######################
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


        results["                                                                                                                                 "] = ""
        
        # Effect Sizes
        results["Effect Size ΔMADp "] = round(d_mad_pooled, 4)
        results["Effect Size ΔMADp Corrected "] = round(d_mad_pooled_corrected, 4)
        results["Effect Size Δsp "] = round(d_mdns, 4)
        results["Median Shift Effect Size "] = round(Quantile_Symmetric_Measure_Effect_Size, 4)
       
        results["                                                                                                                                  "] = ""

        # Mean Difference and Confidence Intervals
        results["Lower CI (Price-Bonett)"] = round(Lower_CI_Price_Bonett_between_difference, 4)
        results["Upper CI (Price-Bonett)"] = round(Upper_CI_Price_Bonett_between_difference, 4)

        results["                                                                                                                                    "] = ""


        # Ratio of Medians and Confidecne Intervals 
        results["Ratio of Medians"] = round(ratio_of_medians, 4)
        results["Lower CI Ratio of Medians"] = round(Lower_CI_Price_Bonett_between_ratio, 4)
        results["Upper CI Ratio of Medians"] = round(Upper_CI_Price_Bonett_between_ratio, 4)


        return results
    

    # Consider Using Andrey Akinshin HL estimator
    # https://aakinshin.net/posts/r-hodges-lehmann-problems/
        