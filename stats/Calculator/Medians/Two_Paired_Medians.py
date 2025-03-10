import numpy as np
import math
import itertools
from scipy.stats import binom, t, wilcoxon, iqr, norm
from scipy.stats.mstats import hdmedian
from astropy.stats import biweight_midvariance
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from arch.bootstrap import IndependentSamplesBootstrap
import rpy2.robjects as ro
from statsmodels.stats.descriptivestats import sign_test

# R imports
desctools = importr("DescTools")
rigr = importr("rigr")


# Relevant Functions for two_paired_Medians

# 1. Effect Sizes for Paired Samples Medians
def effect_sizes_for_paired_medians(difference_vector, population_difference):
        
        median_sample = np.median(difference_vector)
        standard_deviation = np.std(difference_vector, ddof = 1)

        # Calculation of Effect Sizes for one Sample Median
        
        # 1. Effect Size based on the Inter Quartile Range (Laird and Mosteller, 1990)
        Inter_Quartile_Range = np.quantile(difference_vector, 0.75) - np.quantile(difference_vector, 0.25)
        MDiqr = (median_sample - population_difference) / Inter_Quartile_Range

        # 2. Effect Size based on Median Absolute Deviation (Grissom and Kim) 
        median_absolute_deviation = np.median(abs(difference_vector - median_sample))
        MDmad = (median_sample - population_difference) / median_absolute_deviation
        MDmad_corrected = (median_sample - population_difference) / (median_absolute_deviation*1.4826)

        # 3. Effect Size based on the biweight standard Error Goldberg & Iglewicz, 1992; Lax, 1985)
        MDbw = (median_sample - population_difference) / (biweight_midvariance(difference_vector)**0.5)

        # 4. Effect Size based on the Standard Deviation (Thompson, 2007)
        MDs = (median_sample - population_difference) / standard_deviation

        # 5. Effect Size based on the Standard Deviation - Another Estimator for the dispertion around median Qn (Rousseeuw and Croux, 1993)
        def pairwise_differences(x):      # a function to calcualte all pairwise comparison
            return [b - a for a, b in itertools.combinations(x, 2)]
        Pairwise_Vector = pairwise_differences(difference_vector)
        Qn = 2.2219 * (np.quantile(abs(np.array(Pairwise_Vector)),0.25))
        MDqn = (median_sample - population_difference) / Qn

        # 6. Quantile Shift Effect Size (Wilcox)
        Median_Shift_Effect_Size = np.mean(difference_vector - median_sample + population_difference <= median_sample)

        return MDiqr, MDmad, MDmad_corrected, MDbw, MDs, MDqn, Median_Shift_Effect_Size


class TwoPairedMedians():
    @staticmethod
    def Two_Paired_Medians_From_Data(params: dict) -> dict:
        
        # Set params
        population_median_difference = params["Difference in the Population"]
        Column_1 = params["Column 1"]
        Column_2 = params["Column 2"]
        Number_of_Boot_Samples = ["Number of Bootstrapp Samples"]
        confidence_level_percentage = params["Confidence Level"]
        confidence_level = confidence_level_percentage / 100

        # Calculation

        #####################################################
        # 1 - Descreptive Statistics and Dispersion Measures#
        #####################################################
        
        # 1A. Descreptive Statistics sample 1
        Median_sample_1 = np.median(Column_1)
        Mean_sample_1 = np.mean(Column_1)
        Sample_Size_1 = len(Column_1)
        Standard_Deviation_sample_1 = np.std(Column_1, ddof = 1)
        pairwise_averages_sample_1 = [(Column_1[i] + Column_1[j]) / 2.0 for i in range(len(Column_1)) for j in range(i, len(Column_1))]
        Pseudo_Median_sample_1 = np.median(pairwise_averages_sample_1) # This is the Hodghes lhemann estimator, note that other packages use different algorithm to estimate the median of the population (Minitab uses Johnson and T. Mizoguchi and stats package, Desc tools uses Monhan which is the same as the default stats package) 
        Harrell_Davis_Estimator_sample_1 = hdmedian(Column_1) 

        # 1B.  Dispersion Measures sample 1  
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


        # 1E. Difference Between Samples - Descreptive Statistics and Dispersion Measures
        Difference_Between_Samples = np.array(Column_1 - Column_2)
        
        # Descreptive Statistics Difference
        Median_sample_Difference = np.median(Difference_Between_Samples)
        Mean_sample_Difference = np.mean(Difference_Between_Samples)
        sample_size = len(Difference_Between_Samples)
        Standard_Deviation_sample_Difference = np.std(Difference_Between_Samples, ddof = 1)
        pairwise_averages_sample_Difference = [(Difference_Between_Samples[i] + Difference_Between_Samples[j]) / 2.0 for i in range(len(Difference_Between_Samples)) for j in range(i, len(Difference_Between_Samples))]
        Pseudo_Median_sample_Difference = np.median(pairwise_averages_sample_Difference) # This is the Hodghes lhemann estimator, note that other packages use different algorithm to estimate the median of the population (Minitab uses Johnson and T. Mizoguchi and stats package, Desc tools uses Monhan which is the same as the default stats package) 
        Harrell_Davis_Estimator_sample_Difference = hdmedian(Difference_Between_Samples) 

        # 1F. Dispersion Measures Difference
        absolute_deviations_from_median_sample_Difference = abs(Difference_Between_Samples-Median_sample_Difference)
        mean_absolute_deviation_sample_Difference = np.mean(absolute_deviations_from_median_sample_Difference)
        normal_corrected_mean_absolout_deviation_sample_Difference = mean_absolute_deviation_sample_Difference * 1.2533
        median_absolute_deviation_sample_Difference = np.median(absolute_deviations_from_median_sample_Difference)
        normal_corrected_median_absoloute_deviation_sample_Difference = 1.4826*median_absolute_deviation_sample_Difference
        max_value_Sample_Difference = np.max(Column_2)
        min_value_Sample_Difference = np.min(Column_2)
        Range_Sample_Difference = max_value_Sample_Difference - min_value_Sample_Difference
        Inter_Quartile_Range_sample_Difference = iqr(Column_2)

        Pairwise_Vector_sample_Difference = pairwise_differences(Column_2)
        Qn_sample_Difference = 2.2219 * (np.quantile(abs(np.array(Pairwise_Vector_sample_Difference)),0.25))

        ###############################################
        # 2. - Effect Sizes for Paired Samples medians#
        ###############################################
        MDiqr, MDmad, MDmad_corrected, MDbw, MDs, MDqn, Median_Shift_Effect_Size = effect_sizes_for_paired_medians(Difference_Between_Samples, population_median_difference)

        #############################
        # 3 - Inferntial Statistics #
        #############################

        # 3.1 Paired-Samples t-test for the median using the MAD as the standard error measure (for normal distribution) - see wrappedtools package (Medianse Function)
        Standard_Error_median = (median_absolute_deviation_sample_Difference*1.4826) / np.sqrt(sample_size)
        t_median_mad_based = (Median_sample_Difference -  population_median_difference) / Standard_Error_median
        P_value_median_mad_based = t.sf(t_median_mad_based, sample_size-1)

        # 3.2 Paired-Samples Sign Test
        Sign_Test = desctools.SignTest(ro.FloatVector(Difference_Between_Samples), conf_level = confidence_level, mu = population_median_difference)
        S_Statistic = Sign_Test[0]
        Sign_Test_P_Value_exact = Sign_Test[2]

        Statistic_Binomial_Test, p_val_Binom = sign_test(samp = Column_1, mu0 = population_median_difference) # This is another way to derieve the p-value and its statistics with the binom.test

        # 3.3 Paired-Samples Wilcoxon Test
        Statistic_Wilcoxon_Exact, p_val_wilcoxon_Exact = wilcoxon(Difference_Between_Samples, y= np.repeat(Median_sample_Difference, (sample_size)), method="exact")
        Statistic_Wilcoxon_Approximated, p_val_wilcoxon_Approx = wilcoxon(Difference_Between_Samples, y= np.repeat(population_median_difference, (sample_size)), method="approx", correction=False)
        Statistic_Wilcoxon_Approximated_Corrected, p_val_wilcoxon_Approx_Corrected = wilcoxon(Difference_Between_Samples, y= np.repeat(population_median_difference, (sample_size)), method="approx", correction=True)


        ############################################################
        # 4. Confidence Intervals for the Median of the Difference #
        ############################################################
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)

        # 4A. Price and Bonett
        Difference_Between_Medians = Median_sample_1 - Median_sample_2
        sorted_x = np.sort(Column_1)
        c1 = np.round((sample_size + 1) / 2 - sample_size**0.5)
        X1 = sorted_x[int(sample_size - c1)]
        X2 = sorted_x[int(c1 - 1)]
        Z1 = norm.ppf(1 - binom.cdf(c1 - 1, sample_size, 0.5))
        Variance_Sample_1 = (((X1 - X2) / (2 * Z1))**2)

        Median_Sample_2 = np.median(Column_2)
        sample_size_2 = len(Column_2)
        sorted_y = np.sort(Column_2)
        c2 = np.round((sample_size_2 + 1) / 2 - sample_size**0.5)
        Y1 = (sorted_y[int(sample_size - c2)])
        Y2 = (sorted_y[int(c2 - 1)])
        Z2 = norm.ppf(1 - binom.cdf(c2 - 1, sample_size, 0.5))
        Variance_Sample_2 = ((Y1 - Y2) / (2 * Z2))**2

        values_lower_than_medians = np.sum((Column_1 < Median_sample_1) & (Column_2<Median_Sample_2))
        Median_Probability = (values_lower_than_medians + 0.25) / (sample_size + 1) if sample_size % 2 == 0 else (values_lower_than_medians + 0.25) / sample_size
        Covariance = (4*Median_Probability - 1)*np.sqrt(Variance_Sample_1)*np.sqrt(Variance_Sample_2)
        Standrd_Error_Within_Difference = np.sqrt(Variance_Sample_1 + Variance_Sample_2 - 2*Covariance)

        LowerCi_Price_Bonett_within_difference = (Median_sample_1 - Median_Sample_2) - zcrit * Standrd_Error_Within_Difference
        UpperCi_Price_Bonett_within_difference = (Median_sample_1 - Median_Sample_2) + zcrit * Standrd_Error_Within_Difference

        # 4B. Hodges Lehmann Confidence Intervals
        Hodges_Lehmann = desctools.MedianCI(ro.FloatVector(Difference_Between_Samples), conf_level = confidence_level)
        
        # 4C. Sign Test Confidence Intervals
        Sign_Test_Exact_CI = Sign_Test[3]

        # 4D. Non-Parametric WIlcox test Based CI's
        Nonparametric_Exact_Wilcox_Based_CIs = rigr.wilcoxon(ro.FloatVector(Difference_Between_Samples), exact = True, correct = True,  conf_int = True, null_hypoth = population_median_difference, conf_level = confidence_level)[10][2:3]
        Nonparametric_Wilcox_Based_CIs = rigr.wilcoxon(ro.FloatVector(Difference_Between_Samples), conf_int = True, correct = True, null_hypoth = population_median_difference, conf_level = confidence_level)[10][2:3]
        Nonparametric_Wilcox_Corrected_Based_CIs  = rigr.wilcoxon(ro.FloatVector(Difference_Between_Samples), conf_int = True, correct = False, null_hypoth = population_median_difference, conf_level = confidence_level)[10][2:3]     # See Hollander and Wolfe, 1999 (Page 133-134)
       
        # 4E. Bootstrapping Confidence Intervals
        boot_sample = IndependentSamplesBootstrap(Difference_Between_Samples)
        ci_basic = boot_sample.conf_int(lambda x: np.median(x), 1000, method="basic", size = confidence_level)
        ci_percentile = boot_sample.conf_int(lambda x: np.median(x), 1000, method="percentile", size = confidence_level)
        ci_bc = boot_sample.conf_int(lambda x: np.median(x), 1000, method="bc", size = confidence_level)
        ci_normal = boot_sample.conf_int(lambda x: np.median(x), 1000, method="norm", size = confidence_level)


        ###############################################
        # 5. Confidence Intervals for the Effect Size #
        ###############################################

        def bootstrap_sample(Difference_Between_Samples, num_samples):
            n = len(Difference_Between_Samples)
            bootstrap_samples = []
            for _ in range(num_samples):
                bootstrap_sample = np.random.choice(Difference_Between_Samples, size=n, replace=True)
                bootstrap_samples.append(bootstrap_sample)
            return bootstrap_samples
        bootstrap_samples = bootstrap_sample(Difference_Between_Samples, Number_of_Boot_Samples)


        # Bootstrapping CI
        Boot_Samples = []
        for Difference_Between_Samples in bootstrap_samples:
            result = effect_sizes_for_paired_medians(Difference_Between_Samples, population_median_difference)
            Boot_Samples.append(result)

        # Calculate percentiles for each parameter
        percentiles = []
        for i in range(7):  # There are 7 parameters returned by the function
            parameter_values = [result[i] for result in Boot_Samples]
            parameter_percentiles = np.percentile(parameter_values, [(100 - confidence_level_percentage) - ((100 - confidence_level_percentage) / 2)  , (confidence_level_percentage) + ((100 - confidence_level_percentage) / 2)])
            percentiles.append(parameter_percentiles)

        MDiqr_ci = percentiles[0]
        MDmad_ci = percentiles[1]
        MDmad_corrected_ci = percentiles[2]
        MDbw_ci = percentiles[3]
        MDs_ci = percentiles[4]
        MDqn_ci = percentiles[5]
        Median_Shift_Effect_Size_ci = percentiles[6]

        ###########################################################
        #### 6. Ratio of Medians and Confidence Intervals (Bonett)#
        ###########################################################

        ratio_of_medians =  Median_sample_1 / Median_Sample_2
        log_X1 = np.log(sorted_x[int(sample_size - c1)])
        log_X2 = np.log(sorted_x[int(c1 - 1)])
        log_Y1 = np.log(sorted_y[int(sample_size_2 - c2)])
        log_Y2 = np.log(sorted_y[int(c2 - 1)])
        log_Z1 = norm.ppf(1 - binom.cdf(c1 - 1, sample_size, 0.5))
        log_Variance_Sample_1 = (((log_X1 - log_X2) / (2 * log_Z1))**2)
        log_Z2 = norm.ppf(1 - binom.cdf(c2 - 1, sample_size_2, 0.5))
        log_Variance_Sample_2 = ((log_Y1 - log_Y2) / (2 * log_Z2))**2
        Covariance_log = (4*Median_Probability - 1)*np.sqrt(log_Variance_Sample_1)*np.sqrt(log_Variance_Sample_2)
        Log_Standard_Error_Within_Ratio = np.sqrt(log_Variance_Sample_1 + log_Variance_Sample_2 - 2*Covariance_log)    
        LowerCi_Price_Bonett_within_ratio = ratio_of_medians * math.exp(-zcrit * Log_Standard_Error_Within_Ratio)
        UpperCi_Price_Bonett_within_ratio = ratio_of_medians * math.exp(zcrit * Log_Standard_Error_Within_Ratio)

        results = {}

        # Descriptive Statistics for Sample 1

        results["                                                                                                                             "] = ""
        results["Sample 1 - Median"] = round(Median_sample_1, 4)
        results["Sample 1 - Mean"] = round(Mean_sample_1, 4)
        results["Sample 1 - Standard Deviation"] = round(Standard_Deviation_sample_1, 4)
        results["Sample 1 - Range"] = (f"R = {Range_Sample_1} ({min_value_Sample_1},{max_value_Sample_1})")
        results["Sample 1 - Inter Quartile Range"] = np.around(Inter_Quartile_Range_sample_1, 4)
        results["Sample 1 - Mean Absolute Deviation"] = round(mean_absolute_deviation_sample_1, 4)
        results["Sample 1 - Median Absolute Deviation"] = round(median_absolute_deviation_sample_1, 4)
        results["Sample 1 - Mean Absolute Deviation (Corrected)"] = round(normal_corrected_mean_absolout_deviation_sample_1, 4)
        results["Sample 1 - Median Absolute Deviation (Corrected)"] = round(normal_corrected_median_absoloute_deviation_sample_1, 4)
        results["Sample 1 - Qn"] = round(Qn_sample_1, 4)
        results["                                                                                                                              "] = ""

        # Descriptive Statistics for Sample 2
        results["Sample 2 - Median"] = round(Median_sample_2, 4)
        results["Sample 2 - Mean"] = round(Mean_sample_2, 4)
        results["Sample 2 - Standard Deviation"] = round(Standard_Deviation_sample_2, 4)
        results["Sample 2 - Range"] = (f"R = {Range_Sample_2} ({min_value_Sample_2},{max_value_Sample_2})")
        results["Sample 2 - Inter Quartile Range"] = np.around(Inter_Quartile_Range_sample_2, 4)
        results["Sample 2 - Mean Absolute Deviation"] = round(mean_absolute_deviation_sample_2, 4)
        results["Sample 2 - Median Absolute Deviation"] = round(median_absolute_deviation_sample_2, 4)
        results["Sample 2 - Mean Absolute Deviation (Corrected)"] = round(normal_corrected_mean_absolout_deviation_sample_2, 4)
        results["Sample 2 - Median Absolute Deviation (Corrected)"] = round(normal_corrected_median_absoloute_deviation_sample_2, 4)
        results["Sample 2 - Qn"] = round(Qn_sample_2, 4)

        results["                                                                                                                                "] = ""

        # Descriptive Statistics of the Difference
        results["Samples Difference - Median"] = round(Median_sample_Difference, 4)
        results["Samples Difference - Mean"] = round(Mean_sample_Difference, 4)
        results["Samples Difference - Standard Deviation"] = round(Standard_Deviation_sample_Difference, 4)
        results["Samples Difference - Range"] = (f"R = {Range_Sample_Difference} ({min_value_Sample_Difference},{max_value_Sample_Difference})")
        results["Samples Difference - Inter Quartile Range"] = np.around(Inter_Quartile_Range_sample_Difference, 4)
        results["Samples Difference - Mean Absolute Deviation"] = round(mean_absolute_deviation_sample_Difference, 4)
        results["Samples Difference - Median Absolute Deviation"] = round(median_absolute_deviation_sample_Difference, 4)
        results["Samples Difference - Mean Absolute Deviation (Corrected)"] = round(normal_corrected_mean_absolout_deviation_sample_Difference, 4)
        results["Samples Difference - Median Absolute Deviation (Corrected)"] = round(normal_corrected_median_absoloute_deviation_sample_Difference, 4)
        results["Samples Difference - Qn"] = round(Qn_sample_Difference, 4)
        results["Difference Between Medians"] = round(Difference_Between_Medians, 4)        

        results["                                                                                                                               "] = ""

        # Inferntial Statistics
        results["Statistic Wilcoxon Exact"] = np.around((Statistic_Wilcoxon_Exact), 4)
        results["Statistic Wilcoxon Approximated"] = np.around((Statistic_Wilcoxon_Approximated), 4)
        results["Statistic Wilcoxon Approximated Corrected"] = np.around((Statistic_Wilcoxon_Approximated_Corrected), 4)
        results["Statistic Exact Sign Test"] = np.around((S_Statistic), 4)
        results["Statistic t-MAD based"] = np.around((t_median_mad_based), 4) # Note that this method does not have an official reference (not one we could find at least)
        results["p-value Sign Test"] = np.around((Sign_Test_P_Value_exact[0]), 4)
        results["p-value Wilcoxon Exact"] = ((p_val_wilcoxon_Exact))
        results["p-value Wilcoxon Approximated"] = ((p_val_wilcoxon_Approx))
        results["p-value Wilcoxon Approximated Corrected"] = ((p_val_wilcoxon_Approx_Corrected))
        results["p-value Median MAD Based"] = np.around((P_value_median_mad_based), 4)

        results["                                                                                                                                 "] = ""

        # Effect Sizes
        results["Effect Size ΔIQR "] = round(MDiqr, 4)
        results["Effect Size ΔMAD "] = round(MDmad, 4)
        results["Effect Size ΔMAD Corrected "] = round(MDmad_corrected, 4)
        results["Effect Size ΔBW "] = round(MDbw, 4)
        results["Effect Size Δs "] = round(MDs, 4)
        results["Effect Size ΔQN "] = round(MDqn, 4)
        results["Median Shift Effect Size "] = round(Median_Shift_Effect_Size, 4)
       
        results["                                                                                                                                  "] = ""

        # Mean Difference and Confidence Intervals
        results["Price-Bonett CI's"] = np.array([LowerCi_Price_Bonett_within_difference, UpperCi_Price_Bonett_within_difference])
        results["Median Confidence Intervals Basic Bootstrapping"] = ci_basic[0,0], ci_basic[1,0]
        results["Median Confidence Intervals Percentile Bootstrapping"] = ci_percentile[0,0], ci_percentile[1,0]
        results["Median Confidence Intervals Bias Corrected Bootstrapping"] = ci_bc[0,0], ci_bc[1,0]
        results["Median Confidence Intervals Normal Bootstrapping"] = np.around(ci_normal[0,0],4), np.around(ci_normal[1,0],4)
        results["Median Confidence Interval Hodges-Lehmann Estimator"] =   Hodges_Lehmann[1], Hodges_Lehmann[2]
        results["Median Confidence Intervals Sign Test Exact CI"] =  Sign_Test_Exact_CI[0],Sign_Test_Exact_CI[1] 
        results["Median Confidence Intervals Nonparametric Based on Exact Wilcox Test"] = Nonparametric_Exact_Wilcox_Based_CIs[0]
        results["Median Confidence Intervals Nonparametric Based on Non-Parametric Approximated Wilcox Test"] =  Nonparametric_Wilcox_Based_CIs[0]
        results["Median Confidence Intervals Nonparametric Based on Non-Parametric Approximated Corrected Wilcox Test "] =  Nonparametric_Wilcox_Corrected_Based_CIs[0]

        results["                                                                                                                                    "] = ""

        # Inferntial Statistics
        results["Wilcoxon Statistic Exact"] = np.around((Statistic_Wilcoxon_Exact), 4)
        results["Wilcoxon Statistic Approximated"] = np.around((Statistic_Wilcoxon_Approximated), 4)
        results["Wilcoxon Statistic Approximated Corrected"] = np.around((Statistic_Wilcoxon_Approximated_Corrected), 4)
        results["Wilcoxon Corrected Statistic"] = np.around((P_value_median_mad_based), 4)
        results["Statistic Sign Test"] = np.around((S_Statistic), 4)
        results["t Statistic MAD based"] = np.around((t_median_mad_based), 4)

        results["p-value Median Mad Based"] = np.around((P_value_median_mad_based), 4)
        results["p-value Sign Test (Exact)"] = np.around((Sign_Test_P_Value_exact), 4)
        results["p-value Wilcoxon Exact (Python)"] = ((p_val_wilcoxon_Exact))
        results["p-value Wilcoxon Approximated (Python)"] = ((p_val_wilcoxon_Approx))
        results["p-value Wilcoxon Approximated Corrected (Python)"] = ((p_val_wilcoxon_Approx_Corrected))
        
        # Confidence Intervals for the Effect Size
        results["Confidence Interval ΔIQR"] = np.around((MDiqr_ci), 4)
        results["Confidence Interval ΔMAD"] = np.around((MDmad_ci), 4)
        results["Confidence Interval ΔMAD Corrected"] = np.around((MDmad_corrected_ci), 4)
        results["Confidence Interval ΔBW"] = np.around((MDbw_ci), 4)
        results["Confidence Interval Δs"] = np.around((MDs_ci), 4)
        results["Confidence Interval ΔQN"] = np.around((MDqn_ci), 4)
        results["Confidence Interval Median Shift Effect Size"] = np.around((Median_Shift_Effect_Size_ci), 4)

        formatted_p_value = "{:.3f}".format(float(Sign_Test_P_Value_exact[0])).lstrip('0') if float(Sign_Test_P_Value_exact[0]) >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line ΔIQR"] = " \033[3mS\033[0m = {}, {}{}, ΔIQR = {:.3f}, {}% CI(Bootstrapping) [{:.3f},{:.3f}]".format(int(S_Statistic[0]), '\033[3mp = \033[0m' if Sign_Test_P_Value_exact[0] >= 0.001 else '', formatted_p_value, MDiqr ,confidence_level_percentage, round(MDiqr_ci[0],3), round(MDiqr_ci[1],3))
        results["Statistical Line ΔMAD"] = " \033[3mS\033[0m = {}, {}{}, ΔMAD = {:.3f}, {}% CI(Bootstrapping) [{:.3f},{:.3f}]".format(int(S_Statistic[0]), '\033[3mp = \033[0m' if Sign_Test_P_Value_exact[0] >= 0.001 else '', formatted_p_value, MDmad ,confidence_level_percentage, round(MDmad_ci[0],3), round(MDmad_ci[1],3))
        results["Statistical Line ΔMAD Corrected"] = " \033[3mS\033[0m = {}, {}{}, ΔMAD Corrected = {:.3f}, {}% CI(Bootstrapping) [{:.3f},{:.3f}]".format(int(S_Statistic[0]), '\033[3mp = \033[0m' if Sign_Test_P_Value_exact[0] >= 0.001 else '', formatted_p_value, MDmad_corrected ,confidence_level_percentage, round(MDmad_corrected_ci[0],3), round(MDmad_corrected_ci[1],3))        
        results["Statistical Line Δs"] = " \033[3mS\033[0m = {}, {}{}, Δs = {:.3f}, {}% CI(Bootstrapping) [{:.3f},{:.3f}]".format(int(S_Statistic[0]), '\033[3mp = \033[0m' if Sign_Test_P_Value_exact[0] >= 0.001 else '', formatted_p_value, MDs ,confidence_level_percentage, round(MDs_ci[0],3), round(MDs_ci[1],3))
        results["Statistical Line ΔQN"] = " \033[3mS\033[0m = {}, {}{}, ΔQn = {:.3f}, {}% CI(Bootstrapping) [{:.3f},{:.3f}]".format(int(S_Statistic[0]), '\033[3mp = \033[0m' if Sign_Test_P_Value_exact[0] >= 0.001 else '', formatted_p_value, MDqn ,confidence_level_percentage, round(MDqn_ci[0],3), round(MDqn_ci[1],3))
        results["Statistical Line Δbw"] = " \033[3mS\033[0m = {}, {}{}, ΔBW = {:.3f}, {}% CI(Bootstrapping) [{:.3f},{:.3f}]".format(int(S_Statistic[0]), '\033[3mp = \033[0m' if Sign_Test_P_Value_exact[0] >= 0.001 else '', formatted_p_value, MDbw ,confidence_level_percentage, round(MDbw_ci[0],3), round(MDbw_ci[1],3))
        results["Statistical Line Median Shift Effect Size "] = " \033[3mS\033[0m = {}, {}{}, Median Shift Effect Size = {:.3f}, {}% CI(Bootstrapping) [{:.3f},{:.3f}]".format(int(S_Statistic[0]), '\033[3mp = \033[0m' if Sign_Test_P_Value_exact[0] >= 0.001 else '', formatted_p_value, Median_Shift_Effect_Size ,confidence_level_percentage, round(Median_Shift_Effect_Size_ci[0],3), round(Median_Shift_Effect_Size_ci[1],3))

        # Ratio of Medians and Confidecne Intervals 
        results["Ratio of Medians"] = round(ratio_of_medians, 4)
        results["Standard Error Ratio of Medians "] = round(math.exp(Log_Standard_Error_Within_Ratio), 4)
        results["Lower CI Ratio of Medians"] = round(LowerCi_Price_Bonett_within_ratio, 4)
        results["Upper CI Ratio of Medians"] = round(UpperCi_Price_Bonett_within_ratio, 4)

        return results


# Things to consider
# 1. Add robust measures and wilcoxon table
# 2. Note that Difference between Medians can be different than the median of the Differecne 
