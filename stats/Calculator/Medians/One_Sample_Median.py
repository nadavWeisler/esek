###############################################
##### Effect Size for One Sample Median #######
###############################################

import numpy as np
import itertools
from scipy.stats import t, wilcoxon, iqr
from astropy.stats import biweight_midvariance
from arch.bootstrap import IndependentSamplesBootstrap
from scipy.stats.mstats import hdmedian
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import warnings
from statsmodels.stats.descriptivestats import sign_test


warnings.filterwarnings("ignore", category=RuntimeWarning)

# R imports
desctools = importr("DescTools")
rigr = importr("rigr")

def effect_sizes_for_one_sample_median(Column_1, population_median):
        
        median_sample = np.median(Column_1)
        standard_deviation = np.std(Column_1, ddof = 1)

        # Calculation of Effect Sizes for one Sample Median
        
        # 1. Effect Size based on the Inter Quartile Range (Laird and Mosteller, 1990)
        Inter_Quartile_Range = np.quantile(Column_1, 0.75) - np.quantile(Column_1, 0.25)
        MDiqr = (median_sample - population_median) / Inter_Quartile_Range

        # 2. Effect Size based on Median Absolute Deviation (Grissom and Kim) 
        median_absolute_deviation = np.median(abs(Column_1 - median_sample))
        MDmad = (median_sample - population_median) / median_absolute_deviation
        MDmad_corrected = (median_sample - population_median) / (median_absolute_deviation*1.4826)

        # 3. Effect Size based on the biweight standard Error Goldberg & Iglewicz, 1992; Lax, 1985)
        MDbw = (median_sample - population_median) / (biweight_midvariance(Column_1)**0.5)

        # 4. Effect Size based on the Standard Deviation (Thompson, 2007)
        MDs = (median_sample - population_median) / standard_deviation

        # 5. Effect Size based on the Standard Deviation - Another Estimator for the dispertion around median Qn (Rousseeuw and Croux, 1993)
        def pairwise_differences(x):      # a function to calcualte all pairwise comparison
            return [b - a for a, b in itertools.combinations(x, 2)]
        Pairwise_Vector = pairwise_differences(Column_1)
        Qn = 2.2219 * (np.quantile(abs(np.array(Pairwise_Vector)),0.25))
        MDqn = (median_sample - population_median) / Qn

        # 6. Quantile Shift Effect Size (Wilcox)
        Median_Shift_Effect_Size = np.mean(Column_1 - median_sample + population_median <= median_sample)

        return MDiqr, MDmad, MDmad_corrected, MDbw, MDs, MDqn, Median_Shift_Effect_Size


class One_Sample_median():
    @staticmethod
    def one_sample_median(params: dict) -> dict:
        
        # Set params
        population_median = params["Population Median"]
        Column_1 = params["Column 1"]
        confidence_level_percentage = params["Confidence Level"]
        Number_of_Boot_Samples = params["Number of Bootstrapping Samples"]

        ######################################################
        #1. Descreptive Statistics and Measures of Dispersion#
        ######################################################

        # 1A. Preperation and Descreptive Statistics
        data = Column_1
        confidence_level = confidence_level_percentage / 100
        standard_deviation = np.std(Column_1, ddof = 1)
        sample_size = len(Column_1)
        median_sample = np.median(Column_1)
        median_sample = np.median(Column_1)
        Mean_Sample = np.mean(Column_1)
        
        # 1B. Measures of Dispersion
        max_value = np.max(Column_1)
        min_value = np.min(Column_1)
        Range = max_value - min_value
        mean_absolut_deviation = np.mean(abs(Column_1 - median_sample))
        standard_deviation = np.std(Column_1, ddof = 1)
        median_absolute_deviation = np.median(abs(Column_1 - median_sample))
        pairwise_averages = [(Column_1[i] + Column_1[j]) / 2.0 for i in range(len(Column_1)) for j in range(i, len(Column_1))]
        Pseudo_Median = np.median(pairwise_averages) # This is the Hodghes lhemann estimator, note that other packages use different algorithm to estimate the median of the population (Minitab uses Johnson and T. Mizoguchi and stats package, Desc tools uses Monhan which is the same as the default stats package) 
        Harrell_Davis_Estimator = hdmedian(Column_1) 
        inter_quartile_range = iqr(Column_1)

        # Another Estimator for the dispertion around median Qn (Rousseeuw and Croux, 1993)

        def pairwise_differences(x):      # a function to calcualte all pairwise comparisons
            return [b - a for a, b in itertools.combinations(x, 2)]
        Pairwise_Vector = pairwise_differences(Column_1)
        Qn = 2.2219 * (np.quantile(abs(np.array(Pairwise_Vector)),0.25))

        ####################################################
        # 2. Effect Sizes Calculation for One Sample Median
        ####################################################
        MDiqr, MDmad, MDmad_corrected, MDbw, MDs, MDqn, Median_Shift_Effect_Size = effect_sizes_for_one_sample_median(Column_1, population_median)


        ################################################
        # 3. Inferntial Statistics for one sample median
        ################################################

        # 3.1 Simple t-test for the median using the MAD as the standard error measure (for normal distribution) - see wrappedtools package (Medianse Function)
        Standard_Error_median = (median_absolute_deviation*1.4826) / np.sqrt(sample_size)
        t_median_mad_based = (median_sample -  population_median) / Standard_Error_median
        P_value_median_mad_based = t.sf(t_median_mad_based, sample_size-1)

        # 3.2 One-Sample Sign Test
        Sign_Test = desctools.SignTest(ro.FloatVector(data), conf_level = confidence_level, mu = population_median)
        S_Statistic = Sign_Test[0]
        Sign_Test_P_Value_exact = Sign_Test[2]

        Statistic_Binomial_Test, p_val_Binom = sign_test(samp = Column_1, mu0 = population_median) # This is another way to derieve the p-value and its statistics with the binom.test

        # 3.3 One sample Wilcoxon Test
        Statistic_Wilcoxon_Exact, p_val_wilcoxon_Exact = wilcoxon(Column_1, y= np.repeat(population_median, (sample_size)), method="exact")
        Statistic_Wilcoxon_Approximated, p_val_wilcoxon_Approx = wilcoxon(Column_1, y= np.repeat(population_median, (sample_size)), method="approx", correction=False)
        Statistic_Wilcoxon_Approximated_Corrected, p_val_wilcoxon_Approx_Corrected = wilcoxon(Column_1, y= np.repeat(population_median, (sample_size)), method="approx", correction=True)

        ################################################
        # 4. Confidence Intervals for one sample median
        ################################################
        
        # 4A . Bootstrapping Confidence Intervals
        boot_sample = IndependentSamplesBootstrap(Column_1)
        ci_basic = boot_sample.conf_int(lambda x: np.median(x), Number_of_Boot_Samples, method="basic", size = confidence_level)
        ci_percentile = boot_sample.conf_int(lambda x: np.median(x), Number_of_Boot_Samples, method="percentile", size = confidence_level)
        ci_bc = boot_sample.conf_int(lambda x: np.median(x), Number_of_Boot_Samples, method="bc", size = confidence_level)
        ci_normal = boot_sample.conf_int(lambda x: np.median(x), Number_of_Boot_Samples, method="norm", size = confidence_level)

        # 4B. Hodges Lhemann Confidence Intervals
        Hodges_Lehmann = desctools.MedianCI(ro.FloatVector(data), conf_level = confidence_level)
        
        # 4C. Sign Test Confidence Intervals
        Sign_Test_Exact_CI = Sign_Test[3]

        # 4D. Non-Parametric WIlcox test Based CI's
        Nonparametric_Exact_Wilcox_Based_CIs = rigr.wilcoxon(ro.FloatVector(data), exact = True, correct = True,  conf_int = True, null_hypoth = population_median, conf_level = confidence_level)[10][2:3]
        Nonparametric_Wilcox_Based_CIs = rigr.wilcoxon(ro.FloatVector(data), conf_int = True, correct = True, null_hypoth = population_median, conf_level = confidence_level)[10][2:3]
        Nonparametric_Wilcox_Corrected_Based_CIs  = rigr.wilcoxon(ro.FloatVector(data), conf_int = True, correct = False, null_hypoth = population_median, conf_level = confidence_level)[10][2:3]     # See Hollander and Wolfe, 1999 (Page 133-134)

        # 4E. MAD based Confidence Intervals (see Cousineau)
        t_critical_value = t.ppf(confidence_level + ((1 - confidence_level) / 2), sample_size-1)
        LowerCi_MAD = median_sample - Standard_Error_median * t_critical_value
        UpperCi_MAD = median_sample + Standard_Error_median * t_critical_value


        ####################################################################################
        # 5. Confidence Intervals for the effect size using Different Bootstrapping Methods#
        ####################################################################################

        # Bootstrap function
        def bootstrap_sample(Column_1, num_samples):
            n = len(Column_1)
            bootstrap_samples = []
            for _ in range(num_samples):
                bootstrap_sample = np.random.choice(Column_1, size=n, replace=True)
                bootstrap_samples.append(bootstrap_sample)
            return bootstrap_samples
        bootstrap_samples = bootstrap_sample(Column_1, Number_of_Boot_Samples)


        # Bootstrapping CI
        Boot_Samples = []
        for Column_1 in bootstrap_samples:
            result = effect_sizes_for_one_sample_median(Column_1, population_median)
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


        # Set results
        results = {}

        results["Sample Size"] = round(sample_size, 4)
        results["Sample's Median"] = round(median_sample, 4)
        results["Sample's Mean"] = round(Mean_Sample, 4)
        results["Sample's Standard Deviation"] = round(standard_deviation, 4)
        results["Pseudo Median (Hodghes Lhemann)"] = Pseudo_Median
        results["Hodges-Lehmann Estimator (Monahan Algorithm)"] = Hodges_Lehmann[0]
        results["Harrell-Davis Estimator"] = Harrell_Davis_Estimator

        # Dispersion Measures
        results["Range"] = (f"R = {Range} = ({min_value}, {max_value})")
        results["Inter Quartile Range"] = (inter_quartile_range)
        results["Mean Absolute Deviation"] = round(mean_absolut_deviation, 4)
        results["Median Absolute Deviation"] = median_absolute_deviation
        results["Mean Absolute Deviation Corrected"] = median_absolute_deviation * 1.2533
        results["Median Absolute Deviation Corrected"] = median_absolute_deviation * 1.4826
        results["Qn (Rousseeuw and Croux)"] = Qn

        # Confidence Intervals for the Median
        results["Median Confidence Intervals Basic Bootstrapping"] = ci_basic[0,0], ci_basic[1,0]
        results["Median Confidence Intervals Percentile Bootstrapping"] = ci_percentile[0,0], ci_percentile[1,0]
        results["Median Confidence Intervals Bias Corrected Bootstrapping"] = ci_bc[0,0], ci_bc[1,0]
        results["Median Confidence Intervals Normal Bootstrapping"] = np.around(ci_normal[0,0],4), np.around(ci_normal[1,0],4)
        results["Median Confidence Interval Hodges-Lehmann Estimator"] =   Hodges_Lehmann[1], Hodges_Lehmann[2]
        results["Median Confidence Intervals Sign Test Exact CI"] =  Sign_Test_Exact_CI[0],Sign_Test_Exact_CI[1] 
        results["Median Confidence Intervals Nonparametric Based on Exact Wilcox Test"] = Nonparametric_Exact_Wilcox_Based_CIs[0]
        results["Median Confidence Intervals Nonparametric Based on Non-Parametric Approximated Wilcox Test"] =  Nonparametric_Wilcox_Based_CIs[0]
        results["Median Confidence Intervals Nonparametric Based on Non-Parametric Approximated Corrected Wilcox Test "] =  Nonparametric_Wilcox_Corrected_Based_CIs[0]
        results["Median Confidence Intervals MAD based "] =  round(LowerCi_MAD,4), round(UpperCi_MAD,4)

        # Effect Sizes
        results["Effect Size ΔIQR "] = round(MDiqr, 4)
        results["Effect Size ΔMAD "] = round(MDmad, 4)
        results["Effect Size ΔMAD Corrected "] = round(MDmad_corrected, 4)
        results["Effect Size ΔBW "] = round(MDbw, 4)
        results["Effect Size Δs "] = round(MDs, 4)
        results["Effect Size ΔQN "] = round(MDqn, 4)
        results["Median Shift Effect Size "] = round(Median_Shift_Effect_Size, 4)

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

        # Confidence Intervals for the Effect Size
        results["Confidence Interval ΔIQR"] = np.around((MDiqr_ci), 4)
        results["Confidence Interval ΔMAD"] = np.around((MDmad_ci), 4)
        results["Confidence Interval ΔMAD Corrected"] = np.around((MDmad_corrected_ci), 4)
        results["Confidence Interval ΔBW"] = np.around((MDbw_ci), 4)
        results["Confidence Interval Δs"] = np.around((MDs_ci), 4)
        results["Confidence Interval ΔQN"] = np.around((MDqn_ci), 4)
        results["Confidence Interval Median Shift Effect Size"] = np.around((Median_Shift_Effect_Size_ci), 4)

        # Statistical Lines
        formatted_p_value = "{:.3f}".format(float(Sign_Test_P_Value_exact[0])).lstrip('0') if float(Sign_Test_P_Value_exact[0]) >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line ΔIQR"] = " \033[3mS\033[0m = {}, {}{}, ΔIQR = {:.3f}, {}% CI(Bootstrapping) [{:.3f},{:.3f}]".format(int(S_Statistic[0]), '\033[3mp = \033[0m' if Sign_Test_P_Value_exact[0] >= 0.001 else '', formatted_p_value, MDiqr ,confidence_level_percentage, round(MDiqr_ci[0],3), round(MDiqr_ci[1],3))
        results["Statistical Line ΔMAD"] = " \033[3mS\033[0m = {}, {}{}, ΔMAD = {:.3f}, {}% CI(Bootstrapping) [{:.3f},{:.3f}]".format(int(S_Statistic[0]), '\033[3mp = \033[0m' if Sign_Test_P_Value_exact[0] >= 0.001 else '', formatted_p_value, MDmad ,confidence_level_percentage, round(MDmad_ci[0],3), round(MDmad_ci[1],3))
        results["Statistical Line ΔMAD Corrected"] = " \033[3mS\033[0m = {}, {}{}, ΔMAD Corrected = {:.3f}, {}% CI(Bootstrapping) [{:.3f},{:.3f}]".format(int(S_Statistic[0]), '\033[3mp = \033[0m' if Sign_Test_P_Value_exact[0] >= 0.001 else '', formatted_p_value, MDmad_corrected ,confidence_level_percentage, round(MDmad_corrected_ci[0],3), round(MDmad_corrected_ci[1],3))        
        results["Statistical Line Δs"] = " \033[3mS\033[0m = {}, {}{}, Δs = {:.3f}, {}% CI(Bootstrapping) [{:.3f},{:.3f}]".format(int(S_Statistic[0]), '\033[3mp = \033[0m' if Sign_Test_P_Value_exact[0] >= 0.001 else '', formatted_p_value, MDs ,confidence_level_percentage, round(MDs_ci[0],3), round(MDs_ci[1],3))
        results["Statistical Line ΔQN"] = " \033[3mS\033[0m = {}, {}{}, ΔQn = {:.3f}, {}% CI(Bootstrapping) [{:.3f},{:.3f}]".format(int(S_Statistic[0]), '\033[3mp = \033[0m' if Sign_Test_P_Value_exact[0] >= 0.001 else '', formatted_p_value, MDqn ,confidence_level_percentage, round(MDqn_ci[0],3), round(MDqn_ci[1],3))
        results["Statistical Line Δbw"] = " \033[3mS\033[0m = {}, {}{}, ΔBW = {:.3f}, {}% CI(Bootstrapping) [{:.3f},{:.3f}]".format(int(S_Statistic[0]), '\033[3mp = \033[0m' if Sign_Test_P_Value_exact[0] >= 0.001 else '', formatted_p_value, MDbw ,confidence_level_percentage, round(MDbw_ci[0],3), round(MDbw_ci[1],3))
        results["Statistical Line Median Shift Effect Size "] = " \033[3mS\033[0m = {}, {}{}, Median Shift Effect Size = {:.3f}, {}% CI(Bootstrapping) [{:.3f},{:.3f}]".format(int(S_Statistic[0]), '\033[3mp = \033[0m' if Sign_Test_P_Value_exact[0] >= 0.001 else '', formatted_p_value, Median_Shift_Effect_Size ,confidence_level_percentage, round(Median_Shift_Effect_Size_ci[0],3), round(Median_Shift_Effect_Size_ci[1],3))


        return results
    

    # Things to Consider
    # 1. Solve the division by zero problem in the bootstrapp (use the arch for the bootstrapp effect size that overcome this problem)
    # 2. Check out all the conditions which do not allow the calcaulation of certain confidence intervals (for example too many ties, small sample size etc)
    # 3. Pay ATTENTION THT THE CI FOR SOME FUNCTION IS CHAGING ATOMATICALLY IN THE NOTIFICATION, MAKE SURE YOU LET THE USER KNOW...
      