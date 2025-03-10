
####################################
# CI for Measures of Associations ##
####################################

import numpy as np
from scipy.stats import norm, ncx2, t, ncf
import math

# Relevant Functions for Nominal by Nominal Correlation
#######################################################

# 1. Non Central (Pivotal) CI
def ncp_ci(chival, df, conf):
    def low_ci(chival, df, conf):
            bounds = [0.001, chival / 2, chival]
            ulim = 1 - (1 - conf) / 2
            while ncx2.cdf(chival, df, bounds[0]) < ulim:
                return [0, ncx2.cdf(chival, df, bounds[0])]
            while (diff := abs(ncx2.cdf(chival, df, bounds[1]) - ulim)) > 0.00001:
                bounds = [bounds[0], (bounds[0] + bounds[1]) / 2, bounds[1]] if ncx2.cdf(chival, df, bounds[1]) < ulim else [bounds[1], (bounds[1] + bounds[2]) / 2, bounds[2]]
            return [bounds[1]]
    def high_ci(chival, df, conf):
        # This first part finds upper and lower starting values.
        uc = [chival, 2 * chival, 3 * chival]
        llim = (1 - conf) / 2
        while ncx2.cdf(chival, df, uc[0]) < llim: uc = [uc[0] / 4, uc[0], uc[2]]
        while ncx2.cdf(chival, df, uc[2]) > llim: uc = [uc[0], uc[2], uc[2] + chival]

        diff = 1
        while diff > 0.00001:
            uc = [uc[0], (uc[0] + uc[1]) / 2, uc[1]] if ncx2.cdf(chival, df, uc[1]) < llim else [uc[1], (uc[1] + uc[2]) / 2, uc[2]]
            diff = abs(ncx2.cdf(chival, df, uc[1]) - llim)
            lcdf = ncx2.cdf(chival, df, uc[1])   
        return uc[1]
    low_ncp = low_ci(chival, df, conf)
    high_ncp = high_ci(chival, df, conf)

    return low_ncp[0], high_ncp

# Non Central F CI Function
def NonCentralCiF(f_statistic, df1, df2, confidence_level):
    upper_limit = 1 - (1 - confidence_level) / 2
    lower_limit = 1 - upper_limit
    lower_ci_difference_value = 1

    def LowerCi(f_statistic, df1, df2, upper_limit, lower_ci_difference_value):
        lower_bound = [0.001, f_statistic / 2, f_statistic]       
        while ncf.cdf(f_statistic, df1, df2, lower_bound[0]) < upper_limit:
            return [0, ncf.cdf(f_statistic, df1, df2)] if ncf.cdf(f_statistic, df1, df2) < upper_limit else None
            lower_bound = [lower_bound[0] / 4, lower_bound[0], lower_bound[2]]   
        while ncf.cdf(f_statistic, df1, df2, lower_bound[2]) > upper_limit: lower_bound = [lower_bound[0], lower_bound[2], lower_bound[2] + f_statistic]     
        while lower_ci_difference_value > 0.0000001:
            lower_bound = [lower_bound[0], (lower_bound[0] + lower_bound[1]) / 2, lower_bound[1]] if ncf.cdf(f_statistic, df1, df2, lower_bound[1]) < upper_limit else [lower_bound[1], (lower_bound[1] + lower_bound[2]) / 2, lower_bound[2]]  
            lower_ci_difference_value = abs(ncf.cdf(f_statistic, df1, df2, lower_bound[1]) - upper_limit)        
        return [lower_bound[1]]
    
    def UpperCi(f_statistic, df1, df2, lower_limit, lower_ci_difference_value):
        upper_bound = [f_statistic, 2 * f_statistic, 3 * f_statistic]
        while ncf.cdf(f_statistic, df1, df2, upper_bound[0]) < lower_limit:upper_bound = [upper_bound[0] / 4, upper_bound[0], upper_bound[2]]
        while ncf.cdf(f_statistic, df1, df2, upper_bound[2]) > lower_limit: upper_bound = [upper_bound[0], upper_bound[2], upper_bound[2] + f_statistic]
        while lower_ci_difference_value > 0.00001: upper_bound = [upper_bound[0], (upper_bound[0] + upper_bound[1]) / 2, upper_bound[1]] if ncf.cdf(f_statistic, df1, df2, upper_bound[1]) < lower_limit else [upper_bound[1], (upper_bound[1] + upper_bound[2]) / 2, upper_bound[2]]; lower_ci_difference_value = abs(ncf.cdf(f_statistic, df1, df2, upper_bound[1]) - lower_limit)
        return [upper_bound[1]]
    
    # Calculate lower and upper bounds
    Lower_NCF_CI_Final = LowerCi(f_statistic, df1, df2, upper_limit, lower_ci_difference_value)[0]
    Upper_NCF_CI_Final = UpperCi(f_statistic, df1, df2, lower_limit, lower_ci_difference_value)[0]

    return Lower_NCF_CI_Final, Upper_NCF_CI_Final

class CI_Constructor_Association():


# 1. Confidence Intervals for Cramer V

    @staticmethod
    def Cramer_V_CI_From_Effect_Size(params: dict) -> dict:
        
        Cramer_V = params["Cramer V"]
        Sample_Size = params["Sample Size"]
        Degrees_of_Freedom = params["Degrees Of Freedom"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100

        Chi_Score = Cramer_V**2 * Sample_Size * Degrees_of_Freedom
        lower_ncp, upper_ncp = ncp_ci(Chi_Score, Degrees_of_Freedom, confidence_level)
        LowerCi_Cramer = np.sqrt(lower_ncp / Sample_Size / Degrees_of_Freedom)
        UpperCi_Cramer = np.sqrt(upper_ncp / Sample_Size / Degrees_of_Freedom)

        results = {}
        results["Confidence Intervals Cramer's V"] = np.array([round(LowerCi_Cramer, 4), round(UpperCi_Cramer, 4)])

        return results


# 2. Confidence Intervals for Cramer V from Chi Score

    @staticmethod
    def Cramer_V_CI_From_Chi_Score(params: dict) -> dict:
        
        Chi_Score = params["Chi Square Score"]
        Sample_Size = params["Sample Size"]
        Degrees_of_Freedom = params["Degrees Of Freedom"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100

        Cramer_V = np.sqrt(Chi_Score / (Sample_Size*Degrees_of_Freedom))
        lower_ncp, upper_ncp = ncp_ci(Chi_Score, Degrees_of_Freedom, confidence_level)
        LowerCi_Cramer = np.sqrt(lower_ncp / Sample_Size / Degrees_of_Freedom)
        UpperCi_Cramer = np.sqrt(upper_ncp / Sample_Size / Degrees_of_Freedom)

        results = {}
        results["Cramer's V"] = Cramer_V
        results["Confidence Intervals Cramer's V"] = np.array([round(LowerCi_Cramer, 4), round(UpperCi_Cramer, 4)])

        return results

# 3. Confidence Intervals for Cohen's w

    @staticmethod
    def Cohens_w_CI_From_Effect_Size(params: dict) -> dict:
        
        Cohens_w = params["Cohen's w (Phi)"]
        Sample_Size = params["Sample Size"]
        Degrees_of_Freedom = params["Degrees Of Freedom"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100

        Chi_Score = np.sqrt(Cohens_w * Sample_Size)
        lower_ncp, upper_ncp = ncp_ci(Chi_Score, Degrees_of_Freedom, confidence_level)
        LowerCi_Cramer = np.sqrt(lower_ncp / Sample_Size / Degrees_of_Freedom)
        UpperCi_Cramer = np.sqrt(upper_ncp / Sample_Size / Degrees_of_Freedom)

        results = {}
        results["Confidence Intervals for Cohen's w (Phi)"] = np.array([round(LowerCi_Cramer, 4), round(UpperCi_Cramer, 4)])

        return results

# 4. Confidence Intervals for Contingency Coefficient

    @staticmethod
    def Contingency_Coefficient_CI_From_Effect_Size(params: dict) -> dict:
        
        CC = params["Contingency Coefficient)"]
        Sample_Size = params["Sample Size"]
        Degrees_of_Freedom = params["Degrees Of Freedom"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100

        Chi_Score = (CC**2 * Sample_Size) / (CC**2 - 1)
        lower_ncp, upper_ncp = ncp_ci(Chi_Score, Degrees_of_Freedom, confidence_level)
        LowerCi_Cramer = np.sqrt(lower_ncp / Sample_Size / Degrees_of_Freedom)
        UpperCi_Cramer = np.sqrt(upper_ncp / Sample_Size / Degrees_of_Freedom)

        results = {}
        results["Confidence Intervals for Contingency Coefficient"] = np.array([round(LowerCi_Cramer, 4), round(UpperCi_Cramer, 4)])

        return results

# 5. Confidence Intervals for Contingency Coefficeint from Chi Score

    @staticmethod
    def Contingency_Coefficient_CI_From_Chi_Score(params: dict) -> dict:
        
        Chi_Score = params["Cohen's w (Phi)"]
        Sample_Size = params["Sample Size"]
        Degrees_of_Freedom = params["Degrees Of Freedom"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100

        CC = np.sqrt(Chi_Score / (Chi_Score + Sample_Size))
        lower_ncp, upper_ncp = ncp_ci(Chi_Score, Degrees_of_Freedom, confidence_level)
        LowerCi_Cramer = np.sqrt(lower_ncp / Sample_Size / Degrees_of_Freedom)
        UpperCi_Cramer = np.sqrt(upper_ncp / Sample_Size / Degrees_of_Freedom)

        results = {}
        results["Contingency Coefficient"] = CC       
        results["Confidence Intervals for Contingency Coefficient"] = np.array([round(LowerCi_Cramer, 4), round(UpperCi_Cramer, 4)])

        return results

# 6. Confidence Intervals for Spearman 
    @staticmethod
    def Spearman_Correlation_CI(params: dict) -> dict:
        
        Spearman = params["Spearman's r)"]
        sample_size = params["Sample Size"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100

        Z_rho = 0.5 * np.log((1 + Spearman) / (1 - Spearman))
        Fieller_Standard_Error = np.sqrt(1.06 / (sample_size-3)) # 1957
        Caruso_and_Cliff = np.sqrt(1 / (sample_size - 2)) +  (abs(Z_rho)/(6*sample_size + 4 *np.sqrt(sample_size)))# 1997
        Bonett_Wright_Standard_Error = (1 + Spearman / 2) / (sample_size - 3) # 2000 - I use this as the default

        # Confidence Interval Based on the Different Standard Errors (Default is Bonette Wright)
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
        tcrit = t.ppf(1 - (1 - confidence_level) / 2, sample_size-2)
        Lower_ci_BW_z = math.tanh(Z_rho - zcrit * Bonett_Wright_Standard_Error)
        Upper_ci_BW_z = math.tanh(Z_rho + zcrit * Bonett_Wright_Standard_Error)
        Lower_ci_BW_t = math.tanh(Z_rho - tcrit * Bonett_Wright_Standard_Error)
        Upper_ci_BW_t = math.tanh(Z_rho + tcrit * Bonett_Wright_Standard_Error)
        Lower_ci_Fieller = math.tanh(Z_rho - zcrit * Fieller_Standard_Error)
        Upper_ci_Fieller = math.tanh(Z_rho + zcrit * Fieller_Standard_Error)
        Lower_ci_CC_z = math.tanh(Z_rho - zcrit * Caruso_and_Cliff)
        Upper_ci_CC_z = math.tanh(Z_rho + zcrit * Caruso_and_Cliff)
        Lower_ci_CC_t = math.tanh(Z_rho - tcrit * Caruso_and_Cliff)
        Upper_ci_CC_t = math.tanh(Z_rho + tcrit * Caruso_and_Cliff)
        Lower_ci_Fisher_t = math.tanh(Z_rho - tcrit * (np.sqrt(1/(sample_size-3))))
        Upper_ci_Fisher_t = math.tanh(Z_rho + tcrit * (np.sqrt(1/(sample_size-3))))
        Lower_ci_Fisher_z = math.tanh(Z_rho - zcrit * (np.sqrt(1/(sample_size-3))))
        Upper_ci_Fisher_z = math.tanh(Z_rho + zcrit * (np.sqrt(1/(sample_size-3))))

        results = {}
        results["Confidence Intervals (Bonett Wright Z)"] = f"({round(Lower_ci_BW_z, 4)}, {round(Upper_ci_BW_z, 4)})"
        results["Confidence Intervals (Bonett Wright t)"] = f"({round(Lower_ci_BW_t, 4)}, {round(Upper_ci_BW_t, 4)})"
        results["Confidence Intervals (Fieller)"] = f"({round(Lower_ci_Fieller, 4)}, {round(Upper_ci_Fieller, 4)})"
        results["Confidence Intervals (Caruso & Cliff Z)"] = f"({round(Lower_ci_CC_z, 4)}, {round(Upper_ci_CC_z, 4)})"
        results["Confidence Intervals (Caruso & Cliff t)"] = f"({round(Lower_ci_CC_t, 4)}, {round(Upper_ci_CC_t, 4)})"
        results["Confidence Intervals (Fisher Z)"] = f"({round(Lower_ci_Fisher_z, 4)}, {round(Upper_ci_Fisher_z, 4)})"
        results["Confidence Intervals (Fisher t)"] = f"({round(Lower_ci_Fisher_t, 4)}, {round(Upper_ci_Fisher_t, 4)})" 
 
        return results
    
# 7. Confidence Intervals for Pearson's r
    @staticmethod
    def Pearson_Correlation_CI(params: dict) -> dict:
        
        Pearosn = params["Pearson's r)"]
        sample_size = params["Sample Size"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)

        Zr = 0.5 * np.log((1+Pearosn) / (1-Pearosn)) # Fisher's Transformation
        Standard_Error_ZR =  1 / (np.sqrt(sample_size-3)) #Fisher (1921) Approximation of the Variance
        Z_Lower = Zr - zcrit * Standard_Error_ZR
        Z_Upper = Zr + zcrit * Standard_Error_ZR
        Pearosn_Fisher_CI_lower = (np.exp(2*Z_Lower) - 1) / (np.exp(2*Z_Lower) + 1)
        Pearosn_Fisher_CI_upper = (np.exp(2*Z_Upper) - 1) / (np.exp(2*Z_Upper) + 1)
        
        Bonett_Wright_Standard_Error = (1 + Pearosn / 2) / (sample_size - 3) # 2000 - I use this as the default
        Z_Lower_bw = Zr - zcrit * Bonett_Wright_Standard_Error
        Z_Upper_bw = Zr + zcrit * Bonett_Wright_Standard_Error        
        
        Pearosn_BW_CI_lower = (np.exp(2*Z_Lower_bw) - 1) / (np.exp(2*Z_Lower_bw) + 1)
        Pearosn_bw_CI_upper = (np.exp(2*Z_Upper_bw) - 1) / (np.exp(2*Z_Upper_bw) + 1)

        results = {}
        results["Confidence Intervals (Bonett Wright Z)"] = f"({round(Pearosn_Fisher_CI_lower, 4)}, {round(Pearosn_Fisher_CI_upper, 4)})"
        results["Confidence Intervals (Fisher Z)"] = f"({round(Pearosn_BW_CI_lower, 4)}, {round(Pearosn_bw_CI_upper, 4)})"
 
        return results


# 8. Confidence Intervals for R square
    @staticmethod
    def R_Square_CI(params: dict) -> dict:
        
        Rsquare = params["R Square)"]
        sample_size = params["Sample Size"]
        Number_of_Predictors = params["Number of Predictors"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
        tcrit = t.ppf(1 - (1 - confidence_level) / 2, sample_size-2)

        # 3. Olkin & Fin 1995
        Standard_Error_R_Sqaure = np.sqrt((4 * Rsquare * (1- Rsquare)**2*(sample_size-Number_of_Predictors-1)**2) / ((sample_size**2 -1)*(sample_size+3)))# Olkin & Finn, 1995
        R2_CI_Olkin_Fin_Lower = np.sqrt(Rsquare - zcrit * Standard_Error_R_Sqaure)
        R2_CI_Olkin_Fin_Upper = np.sqrt(Rsquare + zcrit * Standard_Error_R_Sqaure)
        R2_CI_Olkin_Fin_Lower_t_crit = np.sqrt(Rsquare - tcrit * Standard_Error_R_Sqaure)
        R2_CI_Olkin_Fin_Upper_t_crit = np.sqrt(Rsquare + tcrit * Standard_Error_R_Sqaure)

        # 4. Non-Central CI's based on F distribtion
        F_Score = (Rsquare / Number_of_Predictors) / ((1-Rsquare) / (sample_size - Number_of_Predictors - 1))
        Lowerc_NCF_CI, Upper_NCF_CI = NonCentralCiF(F_Score, Number_of_Predictors, sample_size-Number_of_Predictors-1, confidence_level)
        R2_non_central_CI_lower = np.sqrt(Lowerc_NCF_CI / (Lowerc_NCF_CI + (Number_of_Predictors + (sample_size-Number_of_Predictors-1) + 1)))
        R2_non_central_CI_upper = np.sqrt(Upper_NCF_CI / (Upper_NCF_CI +(Number_of_Predictors + (sample_size-Number_of_Predictors-1) + 1)))

        results = {}
        results["Confidence Intervals Olkin & Fin (Z)"] = f"({round(R2_CI_Olkin_Fin_Lower, 4)}, {round(R2_CI_Olkin_Fin_Upper, 4)})"
        results["Confidence Intervals Olkin & Fin (t)"] = f"({round(R2_CI_Olkin_Fin_Lower_t_crit, 4)}, {round(R2_CI_Olkin_Fin_Upper_t_crit, 4)})"
        results["Confidence Intervals Non Central"] = f"({round(R2_non_central_CI_lower, 4)}, {round(R2_non_central_CI_upper, 4)})"
 
        return results
    
    # Things to Consider
    # 1. Add Helland (statpsych) CI for R square
