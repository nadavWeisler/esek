
######################
# CI Constructor eta #
######################

from scipy.stats import  ncf
import numpy as np


def Confidence_Interval_Partial_Eta_Square_Family(f_statistic, df1, df2, confidence_level):
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
    LowerCi_NCF_Final = LowerCi(f_statistic, df1, df2, upper_limit, lower_ci_difference_value)[0]
    UpperCi__NCF_Final = UpperCi(f_statistic, df1, df2, lower_limit, lower_ci_difference_value)[0]
    
    # The R square Method
    Lower_Partial_Eta_Square_CI_Method_1 = LowerCi_NCF_Final / (LowerCi_NCF_Final+df1+df2+1)
    Upper_Partial_Eta_Square_CI_Method_1 = UpperCi__NCF_Final / (UpperCi__NCF_Final+df1+df2+1)
    
    # F Converted Method
    Lower_Partial_Eta_Square_CI_F_converted = ((LowerCi_NCF_Final / df1) * df1) / ((LowerCi_NCF_Final / df1) * df1 + df2)
    Upper_Partial_Eta_Square_CI_F_Converted = ((UpperCi__NCF_Final / df1) * df1) / ((UpperCi__NCF_Final / df1) * df1 + df2)
    Lower_Partial_Omega_Square_CI_F_Converted = ((LowerCi_NCF_Final / df1-1) * df1) / ((LowerCi_NCF_Final / df1) * df1 + df2 + 1)
    Upper_Partial_Omega_Square_CI_F_Converted = ((UpperCi__NCF_Final / df1-1) * df1) / ((UpperCi__NCF_Final / df1) * df1 + df2 + 1)
    Lower_Partial_Epsilon_Square_CI_F_Converted = ((LowerCi_NCF_Final / df1-1) * df1) / ((LowerCi_NCF_Final / df1) * df1 + df2)
    Upper_Partial_Epsilon_Square_CI_F_Converted = ((UpperCi__NCF_Final / df1-1) * df1) / ((UpperCi__NCF_Final / df1) * df1 + df2)
        
    return Lower_Partial_Eta_Square_CI_Method_1, Upper_Partial_Eta_Square_CI_Method_1, Lower_Partial_Eta_Square_CI_F_converted, Upper_Partial_Eta_Square_CI_F_Converted, Lower_Partial_Omega_Square_CI_F_Converted, \
            Upper_Partial_Omega_Square_CI_F_Converted, Lower_Partial_Epsilon_Square_CI_F_Converted, Upper_Partial_Epsilon_Square_CI_F_Converted

class CI_Constructor_Partial_Eta_Square():
    
    #########################################################
    ## 1.1 CI for Partial Eta Square Family From F Score ####
    #########################################################

    @staticmethod
    def Partial_Eta_Square_From_F(params: dict) -> dict:
        
        F_Value = params["F Value"]
        df1 = params["Degrees Of Freedom Numerator"]
        df2 = params["Degrees Of Freedom Denominator"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100
        Lower_Partial_Eta_Square_CI_Method_1, Upper_Partial_Eta_Square_CI_Method_1, Lower_Partial_Eta_Square_CI_F_converted, Upper_Partial_Eta_Square_CI_F_Converted, Lower_Partial_Omega_Square_CI_F_Converted, \
            Upper_Partial_Omega_Square_CI_F_Converted, Lower_Partial_Epsilon_Square_CI_F_Converted, Upper_Partial_Epsilon_Square_CI_F_Converted= Confidence_Interval_Partial_Eta_Square_Family(F_Value, df1, df2, confidence_level)


        results = {}
        results["Confidence Intervals Partial Eta Square (Fleishman)"] = np.array([round(Lower_Partial_Eta_Square_CI_Method_1, 4), round(Upper_Partial_Eta_Square_CI_Method_1, 4)])
        results["Confidence Intervals Partial Eta Square (Converted F Score)"] = np.array([round(Lower_Partial_Eta_Square_CI_F_converted, 4), round(Upper_Partial_Eta_Square_CI_F_Converted, 4)])
        results["Confidence Intervals Partial Omega Square (Converted F Score)"] = np.array([round(Lower_Partial_Omega_Square_CI_F_Converted, 4), round(Upper_Partial_Omega_Square_CI_F_Converted, 4)])
        results["Confidence Intervals Partial Epsilon Square (Converted F Score)"] = np.array([round(Lower_Partial_Epsilon_Square_CI_F_Converted, 4), round(Upper_Partial_Epsilon_Square_CI_F_Converted, 4)])

        return results

    ###################################################################
    ## 1.2 CI for Partial Eta Square Family From Effect Size Score ####
    ###################################################################
    
    @staticmethod
    def Partial_Eta_Square_From_effect_size(params: dict) -> dict:
        
        Partial_Eta_Square = params["Partial Eta Square"]
        df1 = params["Degrees Of Freedom Numerator"]
        df2 = params["Degrees Of Freedom Denominator"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100
        F_Value = ((-1 * Partial_Eta_Square) * df2) / (Partial_Eta_Square*df1 - df1)
        Lower_Partial_Eta_Square_CI_Method_1, Upper_Partial_Eta_Square_CI_Method_1, Lower_Partial_Eta_Square_CI_F_converted, Upper_Partial_Eta_Square_CI_F_Converted, Lower_Partial_Omega_Square_CI_F_Converted, \
            Upper_Partial_Omega_Square_CI_F_Converted, Lower_Partial_Epsilon_Square_CI_F_Converted, Upper_Partial_Epsilon_Square_CI_F_Converted= Confidence_Interval_Partial_Eta_Square_Family(F_Value, df1, df2, confidence_level)


        results = {}
        results["Confidence Intervals Partial Eta Square (Fleishman)"] = np.array([round(Lower_Partial_Eta_Square_CI_Method_1, 4), round(Upper_Partial_Eta_Square_CI_Method_1, 4)])
        results["Confidence Intervals Partial Eta Square (Converted F Score)"] = np.array([round(Lower_Partial_Eta_Square_CI_F_converted, 4), round(Upper_Partial_Eta_Square_CI_F_Converted, 4)])
        results["Confidence Intervals Partial Omega Square (Converted F Score)"] = np.array([round(Lower_Partial_Omega_Square_CI_F_Converted, 4), round(Upper_Partial_Omega_Square_CI_F_Converted, 4)])
        results["Confidence Intervals Partial Epsilon Square (Converted F Score)"] = np.array([round(Lower_Partial_Epsilon_Square_CI_F_Converted, 4), round(Upper_Partial_Epsilon_Square_CI_F_Converted, 4)])

        return results

# Things to consider
# 1. Confidence intervals for the Fleishman method can be converted into confidecn intervals for omega sqare however, While confidence intervals for ε2 can be constructed using the same transformation that links it with η2, there are several arguments for not using them in practice. See Smithson (2003, 54) for further details.can be converted 


