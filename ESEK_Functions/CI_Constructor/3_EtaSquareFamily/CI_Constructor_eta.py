
######################
# CI Constructor eta #
######################

from scipy.stats import  ncf
import numpy as np


def Confidence_Interval_Partial_Eta_Square_Family(F_Statistic, df1, df2, confidence_level):
    Upper_Limit = 1 - (1 - confidence_level) / 2
    Lower_Limit = 1 - Upper_Limit
    Lower_CI_Difference_Value = 1

    def Lower_CI(F_Statistic, df1, df2, Upper_Limit, Lower_CI_Difference_Value):
        Lower_Bound = [0.001, F_Statistic / 2, F_Statistic]       
        while ncf.cdf(F_Statistic, df1, df2, Lower_Bound[0]) < Upper_Limit:
            return [0, ncf.cdf(F_Statistic, df1, df2)] if ncf.cdf(F_Statistic, df1, df2) < Upper_Limit else None
            Lower_Bound = [Lower_Bound[0] / 4, Lower_Bound[0], Lower_Bound[2]]   
        while ncf.cdf(F_Statistic, df1, df2, Lower_Bound[2]) > Upper_Limit: Lower_Bound = [Lower_Bound[0], Lower_Bound[2], Lower_Bound[2] + F_Statistic]     
        while Lower_CI_Difference_Value > 0.0000001:
            Lower_Bound = [Lower_Bound[0], (Lower_Bound[0] + Lower_Bound[1]) / 2, Lower_Bound[1]] if ncf.cdf(F_Statistic, df1, df2, Lower_Bound[1]) < Upper_Limit else [Lower_Bound[1], (Lower_Bound[1] + Lower_Bound[2]) / 2, Lower_Bound[2]]  
            Lower_CI_Difference_Value = abs(ncf.cdf(F_Statistic, df1, df2, Lower_Bound[1]) - Upper_Limit)        
        return [Lower_Bound[1]]
    
    def Upper_CI(F_Statistic, df1, df2, Lower_Limit, Lower_CI_Difference_Value):
        Upper_Bound = [F_Statistic, 2 * F_Statistic, 3 * F_Statistic]
        while ncf.cdf(F_Statistic, df1, df2, Upper_Bound[0]) < Lower_Limit:Upper_Bound = [Upper_Bound[0] / 4, Upper_Bound[0], Upper_Bound[2]]
        while ncf.cdf(F_Statistic, df1, df2, Upper_Bound[2]) > Lower_Limit: Upper_Bound = [Upper_Bound[0], Upper_Bound[2], Upper_Bound[2] + F_Statistic]
        while Lower_CI_Difference_Value > 0.00001: Upper_Bound = [Upper_Bound[0], (Upper_Bound[0] + Upper_Bound[1]) / 2, Upper_Bound[1]] if ncf.cdf(F_Statistic, df1, df2, Upper_Bound[1]) < Lower_Limit else [Upper_Bound[1], (Upper_Bound[1] + Upper_Bound[2]) / 2, Upper_Bound[2]]; Lower_CI_Difference_Value = abs(ncf.cdf(F_Statistic, df1, df2, Upper_Bound[1]) - Lower_Limit)
        return [Upper_Bound[1]]
    
    # Calculate lower and upper bounds
    Lower_CI_NCF_Final = Lower_CI(F_Statistic, df1, df2, Upper_Limit, Lower_CI_Difference_Value)[0]
    Upper_CI__NCF_Final = Upper_CI(F_Statistic, df1, df2, Lower_Limit, Lower_CI_Difference_Value)[0]
    
    # The R square Method
    Lower_Partial_Eta_Square_CI_Method_1 = Lower_CI_NCF_Final / (Lower_CI_NCF_Final+df1+df2+1)
    Upper_Partial_Eta_Square_CI_Method_1 = Upper_CI__NCF_Final / (Upper_CI__NCF_Final+df1+df2+1)
    
    # F Converted Method
    Lower_Partial_Eta_Square_CI_F_converted = ((Lower_CI_NCF_Final / df1) * df1) / ((Lower_CI_NCF_Final / df1) * df1 + df2)
    Upper_Partial_Eta_Square_CI_F_Converted = ((Upper_CI__NCF_Final / df1) * df1) / ((Upper_CI__NCF_Final / df1) * df1 + df2)
    Lower_Partial_Omega_Square_CI_F_Converted = ((Lower_CI_NCF_Final / df1-1) * df1) / ((Lower_CI_NCF_Final / df1) * df1 + df2 + 1)
    Upper_Partial_Omega_Square_CI_F_Converted = ((Upper_CI__NCF_Final / df1-1) * df1) / ((Upper_CI__NCF_Final / df1) * df1 + df2 + 1)
    Lower_Partial_Epsilon_Square_CI_F_Converted = ((Lower_CI_NCF_Final / df1-1) * df1) / ((Lower_CI_NCF_Final / df1) * df1 + df2)
    Upper_Partial_Epsilon_Square_CI_F_Converted = ((Upper_CI__NCF_Final / df1-1) * df1) / ((Upper_CI__NCF_Final / df1) * df1 + df2)
        
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


