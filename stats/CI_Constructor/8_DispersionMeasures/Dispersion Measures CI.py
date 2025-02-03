
###################################################
#### Confidence Intervals for Dispersion Measures #
###################################################

from scipy.stats import norm
import numpy as np
import math
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep
from scipy.optimize import newton, root_scalar
import rpy2.robjects as robjects


class CI_Constructor_Dispersion_Measures():
    
    ##########################################
    ## 1.1 Mean Absolute Deviation ###########
    ##########################################

    @staticmethod
    def One_Sample_MAD_CI(params: dict) -> dict:
        
        Column_1 = params["Column 1"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100

        # Preperations
        Sample_Size = len(Column_1)
        Median = np.median(Column_1)
        MAD = np.mean(np.abs(Column_1 - Median))
        MAD_corrected = MAD * (Sample_Size / (Sample_Size - 1))
        Standard_Deviation = np.std(Column_1, ddof = 1)
        Mean = np.mean(Column_1)

        # Confidence Intervals
        Z_Critical = norm.ppf( 1 - (1 - confidence_level) / 2 )
        SE_MAD = np.sqrt((((Mean - Median) / MAD) ** 2 + ((Standard_Deviation / MAD) ** 2) - 1) / Sample_Size)
        Lower_CI_MAD = np.exp(np.log(MAD_corrected) - Z_Critical * SE_MAD)
        Upper_CI_MAD = np.exp(np.log(MAD_corrected) + Z_Critical * SE_MAD)

        results = {}

        # Confidence Intervals for One Sample Proportion
        results["Mean Abosolute Deviation"] = np.round(np.array(MAD),4)
        results["Mean Abosolute Deviation Corrected"] = np.round(np.array(MAD_corrected),4)
        results["Confidence Intervals for the Mean Abosolute Deviation Corrected"] = np.round(np.array([Lower_CI_MAD, Upper_CI_MAD]),4)

        return results


    #####################################################
    ## 1.2 Mean Absolute Deviation Difference ###########
    #####################################################

    @staticmethod
    def MAD_CI(params: dict) -> dict:
        
        Column_1 = params["Column 1"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100

        # Preperations
        Sample_Size = len(Column_1)
        Median = np.median(Column_1)
        MAD = np.mean(np.abs(Column_1 - Median))
        MAD_corrected = MAD * (Sample_Size / (Sample_Size - 1))
        Standard_Deviation = np.std(Column_1, ddof = 1)
        Mean = np.mean(Column_1)

        # Confidence Intervals
        Z_Critical = norm.ppf( 1 - (1 - confidence_level) / 2 )
        SE_MAD = np.sqrt((((Mean - Median) / MAD) ** 2 + ((Standard_Deviation / MAD) ** 2) - 1) / Sample_Size)
        Lower_CI_MAD = np.exp(np.log(MAD_corrected) - Z_Critical * SE_MAD)
        Upper_CI_MAD = np.exp(np.log(MAD_corrected) + Z_Critical * SE_MAD)

        results = {}

        # Confidence Intervals for One Sample Proportion
        results["Mean Abosolute Deviation"] = np.round(np.array(MAD),4)
        results["Mean Abosolute Deviation Corrected"] = np.round(np.array(MAD_corrected),4)
        results["Confidence Intervals for the Mean Abosolute Deviation Corrected"] = np.round(np.array([Lower_CI_MAD, Upper_CI_MAD]),4)

        return results
    


    @staticmethod
    def SD_Ratio_CI(params: dict) -> dict:
        
        Column_1 = params["Column 1"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100

        # Preperations
        Sample_Size = len(Column_1)
        Median = np.median(Column_1)
        MAD = np.mean(np.abs(Column_1 - Median))
        MAD_corrected = MAD * (Sample_Size / (Sample_Size - 1))
        Standard_Deviation = np.std(Column_1, ddof = 1)
        Mean = np.mean(Column_1)

        # Confidence Intervals
        Z_Critical = norm.ppf( 1 - (1 - confidence_level) / 2 )
        SE_MAD = np.sqrt((((Mean - Median) / MAD) ** 2 + ((Standard_Deviation / MAD) ** 2) - 1) / Sample_Size)
        Lower_CI_MAD = np.exp(np.log(MAD_corrected) - Z_Critical * SE_MAD)
        Upper_CI_MAD = np.exp(np.log(MAD_corrected) + Z_Critical * SE_MAD)

        results = {}

        # Confidence Intervals for One Sample Proportion
        results["Mean Abosolute Deviation"] = np.round(np.array(MAD),4)
        results["Mean Abosolute Deviation Corrected"] = np.round(np.array(MAD_corrected),4)
        results["Confidence Intervals for the Mean Abosolute Deviation Corrected"] = np.round(np.array([Lower_CI_MAD, Upper_CI_MAD]),4)

        return results



    @staticmethod
    def IQR_CI(params: dict) -> dict:
        
        Column_1 = params["Column 1"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100

        # Preperations
        Sample_Size = len(Column_1)
        Median = np.median(Column_1)
        MAD = np.mean(np.abs(Column_1 - Median))
        MAD_corrected = MAD * (Sample_Size / (Sample_Size - 1))
        Standard_Deviation = np.std(Column_1, ddof = 1)
        Mean = np.mean(Column_1)

        # Confidence Intervals
        Z_Critical = norm.ppf( 1 - (1 - confidence_level) / 2 )
        SE_MAD = np.sqrt((((Mean - Median) / MAD) ** 2 + ((Standard_Deviation / MAD) ** 2) - 1) / Sample_Size)
        Lower_CI_MAD = np.exp(np.log(MAD_corrected) - Z_Critical * SE_MAD)
        Upper_CI_MAD = np.exp(np.log(MAD_corrected) + Z_Critical * SE_MAD)

        results = {}

        # Confidence Intervals for One Sample Proportion
        results["Mean Abosolute Deviation"] = np.round(np.array(MAD),4)
        results["Mean Abosolute Deviation Corrected"] = np.round(np.array(MAD_corrected),4)
        results["Confidence Intervals for the Mean Abosolute Deviation Corrected"] = np.round(np.array([Lower_CI_MAD, Upper_CI_MAD]),4)

        return results