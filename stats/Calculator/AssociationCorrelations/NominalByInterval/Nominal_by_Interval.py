

import numpy as np
from scipy.stats import pearsonr, ncf,f, norm
import math 

x = np.array([0,0,0,0,0,0,0, 0, 0, 1,1,1])
y = np.array([5,6,7,8,8,8,12,13,14,11,15,16])
confidence_level = 0.95

# Relevant Functions
def NonCentralCiF(f_statistic, df1, df2, confidence_level):
    upper_limit = 1 - (1 - confidence_level) / 2
    lower_limit = 1 - upper_limit
    lower_ci_difference_value = 1

    def LowerCi(f_statistic, df1, df2, upper_limit, lower_ci_difference_value):
        lower_bound = [0.001, f_statistic / 2, f_statistic]       
        while ncf.cdf(f_statistic, df1, df2, lower_bound[0]) < upper_limit:
            return [0, f.cdf(f_statistic, df1, df2)] if f.cdf(f_statistic, df1, df2) < upper_limit else None
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
    LowerCi_Final = LowerCi(f_statistic, df1, df2, upper_limit, lower_ci_difference_value)[0]
    UpperCi_Final = UpperCi(f_statistic, df1, df2, lower_limit, lower_ci_difference_value)[0]

    return LowerCi_Final, UpperCi_Final

def point_biserial_correlation(x,y,confidence_level):
    Pearson_Correlation_Test = pearsonr(x,y)
    point_biserial_correlation, pvalue = Pearson_Correlation_Test

    # Maximum Corrected Point Biserial Correlation
    x_sorted = x[np.argsort(x)]
    y_sorted = y[np.argsort(y)]
    ss_total_x_indpednent = np.var(y) * len(x)
    ss_between_x_max = np.sum([(len(y_sorted[x_sorted == value]) * (np.mean(y_sorted[x_sorted == value]) - np.mean(y_sorted)) ** 2) for value in np.unique(x_sorted)])
    Pearson_max_x_indpendent = np.sqrt(ss_between_x_max/ss_total_x_indpednent)
    Max_Corrected_Point_Biserial = point_biserial_correlation/Pearson_max_x_indpendent
    Approximated_Point_Biserial_Correlation = point_biserial_correlation + (point_biserial_correlation*(1-point_biserial_correlation**2)) / (2*(len(x)-3))

    # Confidence Intervals Fisher Trnasformed
    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
    lowerCI_rpb, upperCI_rpb = Pearson_Correlation_Test.confidence_interval(confidence_level)
    SE_rpb = 1/ np.sqrt(len(x) - 3)

    Lower_ci_Max_Corrected = math.tanh(math.atanh(Max_Corrected_Point_Biserial) - zcrit * SE_rpb)
    Upper_ci_Max_Corrected = math.tanh(math.atanh(Max_Corrected_Point_Biserial) + zcrit * SE_rpb)
    Lower_ci_Approximated_Point_Biserial_Correlation = math.tanh(math.atanh(Approximated_Point_Biserial_Correlation) - zcrit * SE_rpb)
    Upper_ci_Approximated_Point_Biserial_Correlation = math.tanh(math.atanh(Approximated_Point_Biserial_Correlation) + zcrit * SE_rpb)


    results = {}
    results["Point Biserial Correlation"] = (point_biserial_correlation)
    results["p-value"] = (pvalue)
    results["Confidence Intervals Point Biserial Correlation (Fisher)"] = f"({round(lowerCI_rpb, 4)}, {round(upperCI_rpb, 4)})"
    results["Approximated point biserial correlation (Hedges & Olkin, 1985)"] = (Approximated_Point_Biserial_Correlation)
    results["Fisher Transformed Confidence Intervals Approximated Point Biserial Correlation)"] = f"({round(Lower_ci_Approximated_Point_Biserial_Correlation, 4)}, {round(Upper_ci_Approximated_Point_Biserial_Correlation, 4)})"
    results["Max Corrected point biserial correlation (Hedges & Olkin, 1985)"] = (Max_Corrected_Point_Biserial)
    results["Fisher Transformed Confidence Intervals Max Corrected Point Biserial Correlation)"] = f"({round(Lower_ci_Max_Corrected, 4)}, {round(Upper_ci_Max_Corrected, 4)})"


    result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
    return result_str

def Eta_Correlation_Ratio(x,y,confidence_level):
    ss_between_x_independent = np.sum([(len(y[x == value]) * (np.mean(y[x == value]) - np.mean(y)) ** 2) for value in np.unique(x)])
    ss_total_x_indpednent = np.sum((y - np.mean(y)) ** 2)
    eta_x_independent = np.sqrt(ss_between_x_independent/ss_total_x_indpednent)
    ss_between_y_independent = np.sum([(len(x[y == value]) * (np.mean(x[y == value]) - np.mean(x)) ** 2) for value in np.unique(y)])
    ss_total_y_indpednent = np.sum((x - np.mean(x)) ** 2)
    eta_y_independent = np.sqrt(ss_between_y_independent/ss_total_y_indpednent)

    # Attenuated Corrected Eta by Metsamuronen (2023):
    x_sorted = x[np.argsort(x)]
    y_sorted = y[np.argsort(y)]
    ss_between_x_max = np.sum([(len(y_sorted[x_sorted == value]) * (np.mean(y_sorted[x_sorted == value]) - np.mean(y_sorted)) ** 2) for value in np.unique(x_sorted)])
    eta_max_x_indpendent = np.sqrt(ss_between_x_max/ss_total_x_indpednent)

    ss_between_y_max = np.sum([(len(x_sorted[y_sorted == value]) * (np.mean(x_sorted[y_sorted == value]) - np.mean(x_sorted)) ** 2) for value in np.unique(y_sorted)])
    eta_max_y_indpendent = np.sqrt(ss_between_y_max/ss_total_y_indpednent)

    Attenuated_Corrected_Eta_X_Independent = (eta_x_independent/eta_max_x_indpendent)
    Attenuated_Corrected_Eta_Y_Independent = (eta_y_independent/eta_max_y_indpendent)

    # Confidence Intervals based on the Non Central Distribution
    Number_Of_Groups = len(np.unique(x))
    sample_size = len(x)
    df1 = Number_Of_Groups-1
    df2 = sample_size - Number_Of_Groups

    F_eta_x_independent = ((-eta_x_independent**2 * df2) / (df1*(eta_x_independent**2-1)))
    F_eta_y_independent = ((-eta_y_independent**2 * df2) / (df1*(eta_y_independent**2-1)))
    F_eta_corrected_x_independent = ((-Attenuated_Corrected_Eta_X_Independent**2 * df2) / (df1*(Attenuated_Corrected_Eta_X_Independent**2-1)))
    F_eta_corrected_y_independent = ((-Attenuated_Corrected_Eta_Y_Independent**2 * df2) / (df1*(Attenuated_Corrected_Eta_Y_Independent**2-1)))

    NCP_F_eta_x_independent_lower, NCP_F_eta_x_independent_upper = NonCentralCiF(F_eta_x_independent, df1, df2, confidence_level)
    NCP_F_eta_y_independent_lower, NCP_F_eta_y_independent_upper = NonCentralCiF(F_eta_y_independent, df1, df2, confidence_level)
    NCP_F_eta_corrected_x_independent_lower, NCP_F_eta_corrected_x_independent_upper = NonCentralCiF(F_eta_corrected_x_independent, df1, df2, confidence_level)
    NCP_F_eta_corrected_y_independent_lower, NCP_F_eta_corrected_y_independent_upper = NonCentralCiF(F_eta_corrected_y_independent, df1, df2, confidence_level)

    Lower_Ci_eta_x_independent = np.sqrt(NCP_F_eta_x_independent_lower / (NCP_F_eta_x_independent_lower + df2))
    Upper_Ci_eta_x_independent = np.sqrt(NCP_F_eta_x_independent_upper / (NCP_F_eta_x_independent_upper + df2)) 
    Lower_Ci_eta_y_independent = np.sqrt(NCP_F_eta_y_independent_lower / (NCP_F_eta_y_independent_lower + df2)) 
    Upper_Ci_eta_y_independent = np.sqrt(NCP_F_eta_y_independent_upper / (NCP_F_eta_y_independent_upper + df2))
    Lower_Ci_eta_corrected_x_independent = np.sqrt(NCP_F_eta_corrected_x_independent_lower / (NCP_F_eta_corrected_x_independent_lower + df2))
    Upper_Ci_eta_corrected_x_independent = np.sqrt(NCP_F_eta_corrected_x_independent_upper / (NCP_F_eta_corrected_x_independent_upper + df2))
    Lower_Ci_eta_corrected_y_independent = np.sqrt(NCP_F_eta_corrected_y_independent_lower / (NCP_F_eta_corrected_y_independent_lower + df2))
    Upper_Ci_eta_corrected_y_independent = np.sqrt(NCP_F_eta_corrected_y_independent_upper / (NCP_F_eta_corrected_y_independent_upper + df2))

    results = {}
    results["Eta (Variable X is Indpendent)"] = (eta_x_independent)
    results["Eta (Variable X is Indpendent) - Confidence Intervals"] = [Lower_Ci_eta_x_independent, Upper_Ci_eta_x_independent]
    results["Eta (Variable Y is Indpendent)"] = (eta_y_independent)
    results["Eta (Variable Y is Indpendent) - Confidence Intervals"] = [Lower_Ci_eta_y_independent, Upper_Ci_eta_y_independent]
    results["Attenuated Correct Eta (Variable X is Indpendent)"] = (Attenuated_Corrected_Eta_X_Independent)
    results["Attenuated Correct Eta (Variable X is Indpendent) - Confidence Intervals"] = [Lower_Ci_eta_corrected_x_independent, Upper_Ci_eta_corrected_x_independent]
    results["Attenuated Correct Eta (Variable Y is Indpendent)"] = (Attenuated_Corrected_Eta_Y_Independent)
    results["Attenuated Correct Eta (Variable Y is Indpendent) - Confidence Intervals"] = [Lower_Ci_eta_corrected_y_independent, Upper_Ci_eta_corrected_y_independent] 


    result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
    return result_str

class NominalInterval():
    @staticmethod
    def Nominal_by_Interval_From_Data(params: dict) -> dict:
        # Set Params:
        Nominlal_Variable = params["Nominal"]
        Interval_Variable = params["Interval"]
        confidence_level_percentages = params["Confidence Level"]

        # Convrt the variables into vectors
        confidence_level = confidence_level_percentages/ 100

        # if the Nominal variable is not Numeric convert it to a numeric one
        Nominlal_Variable = np.array([({val: i for i, val in enumerate(np.unique(Nominlal_Variable))})[val] for val in Nominlal_Variable])

        point_biserial_correlation_output = point_biserial_correlation(Nominlal_Variable, Interval_Variable, confidence_level) if len(np.unique(Nominlal_Variable))==2 else "Point Biserial Correlation only relvent when the Nominal Variable has 2 levels"
        eta_output = Eta_Correlation_Ratio(Nominlal_Variable,Interval_Variable,confidence_level)

        results = {}
        results["eta_output "] =  eta_output
        results["_______________________"] = ""
        results["Point Biserial Correlation"] = point_biserial_correlation_output

        return results

    @staticmethod
    def Nominal_by_Interval_From_Contingency_Table(params: dict) -> dict:
        
        # Set Params:
        Contingency_Table = params["Table"]
        confidence_level_percentages = params["Confidence Level"]

        # Convrt the variables into vectors
        confidence_level = confidence_level_percentages/ 100


        Interval_Variable, Nominlal_Variable = [j + 1 for i in range(Contingency_Table.shape[0]) for j in range(Contingency_Table.shape[1]) for _ in range(Contingency_Table[i, j])], [i + 1 for i in range(Contingency_Table.shape[0]) for j in range(Contingency_Table.shape[1]) for _ in range(Contingency_Table[i, j])]

        # if the Nominal variable is not Numeric convert it to a numeric one
        

        point_biserial_correlation_output = point_biserial_correlation(np.array(Nominlal_Variable), np.array(Interval_Variable), confidence_level) if len(np.unique(Nominlal_Variable))==2 else "Point Biserial Correlation only relvent when the Nominal Variable has 2 levels"
        eta_output = Eta_Correlation_Ratio(np.array(Nominlal_Variable),np.array(Interval_Variable),confidence_level)

        results = {}
        results["eta_output "] = eta_output 
        results["_______________________"] = ""
        results["Point Biserial Correlation"] = point_biserial_correlation_output

        return results


