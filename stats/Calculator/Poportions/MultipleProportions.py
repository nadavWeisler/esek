
## Multiple Proportions


# Mantel-Haenszel Test 

import pandas as pd
import numpy as np
from scipy.stats import chi2
from statsmodels.stats.contingency_tables import cochrans_q, StratifiedTable # BSD 3-Clause "New" or "Revised" License
 
#Cochran todo = it only works with 1 and 0 so change the first value in data to sucess (1) and the second failure (0)


# The data should be a organized as a data frame in a wide format in which each column represents a different variable with 2 levels (success and failure) and each row represents a different subject.
def Cochran_Q_based_Effect_Size(Final_Data):
    sample_size = len(Final_Data)

    Variables_Number = Final_Data.shape[1]
    Degrees_Of_Freedom = Variables_Number - 1
    row_sums = np.sum(Final_Data, axis=1)
    Pis = ((1/Variables_Number)*row_sums)
    A = sum(Pis)
    B = sample_size - A
    C = Degrees_Of_Freedom / (2*sum(Pis * (1-Pis)))

    # Display the transposed DataFrame
    Q = cochrans_q(Final_Data).statistic
    pval = cochrans_q(Final_Data).pvalue
    VarianceQ = (Q/C - 2*A*B) / (sample_size *(sample_size-1))
    MeanQ = (2 / (sample_size*(sample_size-1))) * (A * B - (sum(Pis * (1-Pis))))
    Effect_Size_R = 1 - VarianceQ/MeanQ

    results = {}
    
    results["Cochrans Q"] = round(Q, 7)
    results["Degrees of Freedom"] = round(Degrees_Of_Freedom)
    results["Chocran's Q p-value"] = round(pval, 7)
    results["Variance of Q"] = round(VarianceQ)
    results["Mean Q"] = round(MeanQ, 7)
    results["Chance Corrected Q-based Effect Size (Berry et al., 2010)"] = round(Effect_Size_R, 7)

    return results

def goodness_of_fit_from_frequency(column_1, expected_proportions=None, expected_frequencies=None, expected_ratios=None, confidence_level=None):
    data_series = pd.Series(column_1)
    Observed = data_series.value_counts()
    sample_size = sum(Observed)
    Final_Data = pd.DataFrame({'level name': Observed.index, 'frequency': Observed.values})
    levels_number = Final_Data.shape[0]


    if expected_proportions is not None:
        Expected = np.array(expected_proportions) * sample_size
    elif expected_frequencies is not None:
        Expected = np.array(expected_frequencies)
    elif expected_ratios is not None:
        ratio_sum = sum(expected_ratios)
        Expected = (np.array(expected_ratios) / ratio_sum) * sample_size
    else:
        Expected = np.array([(1 / levels_number)] * levels_number) * sample_size

    degrees_of_freedom = levels_number - 1
    
    # Pearson Chi Square Test
    Chi_square = sum((Observed - Expected)**2 / Expected)
    p_value_chi_square =  chi2.sf((abs(Chi_square)), degrees_of_freedom)
    
    # Wilks_G_Square Test
    Observed_Proportions = Observed / sample_size
    Expected_Proportions = Expected / sample_size

    Wilks_G_Square = 2 *sample_size * (sum(Observed_Proportions * np.log(Observed_Proportions/Expected_Proportions)))


    # Effect Sizes

    # Cohens W
    Cohens_w = np.sqrt(Chi_square / sample_size)

    # maximum-corrected pearson chi_square
    q_chi = min(Expected)
    max_chi_Square = (sample_size*(sample_size-q_chi))/q_chi
    max_corrected_lambda = Chi_square/max_chi_Square
    
    # maximum-corrected wilks g square
    q_wilks = min(Expected_Proportions)
    max_wilks_G = -2*sample_size*np.log(q_wilks)
    max_corrected_gamma = Wilks_G_Square / max_wilks_G

    # Chance_Corrected measure of effect size
    Variance = (1/ levels_number) * (np.sum((np.array(Observed_Proportions) - (Expected_Proportions))**2))
    Mean = (1/levels_number**2) * (np.sum(([(elem1 - elem2)**2 for elem1 in np.array(Observed_Proportions) for elem2 in Expected_Proportions])))
    Chance_Corrected_R = Variance / Mean

    results = {}

    results["Chi Square"] = round(Chi_square, 7)
    results["Degrees of Freedom"] = round(degrees_of_freedom, 7)
    results["p value Chi Square"] = np.around(p_value_chi_square)
    results["Wilks' G Square"] = round(Wilks_G_Square, 7)
    results["max_corrected_lambda"] = round(max_corrected_lambda,4)
    results["max_corrected_gamma"] = np.around(max_corrected_gamma,4)
    results["Chance_Corrected_R"] = Chance_Corrected_R
    results["Variance of R"] = np.around(Variance,4)
    results["Mean of R"] = Mean

    return results


# 1. MH odds ratio from data and from tables (Allow both)
# 2. Test and Compare MH odds ratio of others


