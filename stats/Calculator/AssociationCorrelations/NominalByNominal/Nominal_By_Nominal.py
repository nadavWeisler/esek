############################################################
##### Effect Size for Nominal By Nominal Correlation #######
############################################################
from itertools import product
import random
import numpy as np
from scipy.stats import norm, ncx2, chi2, chi2_contingency
import pandas as pd

# Relevant Functions for Nominal by Nominal Correlation
#######################################################

# 1. Non Central (Pivotal) CI
def ncp_ci(chival, df, conf):
    """
    Calculate the non-central confidence interval for a given chi-square value, degrees of freedom, and confidence level.

    Parameters:
    chival (float): Chi-square value.
    df (int): Degrees of freedom.
    conf (float): Confidence level.

    Returns:
    tuple: Lower and upper bounds of the non-central confidence interval.
    """
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


# 2. Maximum Corrected Chi Square by Berry & Mialke
def Berry_Mielke_Maximum_Corrected_Cramer_V_output_matrix(matrix):
    """
    Calculate the maximum corrected Cramer's V output matrix using the Berry & Mialke method.

    Parameters:
    matrix (numpy.ndarray): Contingency table.

    Returns:
    numpy.ndarray: Maximum corrected Cramer's V output matrix.
    """
    Observed_Chi_Square = chi2_contingency(matrix)[0]
    r, c = matrix.shape 
    sum_of_rows_vector = matrix.sum(axis=1)  
    sum_of_cols_vector = matrix.sum(axis=0)  
    NR = sum_of_rows_vector.sum()  
    NC = sum_of_cols_vector.sum() 

    matrix = np.zeros((r, c)) 
    x = np.where(np.isin(sum_of_cols_vector, sum_of_rows_vector), np.argmax(np.isin(sum_of_rows_vector, sum_of_cols_vector), axis=0) + 1, np.nan)
    y = np.where(np.isin(sum_of_rows_vector, sum_of_cols_vector), np.argmax(np.isin(sum_of_cols_vector, sum_of_rows_vector), axis=0) + 1, np.nan)
    x = x[~np.isnan(x)].astype(int) - 1  
    y = y[~np.isnan(y)].astype(int) - 1  

    matrix[x,y] = sum_of_rows_vector[x]
    sum_of_rows_vector[x] = sum_of_rows_vector[x]-sum_of_rows_vector[x] 
    sum_of_cols_vector[y] = sum_of_cols_vector[y]-sum_of_cols_vector[y] 

    while NR > 0 and NC > 0:
        NR = np.sum(sum_of_rows_vector)  
        NC = np.sum(sum_of_cols_vector)  
        x = np.argmax(sum_of_rows_vector)  
        y = np.argmax(sum_of_cols_vector)  
        z = min(sum_of_rows_vector[x], sum_of_cols_vector[y]) 
        matrix[x, y] = z
        sum_of_rows_vector[x] -= z
        sum_of_cols_vector[y] -= z

    Maximum_Corrected_Cramers_v = Observed_Chi_Square

    return matrix


# 3. Goodman and Krushkal Tau
def goodman_kruskal_lamda_correlation(matrix, confidence_level):
    """
    Calculate the Goodman and Krushkal Lambda correlation for a given contingency table.

    Parameters:
    matrix (numpy.ndarray): Contingency table.
    confidence_level (float): Confidence level.

    Returns:
    str: Results of the Goodman and Krushkal Lambda correlation.
    """
    # Calcluation of the Matrix follows Hartwig, 1976
    #################################################
    
    # Please Note that SPSS and DescTools have a different method for calculating the SRK in the symmetric method - they are both yields the same results 
    # Marginal total Frequencies
    csum = np.sum(matrix, axis=0)
    rsum = np.sum(matrix, axis=1)
    
    n = np.sum(matrix) # Sample Size
    Nrc = np.sum(np.max(matrix, axis=1)) # Sum of the largest values in each Row
    Nkc = np.sum(np.max(matrix, axis=0)) # Sum of the largest values in each Coloumn
    Nrm = np.max(rsum) # This is the largest value in rows
    Nkm = np.max(csum) # This is the largest value in coloumns
    Um = Nrm + Nkm #For later calculations shortcuts
    Uc = Nrc + Nkc #for later calculations shortcuts
    

    # Lambda Effect Size Values
    Lambda_row = (Nrc - Nkm) / (n - Nkm)
    Lambda_col = (Nkc - Nrm) / (n - Nrm)
    Lambda_symmetric = (Nrc + Nkc - Nrm - Nkm) / (2 * n - Nrm - Nkm)


    # Calculation of Nr_tag and Nk_tag - In case of duplicates one need to store all the combinations of Nr_tag and Nktag

    # Find row(s) with the largest row sum and return a vector for the Nr_tag
    row_sums = np.sum(matrix, axis=1)
    rows_with_largest_rsum = np.where(row_sums == Nrm)[0]
    largest_row_values_vector = [] # Create a vector to store the largest values from the rows with the largest rsum

    # Iterate through the rows with the largest rsum and find the largest value in each row
    for row_idx in rows_with_largest_rsum:
        largest_value_in_row = np.max(matrix[row_idx, :])
        largest_row_values_vector.append(largest_value_in_row)
    largest_rows_vector = np.array(largest_row_values_vector)

    # Find row(s) with the largest coloumn sum and return a vector for the Nk_tag
    column_sums = np.sum(matrix, axis=0)
    columns_with_largest_csum = np.where(column_sums == Nkm)[0]
    largest_col_values_vector = []

    # Iterate through the columns with the largest csum and find the largest value in each column
    for col_idx in columns_with_largest_csum:
        largest_value_in_column = np.max(matrix[:, col_idx])
        largest_col_values_vector.append(largest_value_in_column)
    largest_col_vector = np.array(largest_col_values_vector)

    #Now calculate all the possible combination of Nks_tag and Nr_tag and return them as seperate vectors
    combinations_nks_nrs = list(product(largest_rows_vector, largest_col_vector))
    Nk_tag = [item[0] for item in combinations_nks_nrs]
    Nr_tag = [item[1] for item in combinations_nks_nrs]

    # Calculation of Ntag that can also have multiple values in case of marginal totals frequencies ties
    #Locate the row
    rows_with_most_frequencies = np.where(rsum == Nrm)[0]  
    cols_with_most_frequencies = np.where(csum == Nkm)[0]

    N_tags_combinations = list(product(rows_with_most_frequencies, cols_with_most_frequencies))
    N_tags = []

    #Ntags values - test which rows\coloumns in the table have 
    for row_idx, col_idx in N_tags_combinations:
        N_tags.append(matrix[row_idx, col_idx])

    #Skcr and Srck - we need to search for the rows and coloums with the largest marginal rows and 
    
    #Skcr
    # Create a vector to store the largest values from the rows with the largest rsum
    row_sums = np.sum(matrix, axis=1)
    rows_with_largest_rsum = np.where(row_sums == Nrm)[0]
    largest_row_values_vector = []  # Create a vector to store the largest values from the rows with the largest rsum

    # Iterate through the rows with the largest rsum and find the largest value in each row
    for row_idx in rows_with_largest_rsum:
        largest_value_in_row = np.max(matrix[row_idx, :])
        largest_row_values_vector.append(largest_value_in_row)

    # Sum the values in each row that are the maximum in their respective columns
    sum_of_highest_values_rows = []
    for row_idx in rows_with_largest_rsum:
        max_values_in_cols = [matrix[row_idx, col_idx] for col_idx in range(matrix.shape[1]) if matrix[row_idx, col_idx] == np.max(matrix[:, col_idx])]
        row_sum_of_highest_values = sum(max_values_in_cols)
        sum_of_highest_values_rows.append(row_sum_of_highest_values)

    sum_of_highest_values_row_vector = np.array(sum_of_highest_values_rows)

    #Srck 
    col_sums = np.sum(matrix, axis=0)
    cols_with_largest_csum = np.where(col_sums == Nkm)[0]
    largest_col_values_vector = []  # Create a vector to store the largest values from the rows with the largest rsum

    # Iterate through the rows with the largest rsum and find the largest value in each row
    for col_idx in cols_with_largest_csum:
        largest_value_in_col = np.max(matrix[:, col_idx])
        largest_col_values_vector.append(largest_value_in_col)

    # Sum the values in each col that are the maximum in their respective row
    sum_of_highest_values_col = []
    for col_idx in cols_with_largest_csum:
        max_values_in_rows = [matrix[row_idx, col_idx] for row_idx in range(matrix.shape[0]) if matrix[row_idx,col_idx] == np.max(matrix[row_idx,: ])]
        col_sum_of_highest_values = sum(max_values_in_rows)
        sum_of_highest_values_col.append(col_sum_of_highest_values)

    sum_of_highest_values_col_vector = np.array(sum_of_highest_values_col)

    #Now calculate all the possible combination of Nks_tag and Nr_tag and return them as a seperate vectors
    combinations_Skcr_Srck = list(product(sum_of_highest_values_row_vector, sum_of_highest_values_col_vector))
    Skcr = [item[0] for item in combinations_Skcr_Srck] #This is basically the sum of all largest values in each coloumn seperatly for each row
    Srck = [item[1] for item in combinations_Skcr_Srck] #This is basically the sum of all largest values in each row seperatly for each coloumn

    #Calculate Srk - sum of all  largest values in both their row and coloumn
    largest_both_values = []
    
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if value == np.max(matrix[row_idx, :]) and value == np.max(matrix[:, col_idx]):
                largest_both_values.append(value)
    
    Srk = np.sum(largest_both_values)

    #Now the Utag for shortening calcualtions
    Utag = []
    for i in range(len(Skcr)):
        sum_value = Skcr[i] + Srck[i] + Nr_tag[i] + Nk_tag[i]
        Utag.append(sum_value)


    # Final Parameters - Note that SAS has a wrong calcualation

    #Standard Errors
    Standard_Error_Rows = []
    for srck_value in Srck:
        standard_error_row = 0 if Lambda_row == 0 else 1 / np.sqrt(((n - Nkm) ** 3) / ((n - Nrc) * (Nrc + Nkm - 2 * srck_value)))
        Standard_Error_Rows.append(standard_error_row)

    # Calculate Standard Errors for Columns
    Standard_Error_Coloumns = []
    for skcr_value in Skcr:
        standard_error_col = 0 if Lambda_col == 0 else 1 / np.sqrt(((n - Nrm) ** 3) / ((n - Nkc) * (Nkc + Nrm - 2 * skcr_value)))
        Standard_Error_Coloumns.append(standard_error_col)

    # Calculate Symmetric Standard Errors 
    Standard_Error_Symmetric = []
    for Utag_value, Ntags_Value in zip(Utag, N_tags):
        standard_error_sym  = 0 if Lambda_symmetric == 0 else 1/ ((2*n - Um)**4 / ((2*n - Um) * (2*n - Uc) * (Um + Uc + 4*n - 2 *Utag_value) - 2*(2*n - Um)**2 * (n-Srk) - 2*(2*n - Uc)**2 * (n-Ntags_Value)))**0.5
        Standard_Error_Symmetric.append(standard_error_sym)

    #Final Parameters for Output

    # Standard Errors Methods CI's and inferential statistics
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))

    # 1. Random Method - other software use the first values in matrix - here is the random method
    Standard_Error_Rows_random = random.choice(Standard_Error_Rows)
    Standard_Error_Coloumns_random = random.choice(Standard_Error_Coloumns)
    Standard_Error_Symmetric_random = random.choice(Standard_Error_Symmetric)
    Z_row_rand = 0 if Lambda_row == 0 else Lambda_row / Standard_Error_Rows_random
    Z_col_rand =  0 if Lambda_col == 0 else Lambda_col / Standard_Error_Coloumns_random
    Z_symmetric_rand =  0 if Lambda_symmetric == 0 else Lambda_symmetric / Standard_Error_Symmetric_random
    pval_row_rand = norm.cdf(abs(Z_row_rand))
    pval_col_rand = norm.cdf(abs(Z_col_rand))
    pval_symmetric_rand = norm.cdf(abs(Z_symmetric_rand))
    lower_ci_lambda_rows_rand = max(Lambda_row - z_critical_value * Standard_Error_Rows_random,0)
    upper_ci_lambda_rows_rand = min(Lambda_row + z_critical_value * Standard_Error_Rows_random,1)
    lower_ci_lambda_cols_rand = max(Lambda_col - z_critical_value * Standard_Error_Coloumns_random,0)
    upper_ci_lambda_cols_rand = min(Lambda_col + z_critical_value * Standard_Error_Coloumns_random,1)
    lower_ci_lambda_symmetric_rand = max(Lambda_symmetric - z_critical_value * Standard_Error_Symmetric_random,0)
    upper_ci_lambda_symmetric_rand = min(Lambda_symmetric + z_critical_value * Standard_Error_Symmetric_random,1)
    
    # 2. Maximum Method - Strict Method - Uses the Largest value
    Standard_Error_Rows_max = np.max(Standard_Error_Rows)
    Standard_Error_Coloumns_max = np.max(Standard_Error_Coloumns)
    Standard_Error_Symmetric_max = np.max(Standard_Error_Symmetric)
    Z_row_max = 0 if Lambda_row == 0 else Lambda_row / Standard_Error_Rows_max
    Z_col_max = 0 if Lambda_col == 0 else Lambda_col / Standard_Error_Coloumns_max
    Z_symmetric_max = 0 if Lambda_symmetric == 0 else Lambda_symmetric / Standard_Error_Symmetric_max
    pval_row_max = norm.cdf(abs(Z_row_max))
    pval_col_max = norm.cdf(abs(Z_col_max))
    pval_symmetric_max = norm.cdf(abs(Z_symmetric_max))
    lower_ci_lambda_rows_max = max(Lambda_row - z_critical_value * Standard_Error_Rows_max,0)
    upper_ci_lambda_rows_max = min(Lambda_row + z_critical_value * Standard_Error_Rows_max,1)
    lower_ci_lambda_cols_max = max(Lambda_col - z_critical_value * Standard_Error_Coloumns_max,0)
    upper_ci_lambda_cols_max = min(Lambda_col + z_critical_value * Standard_Error_Coloumns_max,1)
    lower_ci_lambda_symmetric_max = max(Lambda_symmetric - z_critical_value * Standard_Error_Symmetric_max,0)
    upper_ci_lambda_symmetric_max = min(Lambda_symmetric + z_critical_value * Standard_Error_Symmetric_max,1)

    # 3. Mean Method - Average of all standard Error
    Standard_Error_Rows_mean = np.mean(Standard_Error_Rows)
    Standard_Error_Coloumns_mean = np.mean(Standard_Error_Coloumns)
    Standard_Error_Symmetric_mean = np.mean(Standard_Error_Symmetric)
    Z_row_mean = 0 if Lambda_row == 0 else Lambda_row / Standard_Error_Rows_mean
    Z_col_mean = 0 if Lambda_col == 0 else Lambda_col / Standard_Error_Coloumns_mean
    Z_symmetric_mean = 0 if Lambda_symmetric == 0 else Lambda_symmetric / Standard_Error_Symmetric_mean
    pval_row_mean = norm.cdf(abs(Z_row_mean))
    pval_col_mean = norm.cdf(abs(Z_col_mean))
    pval_symmetric_mean = norm.cdf(abs(Z_symmetric_mean))
    lower_ci_lambda_rows_mean = max(Lambda_row - z_critical_value * Standard_Error_Rows_mean,0) # type: ignore
    upper_ci_lambda_rows_mean = min(Lambda_row + z_critical_value * Standard_Error_Rows_mean,1) # type: ignore
    lower_ci_lambda_cols_mean = max(Lambda_col - z_critical_value * Standard_Error_Coloumns_mean,0) # type: ignore
    upper_ci_lambda_cols_mean = min(Lambda_col + z_critical_value * Standard_Error_Coloumns_mean,1) # type: ignore
    lower_ci_lambda_symmetric_mean = max(Lambda_symmetric - z_critical_value * Standard_Error_Symmetric_mean,0) # type: ignore
    upper_ci_lambda_symmetric_mean = min(Lambda_symmetric + z_critical_value * Standard_Error_Symmetric_mean,1) # type: ignore

    results = {}

    # Lambda Values
    results["Lambda Rows"] = round(Lambda_row, 7)
    results["Lambda Columns"] = round(Lambda_col, 7)
    results["Lambda Symmetric"] = round(Lambda_symmetric, 7)

    # Random Method
    results["Random Method - P-Value Rows"] = pval_row_rand
    results["Random Method - P-Value Columns"] = pval_col_rand
    results["Random Method - P-Value Symmetric"] = pval_symmetric_rand
    results["Random Method - Confidence Interval Rows"] = [round(lower_ci_lambda_rows_rand, 7), round(upper_ci_lambda_rows_rand, 7)]
    results["Random Method - Confidence Interval Columns"] = [round(lower_ci_lambda_cols_rand, 7), round(upper_ci_lambda_cols_rand, 7)]
    results["Random Method - Confidence Interval Symmetric"] = [round(lower_ci_lambda_symmetric_rand, 7), round(upper_ci_lambda_symmetric_rand, 4)]
    results["Random Method - Standard Error Rows"] = round(Standard_Error_Rows_random, 7)
    results["Random Method - Standard Error Columns"] = round(Standard_Error_Coloumns_random, 7)
    results["Random Method - Standard Error Symmetric"] = round(Standard_Error_Symmetric_random, 7)
    formatted_p_value_rand_rows = "{:.3f}".format(pval_row_rand).lstrip('0') if pval_row_rand >= 0.001 else "\033[3mp\033[0m < .001"
    formatted_p_value_rand_cols = "{:.3f}".format(pval_col_rand).lstrip('0') if pval_col_rand >= 0.001 else "\033[3mp\033[0m < .001"
    formatted_p_value_rand_sym = "{:.3f}".format(pval_symmetric_rand).lstrip('0') if pval_symmetric_rand >= 0.001 else "\033[3mp\033[0m < .001"
    results["Statistical Line Goodman Krushkal Lambda Rows Random Method"] = "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(Lambda_row, '\033[3mp = \033[0m' if pval_row_rand >= 0.001 else '', formatted_p_value_rand_rows, confidence_level * 100 ,round(lower_ci_lambda_rows_rand, 3),round(upper_ci_lambda_rows_rand,3))
    results["Statistical Line Goodman Krushkal Lambda Columns Random Method"] = "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(Lambda_col, '\033[3mp = \033[0m' if pval_col_rand >= 0.001 else '', formatted_p_value_rand_cols, confidence_level * 100 ,round(lower_ci_lambda_cols_rand, 3),round(upper_ci_lambda_cols_rand,3))
    results["Statistical Line Goodman Krushkal Lambda Symmetric Random Method"] = "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(Lambda_symmetric, '\033[3mp = \033[0m' if pval_row_rand >= 0.001 else '', formatted_p_value_rand_sym, confidence_level * 100 ,round(lower_ci_lambda_symmetric_rand, 3),round(upper_ci_lambda_symmetric_rand,3))


    # Maximum Method
    results["Maximum Method - P-Value Rows"] = pval_row_max
    results["Maximum Method - P-Value Columns"] = pval_col_max
    results["Maximum Method - P-Value Symmetric"] = pval_symmetric_max
    results["Maximum Method - Confidence Interval Rows"] = [round(lower_ci_lambda_rows_max, 7), round(upper_ci_lambda_rows_max, 7)]

############################################################
##### Effect Size for Nominal By Nominal Correlation #######
############################################################
from itertools import product
import random
import numpy as np
from scipy.stats import norm, ncx2, chi2, chi2_contingency
import pandas as pd

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


# 2. Maximum Corrected Chi Square by Berry & Mialke
def Berry_Mielke_Maximum_Corrected_Cramer_V_output_matrix(matrix):
    Observed_Chi_Square = chi2_contingency(matrix)[0]
    r, c = matrix.shape 
    sum_of_rows_vector = matrix.sum(axis=1)  
    sum_of_cols_vector = matrix.sum(axis=0)  
    NR = sum_of_rows_vector.sum()  
    NC = sum_of_cols_vector.sum() 

    matrix = np.zeros((r, c)) 
    x = np.where(np.isin(sum_of_cols_vector, sum_of_rows_vector), np.argmax(np.isin(sum_of_rows_vector, sum_of_cols_vector), axis=0) + 1, np.nan)
    y = np.where(np.isin(sum_of_rows_vector, sum_of_cols_vector), np.argmax(np.isin(sum_of_cols_vector, sum_of_rows_vector), axis=0) + 1, np.nan)
    x = x[~np.isnan(x)].astype(int) - 1  
    y = y[~np.isnan(y)].astype(int) - 1  

    matrix[x,y] = sum_of_rows_vector[x]
    sum_of_rows_vector[x] = sum_of_rows_vector[x]-sum_of_rows_vector[x] 
    sum_of_cols_vector[y] = sum_of_cols_vector[y]-sum_of_cols_vector[y] 

    while NR > 0 and NC > 0:
        NR = np.sum(sum_of_rows_vector)  
        NC = np.sum(sum_of_cols_vector)  
        x = np.argmax(sum_of_rows_vector)  
        y = np.argmax(sum_of_cols_vector)  
        z = min(sum_of_rows_vector[x], sum_of_cols_vector[y]) 
        matrix[x, y] = z
        sum_of_rows_vector[x] -= z
        sum_of_cols_vector[y] -= z

    Maximum_Corrected_Cramers_v = Observed_Chi_Square

    return matrix


# 3. Goodman and Krushkal Tau
def goodman_kruskal_lamda_correlation(matrix, confidence_level):
        
        # Calcluation of the Matrix follows Hartwig, 1976
        #################################################
        
        # Please Note that SPSS and DescTools have a different method for calculating the SRK in the symmetric method - they are both yields the same results 
        # Marginal total Frequencies
        csum = np.sum(matrix, axis=0)
        rsum = np.sum(matrix, axis=1)
        
        n = np.sum(matrix) # Sample Size
        Nrc = np.sum(np.max(matrix, axis=1)) # Sum of the largest values in each Row
        Nkc = np.sum(np.max(matrix, axis=0)) # Sum of the largest values in each Coloumn
        Nrm = np.max(rsum) # This is the largest value in rows
        Nkm = np.max(csum) # This is the largest value in coloumns
        Um = Nrm + Nkm #For later calculations shortcuts
        Uc = Nrc + Nkc #for later calculations shortcuts
        

        # Lambda Effect Size Values
        Lambda_row = (Nrc - Nkm) / (n - Nkm)
        Lambda_col = (Nkc - Nrm) / (n - Nrm)
        Lambda_symmetric = (Nrc + Nkc - Nrm - Nkm) / (2 * n - Nrm - Nkm)


        # Calculation of Nr_tag and Nk_tag - In case of duplicates one need to store all the combinations of Nr_tag and Nktag

        # Find row(s) with the largest row sum and return a vector for the Nr_tag
        row_sums = np.sum(matrix, axis=1)
        rows_with_largest_rsum = np.where(row_sums == Nrm)[0]
        largest_row_values_vector = [] # Create a vector to store the largest values from the rows with the largest rsum

        # Iterate through the rows with the largest rsum and find the largest value in each row
        for row_idx in rows_with_largest_rsum:
            largest_value_in_row = np.max(matrix[row_idx, :])
            largest_row_values_vector.append(largest_value_in_row)
        largest_rows_vector = np.array(largest_row_values_vector)

        # Find row(s) with the largest coloumn sum and return a vector for the Nk_tag
        column_sums = np.sum(matrix, axis=0)
        columns_with_largest_csum = np.where(column_sums == Nkm)[0]
        largest_col_values_vector = []

        # Iterate through the columns with the largest csum and find the largest value in each column
        for col_idx in columns_with_largest_csum:
            largest_value_in_column = np.max(matrix[:, col_idx])
            largest_col_values_vector.append(largest_value_in_column)
        largest_col_vector = np.array(largest_col_values_vector)

        #Now calculate all the possible combination of Nks_tag and Nr_tag and return them as seperate vectors
        combinations_nks_nrs = list(product(largest_rows_vector, largest_col_vector))
        Nk_tag = [item[0] for item in combinations_nks_nrs]
        Nr_tag = [item[1] for item in combinations_nks_nrs]

        # Calculation of Ntag that can also have multiple values in case of marginal totals frequencies ties
        #Locate the row
        rows_with_most_frequencies = np.where(rsum == Nrm)[0]  
        cols_with_most_frequencies = np.where(csum == Nkm)[0]

        N_tags_combinations = list(product(rows_with_most_frequencies, cols_with_most_frequencies))
        N_tags = []

        #Ntags values - test which rows\coloumns in the table have 
        for row_idx, col_idx in N_tags_combinations:
            N_tags.append(matrix[row_idx, col_idx])

        #Skcr and Srck - we need to search for the rows and coloums with the largest marginal rows and 
        
        #Skcr
        # Create a vector to store the largest values from the rows with the largest rsum
        row_sums = np.sum(matrix, axis=1)
        rows_with_largest_rsum = np.where(row_sums == Nrm)[0]
        largest_row_values_vector = []  # Create a vector to store the largest values from the rows with the largest rsum

        # Iterate through the rows with the largest rsum and find the largest value in each row
        for row_idx in rows_with_largest_rsum:
            largest_value_in_row = np.max(matrix[row_idx, :])
            largest_row_values_vector.append(largest_value_in_row)

        # Sum the values in each row that are the maximum in their respective columns
        sum_of_highest_values_rows = []
        for row_idx in rows_with_largest_rsum:
            max_values_in_cols = [matrix[row_idx, col_idx] for col_idx in range(matrix.shape[1]) if matrix[row_idx, col_idx] == np.max(matrix[:, col_idx])]
            row_sum_of_highest_values = sum(max_values_in_cols)
            sum_of_highest_values_rows.append(row_sum_of_highest_values)

        sum_of_highest_values_row_vector = np.array(sum_of_highest_values_rows)

        #Srck 
        col_sums = np.sum(matrix, axis=0)
        cols_with_largest_csum = np.where(col_sums == Nkm)[0]
        largest_col_values_vector = []  # Create a vector to store the largest values from the rows with the largest rsum

        # Iterate through the rows with the largest rsum and find the largest value in each row
        for col_idx in cols_with_largest_csum:
            largest_value_in_col = np.max(matrix[:, col_idx])
            largest_col_values_vector.append(largest_value_in_col)

        # Sum the values in each col that are the maximum in their respective row
        sum_of_highest_values_col = []
        for col_idx in cols_with_largest_csum:
            max_values_in_rows = [matrix[row_idx, col_idx] for row_idx in range(matrix.shape[0]) if matrix[row_idx,col_idx] == np.max(matrix[row_idx,: ])]
            col_sum_of_highest_values = sum(max_values_in_rows)
            sum_of_highest_values_col.append(col_sum_of_highest_values)

        sum_of_highest_values_col_vector = np.array(sum_of_highest_values_col)

        #Now calculate all the possible combination of Nks_tag and Nr_tag and return them as a seperate vectors
        combinations_Skcr_Srck = list(product(sum_of_highest_values_row_vector, sum_of_highest_values_col_vector))
        Skcr = [item[0] for item in combinations_Skcr_Srck] #This is basically the sum of all largest values in each coloumn seperatly for each row
        Srck = [item[1] for item in combinations_Skcr_Srck] #This is basically the sum of all largest values in each row seperatly for each coloumn

        #Calculate Srk - sum of all  largest values in both their row and coloumn
        largest_both_values = []
        
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if value == np.max(matrix[row_idx, :]) and value == np.max(matrix[:, col_idx]):
                    largest_both_values.append(value)
        
        Srk = np.sum(largest_both_values)

        #Now the Utag for shortening calcualtions
        Utag = []
        for i in range(len(Skcr)):
            sum_value = Skcr[i] + Srck[i] + Nr_tag[i] + Nk_tag[i]
            Utag.append(sum_value)


        # Final Parameters - Note that SAS has a wrong calcualation

    
        #Standard Errors
        Standard_Error_Rows = []
        for srck_value in Srck:
            standard_error_row = 0 if Lambda_row == 0 else 1 / np.sqrt(((n - Nkm) ** 3) / ((n - Nrc) * (Nrc + Nkm - 2 * srck_value)))
            Standard_Error_Rows.append(standard_error_row)

        # Calculate Standard Errors for Columns
        Standard_Error_Coloumns = []
        for skcr_value in Skcr:
            standard_error_col = 0 if Lambda_col == 0 else 1 / np.sqrt(((n - Nrm) ** 3) / ((n - Nkc) * (Nkc + Nrm - 2 * skcr_value)))
            Standard_Error_Coloumns.append(standard_error_col)
    
    # Calculate Symmetric Standard Errors 
        Standard_Error_Symmetric = []
        for Utag_value, Ntags_Value in zip(Utag, N_tags):
            standard_error_sym  = 0 if Lambda_symmetric == 0 else 1/ ((2*n - Um)**4 / ((2*n - Um) * (2*n - Uc) * (Um + Uc + 4*n - 2 *Utag_value) - 2*(2*n - Um)**2 * (n-Srk) - 2*(2*n - Uc)**2 * (n-Ntags_Value)))**0.5
            Standard_Error_Symmetric.append(standard_error_sym)

        #Final Parameters for Output
        

        # Standard Errors Methods CI's and inferential statistics
        z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))

        # 1. Random Method - other software use the first values in matrix - here is the random method
        Standard_Error_Rows_random = random.choice(Standard_Error_Rows)
        Standard_Error_Coloumns_random = random.choice(Standard_Error_Coloumns)
        Standard_Error_Symmetric_random = random.choice(Standard_Error_Symmetric)
        Z_row_rand = 0 if Lambda_row == 0 else Lambda_row / Standard_Error_Rows_random
        Z_col_rand =  0 if Lambda_col == 0 else Lambda_col / Standard_Error_Coloumns_random
        Z_symmetric_rand =  0 if Lambda_symmetric == 0 else Lambda_symmetric / Standard_Error_Symmetric_random
        pval_row_rand = norm.cdf(abs(Z_row_rand))
        pval_col_rand = norm.cdf(abs(Z_col_rand))
        pval_symmetric_rand = norm.cdf(abs(Z_symmetric_rand))
        lower_ci_lambda_rows_rand = max(Lambda_row - z_critical_value * Standard_Error_Rows_random,0)
        upper_ci_lambda_rows_rand = min(Lambda_row + z_critical_value * Standard_Error_Rows_random,1)
        lower_ci_lambda_cols_rand = max(Lambda_col - z_critical_value * Standard_Error_Coloumns_random,0)
        upper_ci_lambda_cols_rand = min(Lambda_col + z_critical_value * Standard_Error_Coloumns_random,1)
        lower_ci_lambda_symmetric_rand = max(Lambda_symmetric - z_critical_value * Standard_Error_Symmetric_random,0)
        upper_ci_lambda_symmetric_rand = min(Lambda_symmetric + z_critical_value * Standard_Error_Symmetric_random,1)
        
        # 2. Maximum Method - Strict Method - Uses the Largest value
        Standard_Error_Rows_max = np.max(Standard_Error_Rows)
        Standard_Error_Coloumns_max = np.max(Standard_Error_Coloumns)
        Standard_Error_Symmetric_max = np.max(Standard_Error_Symmetric)
        Z_row_max = 0 if Lambda_row == 0 else Lambda_row / Standard_Error_Rows_max
        Z_col_max = 0 if Lambda_col == 0 else Lambda_col / Standard_Error_Coloumns_max
        Z_symmetric_max = 0 if Lambda_symmetric == 0 else Lambda_symmetric / Standard_Error_Symmetric_max
        pval_row_max = norm.cdf(abs(Z_row_max))
        pval_col_max = norm.cdf(abs(Z_col_max))
        pval_symmetric_max = norm.cdf(abs(Z_symmetric_max))
        lower_ci_lambda_rows_max = max(Lambda_row - z_critical_value * Standard_Error_Rows_max,0)
        upper_ci_lambda_rows_max = min(Lambda_row + z_critical_value * Standard_Error_Rows_max,1)
        lower_ci_lambda_cols_max = max(Lambda_col - z_critical_value * Standard_Error_Coloumns_max,0)
        upper_ci_lambda_cols_max = min(Lambda_col + z_critical_value * Standard_Error_Coloumns_max,1)
        lower_ci_lambda_symmetric_max = max(Lambda_symmetric - z_critical_value * Standard_Error_Symmetric_max,0)
        upper_ci_lambda_symmetric_max = min(Lambda_symmetric + z_critical_value * Standard_Error_Symmetric_max,1)

        # 3. Mean Method - Average of all standard Error
        Standard_Error_Rows_mean = np.mean(Standard_Error_Rows)
        Standard_Error_Coloumns_mean = np.mean(Standard_Error_Coloumns)
        Standard_Error_Symmetric_mean = np.mean(Standard_Error_Symmetric)
        Z_row_mean = 0 if Lambda_row == 0 else Lambda_row / Standard_Error_Rows_mean
        Z_col_mean = 0 if Lambda_col == 0 else Lambda_col / Standard_Error_Coloumns_mean
        Z_symmetric_mean = 0 if Lambda_symmetric == 0 else Lambda_symmetric / Standard_Error_Symmetric_mean
        pval_row_mean = norm.cdf(abs(Z_row_mean))
        pval_col_mean = norm.cdf(abs(Z_col_mean))
        pval_symmetric_mean = norm.cdf(abs(Z_symmetric_mean))
        lower_ci_lambda_rows_mean = max(Lambda_row - z_critical_value * Standard_Error_Rows_mean,0) # type: ignore
        upper_ci_lambda_rows_mean = min(Lambda_row + z_critical_value * Standard_Error_Rows_mean,1) # type: ignore
        lower_ci_lambda_cols_mean = max(Lambda_col - z_critical_value * Standard_Error_Coloumns_mean,0) # type: ignore
        upper_ci_lambda_cols_mean = min(Lambda_col + z_critical_value * Standard_Error_Coloumns_mean,1) # type: ignore
        lower_ci_lambda_symmetric_mean = max(Lambda_symmetric - z_critical_value * Standard_Error_Symmetric_mean,0) # type: ignore
        upper_ci_lambda_symmetric_mean = min(Lambda_symmetric + z_critical_value * Standard_Error_Symmetric_mean,1) # type: ignore

        results = {}

        # Lambda Values
        results["Lambda Rows"] = round(Lambda_row, 7)
        results["Lambda Columns"] = round(Lambda_col, 7)
        results["Lambda Symmetric"] = round(Lambda_symmetric, 7)

        # Random Method
        results["Random Method - P-Value Rows"] = pval_row_rand
        results["Random Method - P-Value Columns"] = pval_col_rand
        results["Random Method - P-Value Symmetric"] = pval_symmetric_rand
        results["Random Method - Confidence Interval Rows"] = [round(lower_ci_lambda_rows_rand, 7), round(upper_ci_lambda_rows_rand, 7)]
        results["Random Method - Confidence Interval Columns"] = [round(lower_ci_lambda_cols_rand, 7), round(upper_ci_lambda_cols_rand, 7)]
        results["Random Method - Confidence Interval Symmetric"] = [round(lower_ci_lambda_symmetric_rand, 7), round(upper_ci_lambda_symmetric_rand, 4)]
        results["Random Method - Standard Error Rows"] = round(Standard_Error_Rows_random, 7)
        results["Random Method - Standard Error Columns"] = round(Standard_Error_Coloumns_random, 7)
        results["Random Method - Standard Error Symmetric"] = round(Standard_Error_Symmetric_random, 7)
        formatted_p_value_rand_rows = "{:.3f}".format(pval_row_rand).lstrip('0') if pval_row_rand >= 0.001 else "\033[3mp\033[0m < .001"
        formatted_p_value_rand_cols = "{:.3f}".format(pval_col_rand).lstrip('0') if pval_col_rand >= 0.001 else "\033[3mp\033[0m < .001"
        formatted_p_value_rand_sym = "{:.3f}".format(pval_symmetric_rand).lstrip('0') if pval_symmetric_rand >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Goodman Krushkal Lambda Rows Random Method"] = "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(Lambda_row, '\033[3mp = \033[0m' if pval_row_rand >= 0.001 else '', formatted_p_value_rand_rows, confidence_level * 100 ,round(lower_ci_lambda_rows_rand, 3),round(upper_ci_lambda_rows_rand,3))
        results["Statistical Line Goodman Krushkal Lambda Columns Random Method"] = "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(Lambda_col, '\033[3mp = \033[0m' if pval_col_rand >= 0.001 else '', formatted_p_value_rand_cols, confidence_level * 100 ,round(lower_ci_lambda_cols_rand, 3),round(upper_ci_lambda_cols_rand,3))
        results["Statistical Line Goodman Krushkal Lambda Symmetric Random Method"] = "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(Lambda_symmetric, '\033[3mp = \033[0m' if pval_row_rand >= 0.001 else '', formatted_p_value_rand_sym, confidence_level * 100 ,round(lower_ci_lambda_symmetric_rand, 3),round(upper_ci_lambda_symmetric_rand,3))


        # Maximum Method
        results["Maximum Method - P-Value Rows"] = pval_row_max
        results["Maximum Method - P-Value Columns"] = pval_col_max
        results["Maximum Method - P-Value Symmetric"] = pval_symmetric_max
        results["Maximum Method - Confidence Interval Rows"] = [round(lower_ci_lambda_rows_max, 7), round(upper_ci_lambda_rows_max, 7)]
        results["Maximum Method - Confidence Interval Columns"] = [round(lower_ci_lambda_cols_max, 7), round(upper_ci_lambda_cols_max, 7)]
        results["Maximum Method - Confidence Interval Symmetric"] = [round(lower_ci_lambda_symmetric_max, 7), round(upper_ci_lambda_symmetric_max, 4)]
        results["Maximum Method - Standard Error Rows"] = round(Standard_Error_Rows_max, 7)
        results["Maximum Method - Standard Error Columns"] = round(Standard_Error_Coloumns_max, 7)
        results["Maximum Method - Standard Error Symmetric"] = round(Standard_Error_Symmetric_max, 7)
        formatted_p_value_Max_rows = "{:.3f}".format(pval_row_max).lstrip('0') if pval_row_max >= 0.001 else "\033[3mp\033[0m < .001"
        formatted_p_value_Max_cols = "{:.3f}".format(pval_col_max).lstrip('0') if pval_col_max >= 0.001 else "\033[3mp\033[0m < .001"
        formatted_p_value_Max_sym = "{:.3f}".format(pval_symmetric_max).lstrip('0') if pval_symmetric_max >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Goodman Krushkal Lambda Rows Maximum Method"] = "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(Lambda_row, '\033[3mp = \033[0m' if pval_row_max >= 0.001 else '', formatted_p_value_Max_rows, confidence_level * 100 ,round(lower_ci_lambda_rows_max, 3),round(upper_ci_lambda_rows_max,3))
        results["Statistical Line Goodman Krushkal Lambda Cols Maximum Method"] = "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(Lambda_col, '\033[3mp = \033[0m' if pval_col_max >= 0.001 else '', formatted_p_value_Max_cols, confidence_level * 100 ,round(lower_ci_lambda_cols_max, 3),round(upper_ci_lambda_cols_max,3))
        results["Statistical Line Goodman Krushkal Lambda Symmetric Maximum Method"] = "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(Lambda_symmetric, '\033[3mp = \033[0m' if pval_symmetric_max >= 0.001 else '', formatted_p_value_Max_sym, confidence_level * 100 ,round(lower_ci_lambda_symmetric_max, 3),round(upper_ci_lambda_symmetric_max,3))

        # Mean Method
        results["Mean Method - P-Value Rows"] = pval_row_mean
        results["Mean Method - P-Value Columns"] = pval_col_mean
        results["Mean Method - P-Value Symmetric"] = pval_symmetric_mean
        results["Mean Method - Confidence Interval Rows"] = [round(lower_ci_lambda_rows_mean, 7), round(upper_ci_lambda_rows_mean, 7)]
        results["Mean Method - Confidence Interval Columns"] = [round(lower_ci_lambda_cols_mean, 7), round(upper_ci_lambda_cols_mean, 7)]
        results["Mean Method - Confidence Interval Symmetric"] = [round(lower_ci_lambda_symmetric_mean, 7), round(upper_ci_lambda_symmetric_mean, 7)]
        results["Mean Method - Standard Error Rows"] = round(Standard_Error_Rows_mean, 7)
        results["Mean Method - Standard Error Columns"] = round(Standard_Error_Coloumns_mean, 7)
        results["Mean Method - Standard Error Symmetric"] = round(Standard_Error_Symmetric_mean, 7)
        formatted_p_value_mean_rows = "{:.3f}".format(pval_row_mean).lstrip('0') if pval_row_mean >= 0.001 else "\033[3mp\033[0m < .001"
        formatted_p_value_mean_cols = "{:.3f}".format(pval_col_mean).lstrip('0') if pval_col_mean >= 0.001 else "\033[3mp\033[0m < .001"
        formatted_p_value_mean_sym = "{:.3f}".format(pval_symmetric_mean).lstrip('0') if pval_symmetric_mean >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Goodman Krushkal Lambda Rows Mean Method"] = "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(Lambda_row, '\033[3mp = \033[0m' if pval_row_mean >= 0.001 else '', formatted_p_value_mean_rows, confidence_level * 100 ,round(lower_ci_lambda_rows_mean, 3),round(upper_ci_lambda_rows_mean,3))
        results["Statistical Line Goodman Krushkal Lambda Cols Mean Method"] = "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(Lambda_col, '\033[3mp = \033[0m' if pval_col_mean >= 0.001 else '', formatted_p_value_mean_cols, confidence_level * 100 ,round(lower_ci_lambda_cols_mean, 3),round(upper_ci_lambda_cols_mean,3))
        results["Statistical Line Goodman Krushkal Lambda Symmetric Mean Method"] = "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(Lambda_symmetric, '\033[3mp = \033[0m' if pval_symmetric_mean >= 0.001 else '', formatted_p_value_mean_sym, confidence_level * 100 ,round(lower_ci_lambda_symmetric_mean, 3),round(upper_ci_lambda_symmetric_mean,3))

        result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
        return result_str

###4. Goodman and Kruskal Tau (Data (x variable) needs to be in a matrix that rperesents contingency Table)

def Goodman_Kruskal_Tau(x, confidence_level):
    #This function uses the Method presented in Libertau 1983 to calcualte the Standard Error

    # Global Variables
    sample_size = np.sum(x)
    Sum_Of_The_Rows = np.sum(x, axis=1)
    Sum_Of_The_Columns = np.sum(x, axis=0)
    Conditional_Errors_Coloumns = sample_size ** 2 - np.sum(Sum_Of_The_Columns ** 2)
    Conditional_Errors_Rows = sample_size ** 2 - np.sum(Sum_Of_The_Rows ** 2)
    Mean_Rows = Conditional_Errors_Rows / (sample_size ** 2)
    Mean_Coloumns = Conditional_Errors_Coloumns / (sample_size ** 2)

    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)

    # Calculate Tau_r for the Rows
    Unconditonal_Error_Rows = (sample_size ** 2) - sample_size * np.sum((x[:, np.newaxis] ** 2) / Sum_Of_The_Columns[np.newaxis])
    Tau_Rows = 1 - (Unconditonal_Error_Rows / Conditional_Errors_Rows)
    v = Unconditonal_Error_Rows / (sample_size ** 2)
    ASE_Rows = np.sqrt(np.sum((x * (-2 * v * (Sum_Of_The_Rows[:, np.newaxis] / sample_size) + Mean_Rows * ((2 * x / Sum_Of_The_Columns) - np.sum((x / Sum_Of_The_Columns) ** 2, axis=0)) - (Mean_Rows * (v + 1) - 2 * v)) ** 2) / (sample_size ** 2 * Mean_Rows ** 4)))
    Confidence_Intervals_rows_lower = max(Tau_Rows - zcrit * ASE_Rows, 0)
    Confidence_Intervals_rows_upper = min(Tau_Rows + zcrit * ASE_Rows, 1)
    Z_Statistic_rows = Tau_Rows / ASE_Rows
    p_value_rows = norm.sf(Z_Statistic_rows)


    # Calculate Tau_c for the Coloumns
    Unconditional_Error_Coloumns = (sample_size ** 2) - sample_size * np.sum((x ** 2) / Sum_Of_The_Rows[:, np.newaxis])
    Tau_Coloumns = 1 - (Unconditional_Error_Coloumns / Conditional_Errors_Coloumns)
    v2 = Unconditional_Error_Coloumns / (sample_size ** 2)

    ASE_Columns = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            term = x[i, j] * (-2 * v2 * (Sum_Of_The_Columns[j] / sample_size) + Mean_Coloumns * ((2 * x[i, j] / Sum_Of_The_Rows[i]) - np.sum((x[i, :] / Sum_Of_The_Rows[i]) ** 2)) - (Mean_Coloumns * (v2 + 1) - 2 * v2)) ** 2 / (sample_size ** 2 * Mean_Coloumns ** 4)
            ASE_Columns = ASE_Columns + term
    ASE_Columns = np.sqrt(ASE_Columns)
    Confidence_Intervals_Columns_lower = max(Tau_Coloumns - zcrit * ASE_Columns, 0)
    Confidence_Intervals_Columns_upper = min(Tau_Coloumns + zcrit * ASE_Columns, 1)
    Z_Statistic_cols = Tau_Coloumns / ASE_Columns
    p_value_cols = norm.sf(Z_Statistic_cols)


    # Symmetric Tau
    alpha = ((sample_size ** 2) - np.sum(Sum_Of_The_Rows**2)) / ((2* (sample_size ** 2)) - np.sum(Sum_Of_The_Rows**2) - np.sum(Sum_Of_The_Columns**2))
    Tau_Symmetric = (Tau_Rows*alpha) + (1-alpha)*Tau_Coloumns
    ASE_Symmetric = (ASE_Rows*alpha) + (1-alpha)*ASE_Columns
    Confidence_Intervals_Symmetric_lower = max(Tau_Symmetric - zcrit * ASE_Symmetric, 0)
    Confidence_Intervals_Symmetric_upper = min(Tau_Symmetric + zcrit * ASE_Symmetric, 1)
    Z_Statistic_symmetric= Tau_Symmetric / ASE_Symmetric
    p_value_symmetric = norm.sf(Z_Statistic_symmetric)

    results = {}

    results["Goodman Kruskal Tau (Rows)"]= Tau_Rows
    results["Goodman Kruskal Tau (Columns)"]= Tau_Coloumns
    results["Goodman Kruskal Tau (Symmetric)"]= Tau_Symmetric
    results["Standard Error Rows"] = ASE_Rows
    results["Standard Error Coloumns"]= ASE_Columns
    results["Standard Error Symmetric"]= ASE_Symmetric
    results["Statistic Goodman Kruskal Tau (Rows)"]= Z_Statistic_rows
    results["Statistic Goodman Kruskal Tau (Columns)"]= Z_Statistic_cols
    results["Statistic Goodman Kruskal Tau (Symmetric)"]= Z_Statistic_symmetric
    results["p-value Goodman Kruskal Tau (Rows)"]= p_value_rows
    results["p-value Goodman Kruskal Tau (Columns)"]= p_value_cols
    results["p-value Goodman Kruskal Tau (Symmetric)"]= p_value_symmetric
    results["Goodman-Kruskal CI's (Rows)"] = [Confidence_Intervals_rows_lower, Confidence_Intervals_rows_upper]
    results["Goodman-Kruskal CI's (Coloumns)"]= [Confidence_Intervals_Columns_lower, Confidence_Intervals_Columns_upper]
    results["Goodman-Kruskal CI's (Symmetric)"]= np.around([Confidence_Intervals_Symmetric_lower, Confidence_Intervals_Symmetric_upper], 4)
    formatted_p_value_rows = "{:.3f}".format(p_value_rows).lstrip('0') if p_value_rows >= 0.001 else "\033[3mp\033[0m < .001"
    formatted_p_value_cols = "{:.3f}".format(p_value_cols).lstrip('0') if p_value_cols >= 0.001 else "\033[3mp\033[0m < .001"
    formatted_p_value_sym = "{:.3f}".format(p_value_symmetric).lstrip('0') if p_value_symmetric >= 0.001 else "\033[3mp\033[0m < .001"
    results["Statistical Line Goodman Krushkal Tau Rows"] = "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(Tau_Rows, '\033[3mp = \033[0m' if p_value_rows >= 0.001 else '', formatted_p_value_rows, confidence_level * 100 ,round(Confidence_Intervals_rows_lower, 3),round(Confidence_Intervals_rows_upper,3))
    results["Statistical Line Goodman Krushkal Tau Columns"] = "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(Tau_Coloumns, '\033[3mp = \033[0m' if p_value_cols >= 0.001 else '', formatted_p_value_cols, confidence_level * 100 ,round(Confidence_Intervals_Columns_lower, 3),round(Confidence_Intervals_Columns_upper,3))
    results["Statistical Line Goodman Krushkal Tau Symmetric"] = "\033[3m\u03BB\033[0m = {:.3f}, {}{}, {}% CI [{:.3f}, {:.3f}]".format(Tau_Symmetric, '\033[3mp = \033[0m' if p_value_symmetric >= 0.001 else '', formatted_p_value_sym, confidence_level * 100 ,round(Confidence_Intervals_Symmetric_lower, 3),round(Confidence_Intervals_Symmetric_upper,3))

    result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
    return result_str


def Columns_to_Contingency(x, y):
    contingency_table = None  # Initialize contingency_table outside the try block
    try:
        # Check if x and y have the same length
        if len(x) != len(y):
            raise ValueError("Number of elements in x and y must be equal.")

        # Filter out empty cells from both x and y using the same indices
        non_empty_indices = np.where((x != "") & (y != ""))
        x = x[non_empty_indices]
        y = y[non_empty_indices]

        # Check if the sample sizes of x and y are equal
        if len(x) != len(y):
            raise ValueError("Sample sizes of x and y must be equal.")

        # Convert categorical variables to numerical values
        x_categories, x_numeric = np.unique(x, return_inverse=True)
        y_categories, y_numeric = np.unique(y, return_inverse=True)

        # Create a contingency table
        contingency_table = np.zeros((len(x_categories), len(y_categories)), dtype=int)

        for i in range(len(x)):
            contingency_table[x_numeric[i], y_numeric[i]] += 1

        # Create a DataFrame with the contingency table
        df = pd.DataFrame(contingency_table, index=x_categories, columns=y_categories)
        
        # Print the contingency table
        print(contingency_table)

    except ValueError as e:
        print(e)

    return contingency_table

    

# 6. Multilevel Tbales Nominal by Nominal Function
def multilevel_contingency_tables(contingency_table, confidence_level):

    # Preperations:
    Sample_Size = np.sum(contingency_table)
    Number_of_rows = np.size(contingency_table, axis = 0)
    Number_of_Coloumns = np.size(contingency_table, axis = 1)
    matrix = np.array(contingency_table)

    # Inferential Statistics
    # A. Calcualte Chi Square and its significance
    row_totals = np.sum(contingency_table, axis=1)
    col_totals = np.sum(contingency_table, axis=0)
    expected = np.multiply.outer(row_totals, col_totals) / Sample_Size
    chi_squared = np.sum( ((contingency_table - expected)**2) / expected)
    degrees_of_freedom_chi_square = (Number_of_Coloumns-1) * (Number_of_rows-1)
    p_value = chi2.sf(((chi_squared)), degrees_of_freedom_chi_square)

    # B. Calcualte Likelihood Ratio and its significance
    Sum_Product = np.sum(contingency_table)
    Expected = np.outer(np.sum(contingency_table, axis=1), np.sum(contingency_table, axis=0)) / Sum_Product

    likelihood_ratio = 0
    for i in range(2):
        for j in range(2):
            if contingency_table[i, j] != 0 and Expected[i, j] != 0:
                likelihood_ratio += contingency_table[i, j] * np.log(contingency_table[i, j] / Expected[i, j])

    likelihood_ratio *= 2
    likelihood_ratio_p_value = 1 - chi2.cdf(likelihood_ratio, degrees_of_freedom_chi_square)


    # Effect Sizes for R X C tables
    phi_square = chi_squared/Sample_Size 
    phi = np.sqrt(phi_square) # This is also know as Cohens w
    Cramer_V = (phi_square / (min(Number_of_Coloumns-1, Number_of_rows-1)))**0.5
    Tschuprows_T = (phi_square / ( (Number_of_Coloumns-1) * (Number_of_rows-1))**0.5)**0.5
    Pearsons_Contingency_Coefficient = np.sqrt(chi_squared/(chi_squared+Sample_Size)) 
    

    # Standard Deviations of the Effect Sizes
    Probabilities_Table = contingency_table/Sample_Size
    Squared_Probabilities_Table = Probabilities_Table**2
    Cubic_Probability_Table = Probabilities_Table**3

    column_marginals = np.sum(Probabilities_Table, axis=0)
    row_marginals = np.sum(Probabilities_Table, axis=1)
    marginal_products = np.outer(row_marginals, column_marginals)
    Corrected_Probabilities_Table = Squared_Probabilities_Table / marginal_products

    column_marginals_Squared = np.sum(Probabilities_Table, axis=0)**2
    row_marginals_Squared = np.sum(Probabilities_Table, axis=1)**2
    marginal_products_Corrected= np.outer(row_marginals_Squared, column_marginals_Squared)
    Corrected_Squared_Probabilities_Table = Cubic_Probability_Table / marginal_products_Corrected

    row_marginals_corrected = np.sum(Corrected_Probabilities_Table, axis=1)
    coloumn_marginals_corrected = np.sum(Corrected_Probabilities_Table, axis=0)
    marginal_products_corrected_table = np.outer(row_marginals_corrected,coloumn_marginals_corrected )

    term1 = np.sum(Corrected_Squared_Probabilities_Table)
    term2 = np.sum(row_marginals_corrected**2 / row_marginals)
    term3 = np.sum(coloumn_marginals_corrected**2 / column_marginals)  
    term4 = np.sum(marginal_products_corrected_table)

    standard_deviation_phi_square = np.sqrt((4*term1 - 3*term2 - 3*term3 + 2*term4) / Sample_Size)
    Standard_deviation_Cramer_V = (1 / (2 * (min(Number_of_Coloumns-1, Number_of_rows-1))**0.5 * Cramer_V)) * standard_deviation_phi_square
    Standard_deviation_Tschuprows_T = standard_deviation_phi_square / np.sqrt((4*Tschuprows_T**2)*np.sqrt((Number_of_Coloumns-1)*(Number_of_rows-1)))
    Standard_deviation_Contingency_Coefficient = (1 / (2*phi*(1+phi_square)**(3/2)))*standard_deviation_phi_square


    # Bias Corrected Measures
    Phi_Square_Bias_Corrected = max((standard_deviation_phi_square - (1 / (Sample_Size-1)) * (Number_of_Coloumns - 1) * (Number_of_rows - 1)), 0)
    chi_square_Bias_corrected = Phi_Square_Bias_Corrected * Sample_Size
    Number_of_rows_Corrected = max(Number_of_rows - (1/(Sample_Size-1)) * (Number_of_rows-1)**2,0)
    Number_of_coloumns_Corrected = max(Number_of_Coloumns - (1/(Sample_Size-1)) * (Number_of_Coloumns-1)**2,0)
    
    Bias_Corrected_Tschuprows_T = (max(Phi_Square_Bias_Corrected,0) / ( (Number_of_coloumns_Corrected-1) * (Number_of_rows_Corrected-1))**0.5)**0.5 # Bergsma, 2013
    Bias_Corrected_Cramer_V = (max(Phi_Square_Bias_Corrected,0) / (min((Number_of_coloumns_Corrected-1), (Number_of_rows_Corrected-1))))**0.5 # Bergsma, 2013
    Bias_Corrected_Contingency_Coefficeint = np.sqrt(max(Phi_Square_Bias_Corrected,0)/ (max(Phi_Square_Bias_Corrected,0)+1))

    Bias_Corrected_Cramer_V_standard_deviation = 0 if Bias_Corrected_Cramer_V == 0 else (1 / (2 * (min(Number_of_coloumns_Corrected-1, Number_of_rows_Corrected-1))**0.5 * Bias_Corrected_Cramer_V)) * standard_deviation_phi_square
    Bias_Corrected_Tschuprows_T_Standard_deviation = 0 if Bias_Corrected_Tschuprows_T == 0 else (1 / (2 * (Number_of_coloumns_Corrected-1) * (Number_of_rows_Corrected-1) * Bias_Corrected_Tschuprows_T)) * standard_deviation_phi_square
    Bias_Corrected_Contingency_Coefficient_Standard_deviation = 0 if Bias_Corrected_Cramer_V == 0 else (1 / (2*np.sqrt(Phi_Square_Bias_Corrected)*(1+Phi_Square_Bias_Corrected)**(3/2)))*standard_deviation_phi_square



    ################################
    # Maximum Corrected Measures ###
    ################################
    
    # Maximum Corrected Chi Square by Berry, Mielke and Jhonston
    Maximum_Corrected_Matrix = Berry_Mielke_Maximum_Corrected_Cramer_V_output_matrix(matrix) # Berry and mielke Lambda
    row_totals_max = np.sum(Maximum_Corrected_Matrix, axis=1)
    col_totals_max = np.sum(Maximum_Corrected_Matrix, axis=0)
    total_max = np.sum(Maximum_Corrected_Matrix)
    expected_max = np.outer(row_totals_max, col_totals_max) / total_max
    
    # This Method removes zeros for the calucaltion to avoid division by zero when one clacualte expected values 
    observeved = Maximum_Corrected_Matrix.flatten()
    expected_max = np.array(expected_max).flatten()
    zero_positions = np.where((observeved == 0) & (expected_max == 0))[0]
    observeved = np.delete(observeved, zero_positions)
    expected_max = np.delete(expected_max, zero_positions)
    Maximum_Corrected_Chi_Square_BMJ = np.sum((observeved - expected_max)**2 / expected_max)
    Maximum_Corrected_cramers_v = "Maximum Corrected Cramer V is not valid for this Sample" if Maximum_Corrected_Chi_Square_BMJ == 0 else chi_squared / Maximum_Corrected_Chi_Square_BMJ

    # Maximum Corrected Measures from Liebtrau (Max Corrected Contingency Coefficeint and Tschuprows T)

    ###################################################################################################
    q = min(Number_of_Coloumns, Number_of_rows)
    Maximum_Chi_Square = Sample_Size*(q-1)
    Maximum_Contingency_Coefficient =  np.sqrt((q-1) / q) #Sakoda, 1977
    Maximum_Tschuprows_T = ((q-1) / (max((Number_of_Coloumns-1), (Number_of_rows-1))))**0.25 #Liebtrau, 1983 Suggestion
    Maximum_Corrected_Tschuprows_T = Tschuprows_T / Maximum_Tschuprows_T
    Maximum_Corrected_Contingency_Coefficient = Pearsons_Contingency_Coefficient / Maximum_Contingency_Coefficient
    Maximum_Corrected_CC_Standard_Error = (  np.sqrt(q / (q-1)) * Standard_deviation_Contingency_Coefficient)
    Maximum_Corrected_Tscuprows_Standard_Error = ((((((max((Number_of_Coloumns-1), (Number_of_rows-1)))/(q-1))**0.25)))  * Standard_deviation_Tschuprows_T)


    ###############################
    ######## Confidence Intervals #
    ###############################
    
    # Confidence Intervals using the asymptotic standard error  (Bishop et al., 1975)
    z_crit = norm.ppf(confidence_level + ((1 - confidence_level) / 2))    
    ci_lower_phi = phi - standard_deviation_phi_square*z_crit
    ci_upper_phi = phi + standard_deviation_phi_square*z_crit

    ci_lower_cramer = max(Cramer_V - Standard_deviation_Cramer_V*z_crit, 0)
    ci_upper_cramer = min(Cramer_V + Standard_deviation_Cramer_V*z_crit,1)
    ci_lower_tschuprows = max(Tschuprows_T - Standard_deviation_Tschuprows_T*z_crit, 0)
    ci_upper_tschuprows = min(Tschuprows_T + Standard_deviation_Tschuprows_T*z_crit, 1)
    ci_lower_cc = max(Pearsons_Contingency_Coefficient - Standard_deviation_Contingency_Coefficient*z_crit, 0)
    ci_upper_cc = min(Pearsons_Contingency_Coefficient + Standard_deviation_Contingency_Coefficient*z_crit,1)
    ci_lower_cramer_corrected = max(Bias_Corrected_Cramer_V - Bias_Corrected_Cramer_V_standard_deviation*z_crit, 0) # type: ignore
    ci_upper_cramer_corrected = min(Bias_Corrected_Cramer_V + Bias_Corrected_Cramer_V_standard_deviation*z_crit, 1) # type: ignore
    ci_lower_tschuprows_corrected = max(Bias_Corrected_Tschuprows_T - Bias_Corrected_Tschuprows_T_Standard_deviation*z_crit, 0) # type: ignore
    ci_upper_tschuprows_corrected = min(Bias_Corrected_Tschuprows_T + Bias_Corrected_Tschuprows_T_Standard_deviation*z_crit, 1) # type: ignore
    ci_lower_CC_corrected = max(Bias_Corrected_Contingency_Coefficeint - Bias_Corrected_Contingency_Coefficient_Standard_deviation*z_crit, 0)
    ci_upper_CC_corrected = min(Bias_Corrected_Contingency_Coefficeint + Bias_Corrected_Contingency_Coefficient_Standard_deviation*z_crit, 1)
    ci_lower_Tschuprows_max_corrected = max(Maximum_Corrected_Tschuprows_T - Maximum_Corrected_Tscuprows_Standard_Error*z_crit, 0)
    ci_upper_Tschuprows_max_corrected = min(Maximum_Corrected_Tschuprows_T + Maximum_Corrected_Tscuprows_Standard_Error*z_crit, 1)
    ci_lower_cc_max_corrected = max(Maximum_Corrected_Contingency_Coefficient - Maximum_Corrected_Tscuprows_Standard_Error*z_crit, 0)
    ci_upper_cc_max_corrected = min(Maximum_Corrected_Contingency_Coefficient + Maximum_Corrected_Tscuprows_Standard_Error*z_crit, 1)


    # Confidence Interals using the non-central distribution

    # Non Central CI's for Chi Square
    lower_ci_ncp, upper_ci_ncp =  ncp_ci(chi_squared ,degrees_of_freedom_chi_square, confidence_level)
    lower_ci_ncp, upper_ci_ncp = max(lower_ci_ncp,0), (upper_ci_ncp)
    
    lower_ci_ncp_bias_corrected, upper_ci_ncp_bias_corrected = ncp_ci(chi_square_Bias_corrected, degrees_of_freedom_chi_square, confidence_level) # Consider Correcting the DFs for these CI's as well
    
    lower_ncp_max_chi_square, upper_ncp_max_chi_square = ncp_ci(Maximum_Chi_Square, degrees_of_freedom_chi_square, confidence_level)
    lower_ncp_max_chi_square, upper_ncp_max_chi_square = max(lower_ncp_max_chi_square,0), upper_ncp_max_chi_square

    lower_ncp_max_chi_square_BMJ, upper_ncp_max_chi_square_BMJ = ncp_ci(Maximum_Corrected_Chi_Square_BMJ, degrees_of_freedom_chi_square, confidence_level)
    lower_ncp_max_chi_square_BMJ, upper_ncp_max_chi_square_BMJ = max(lower_ncp_max_chi_square_BMJ,0), upper_ncp_max_chi_square_BMJ
        
    #lower_ncp_max_chi_square_BMJ, upper_ncp_max_chi_square_BMJ = 0,0 if Maximum_Corrected_Chi_Square_BMJ == 0 else ncp_ci(Maximum_Corrected_Chi_Square_BMJ, degrees_of_freedom_chi_square, confidence_level)
    #lower_ncp_max_chi_square_BMJ, upper_ncp_max_chi_square_BMJ = 0 if Maximum_Corrected_Chi_Square_BMJ == 0  else max(lower_ncp_max_chi_square,0), 0 if Maximum_Corrected_Chi_Square_BMJ == 0  else upper_ncp_max_chi_square

    # Non Central CI's for Phi Square
    lower_ncp_phi_square = max(lower_ci_ncp / Sample_Size,0)
    upper_ncp_phi_square = min(upper_ci_ncp / Sample_Size, 1)
    lower_ncp_phi_square_bias_corrected = 0 if lower_ci_ncp_bias_corrected == 0 else lower_ci_ncp_bias_corrected / Sample_Size
    upper_ncp_phi_square_bias_corrected = 0 if upper_ci_ncp_bias_corrected == 0 else upper_ci_ncp_bias_corrected / Sample_Size  
    lower_ncp_phi_square_max_corrected = max(lower_ncp_max_chi_square / Sample_Size,0)
    upper_ncp_phi_square_max_corrected = min(upper_ncp_max_chi_square / Sample_Size, 1)

    # Non Central CI's for Effect Sizes
    ci_ncp_lower_cramer = max(np.sqrt(lower_ncp_phi_square / (q-1)), 0)
    ci_ncp_upper_cramer = min(np.sqrt(upper_ncp_phi_square / (q-1)), 1)
    ci_ncp_lower_Tschuprow = max(np.sqrt(lower_ncp_phi_square / (np.sqrt(degrees_of_freedom_chi_square))),0)
    ci_ncp_upper_Tschuprow = min(np.sqrt(upper_ncp_phi_square / ( np.sqrt(degrees_of_freedom_chi_square))),1)    
    ci_lower_cc_ncp = max(np.sqrt(lower_ncp_phi_square / (lower_ncp_phi_square + 1)),0)
    ci_upper_cc_ncp = min(np.sqrt(upper_ncp_phi_square / (upper_ncp_phi_square + 1)),1)
    
    # Non Central CI's for Bias Corrected Measures
    ci_ncp_lower_cramer_bias_corrected = max(np.sqrt(lower_ncp_phi_square_bias_corrected / (q-1)), 0)
    ci_ncp_upper_cramer_bias_corrected = (np.sqrt(upper_ncp_phi_square_bias_corrected / (q-1)))
    ci_ncp_lower_Tschuprow_bias_corrected = max(np.sqrt(lower_ncp_phi_square_bias_corrected / (np.sqrt(degrees_of_freedom_chi_square))),0)
    ci_ncp_upper_Tschuprow_bias_corrected = (np.sqrt(upper_ncp_phi_square_bias_corrected / (np.sqrt(degrees_of_freedom_chi_square))))
    ci_lower_cc_bias_corrected_ncp = max(np.sqrt(lower_ncp_phi_square_bias_corrected / (lower_ncp_phi_square_bias_corrected + 1)),0)
    ci_upper_cc_bias_corrected_ncp = (np.sqrt(upper_ncp_phi_square_bias_corrected / (upper_ncp_phi_square_bias_corrected + 1)))
    
    # Non Central CI's for Maximum Corrected Measures
    ci_ncp_lower_cramer_max_corrected = 0 if Maximum_Corrected_Chi_Square_BMJ == 0  else max(lower_ci_ncp / lower_ncp_max_chi_square_BMJ, 0)
    ci_ncp_upper_cramer_max_corrected = 0 if Maximum_Corrected_Chi_Square_BMJ == 0  else upper_ci_ncp / upper_ncp_max_chi_square_BMJ
    ci_ncp_lower_tschuprows_max_corrected = ci_ncp_lower_Tschuprow / (np.sqrt((lower_ncp_phi_square_max_corrected / np.sqrt((Number_of_Coloumns-1)*(Number_of_rows-1))) * ((((max((Number_of_Coloumns-1), (Number_of_rows-1)))/(q-1))**0.25)))) 
    ci_ncp_upper_tschuprows_max_corrected = ci_ncp_upper_Tschuprow / np.sqrt((upper_ncp_phi_square_max_corrected / np.sqrt((Number_of_Coloumns-1)*(Number_of_rows-1))) * ((((max((Number_of_Coloumns-1), (Number_of_rows-1)))/(q-1))**0.25))) 
    ci_lower_cc_max_corrected_ncp =  max(ci_lower_cc_ncp / np.sqrt(lower_ncp_phi_square_max_corrected/ (1+lower_ncp_phi_square_max_corrected)), 0)
    ci_upper_cc_max_corrected_ncp =  min(ci_upper_cc_ncp / np.sqrt(upper_ncp_phi_square_max_corrected/ (1+upper_ncp_phi_square_max_corrected)), 1)

    # Uncertainty Coefficient - Needs to be in a Contingency Table
    Sum_Of_Rows = np.sum(contingency_table, axis=1)
    Sum_Of_Columns = np.sum(contingency_table, axis=0)
    sample_size = np.sum(contingency_table)
    HX = np.sum((Sum_Of_Rows * np.log(Sum_Of_Rows / sample_size)) / sample_size)
    HY = np.sum((Sum_Of_Columns * np.log(Sum_Of_Columns / sample_size)) / sample_size)
    probabilites_table = contingency_table / sample_size
    term1 = np.where(probabilites_table==0, 0, np.where(probabilites_table==1, 0, np.log(probabilites_table + (probabilites_table==0)))) # Adding a small value to prevent log(0)
    term2 = np.outer(Sum_Of_Rows, Sum_Of_Columns) / sample_size ** 2
    term3 = np.where(term2==0, 0, np.log(term2))
    term4 = Sum_Of_Rows / sample_size
    term5 = np.where(term4==0, 0, np.log(term4))
    term6 = contingency_table / Sum_Of_Columns
    term7 = np.where(term6==0, 0, np.where(term6==1, 0, np.log(term6 + (term6==0)))) # Adding a small value to prevent log(0)
    term8 = Sum_Of_Columns / sample_size
    term9 = np.where(term8==0, 0, np.log(term8))
    term10 = contingency_table / Sum_Of_Rows[:, np.newaxis]
    term11 = np.where(term10==0, 0, np.where(term10==1, 0, np.log(term10 + (term10==0)))) # Adding a small value to prevent log(0)

    HXY = np.sum(contingency_table * term1) / sample_size
    
    # Calculate Effect Size UC 
    Uncertainty_Coefficient_Symmetric = 2 * (HX + HY - HXY) / (HX + HY)
    Uncertainty_Coefficient_Rows = (HX + HY - HXY) / HX
    Uncertainty_Coefficient_Columns = (HX + HY - HXY) / HY

    # Calculate the Asymptotic Standard Errors (Standard Errors)
    Standard_Error_Symmetric = np.sqrt((4 * np.sum(contingency_table * (HXY * term3 - (HX + HY) * term1) ** 2) / (sample_size ** 2 * (HX + HY) ** 4)))
    Standard_Error_Rows =    np.sqrt(np.sum(contingency_table * (HX * term7 + (HY - HXY) * term5[:, np.newaxis]) ** 2) / (sample_size ** 2 * HX ** 4))
    Standard_Error_Columns = np.sqrt(np.sum(contingency_table * (HY * term11 + (HX - HXY) * term9) ** 2) / (sample_size ** 2 * HY ** 4))

    # Calculate p_values        
    Z_value_Symmetric = "inf" if Uncertainty_Coefficient_Symmetric == 1 else Uncertainty_Coefficient_Symmetric / Standard_Error_Symmetric
    Z_value_Rows = "inf" if Uncertainty_Coefficient_Rows == 1 else Uncertainty_Coefficient_Rows / Standard_Error_Rows
    Z_value_Columns = "inf" if Uncertainty_Coefficient_Columns == 1 else Uncertainty_Coefficient_Columns / Standard_Error_Columns

    # Confidence Intervals
    z_crit = 1 - (1 - confidence_level) / 2

    # Confidence Intervals for UC
    Symmetric_CI_lower = max(Uncertainty_Coefficient_Symmetric - z_crit * np.sqrt(Standard_Error_Symmetric),0)
    Symmetric_CI_upper = min(Uncertainty_Coefficient_Symmetric + z_crit * np.sqrt(Standard_Error_Symmetric),1)
    Rows_CI_lower = max(Uncertainty_Coefficient_Rows - z_crit * np.sqrt(Standard_Error_Rows),0)
    Rows_CI_upper = min(Uncertainty_Coefficient_Rows + z_crit * np.sqrt(Standard_Error_Rows),1)
    Columns_CI_lower = max(Uncertainty_Coefficient_Columns - z_crit * np.sqrt(Standard_Error_Columns),0)
    Columns_CIs_upper = min(Uncertainty_Coefficient_Columns + z_crit * np.sqrt(Standard_Error_Columns),1) 
    

    results = {}

    # Table 1 - Inferntial Statistics
    results["Confidence_Level"] = round(confidence_level, 4)
    results["chi_square_Bias_corrected"] = round(chi_square_Bias_corrected, 4)

    results["Chi Square"] = round(chi_squared, 4)
    results["Degrees of Freedom Chi Square"] = round(degrees_of_freedom_chi_square, 4) # this is identical for both the likelihood ratio and chi-sqaure
    results["p_value Chi Square"] = p_value # this one is suitable for all kind of effect sizes
    results["Likelihood Ratio"] = np.around(likelihood_ratio, 4)
    results["Likelihood Ratio p_value"] = (likelihood_ratio_p_value)

    # Table 2 - Symmetric Measures of Association

    # Table 2A - Normal Measures
    results["___________________________________________"] = ''  
    results["Cramer V"] = round(Cramer_V, 4)
    results["Pearson's Contingency Coefficient"] = round(Pearsons_Contingency_Coefficient, 4)
    results["Tschuprow's T"] = round(Tschuprows_T, 7)
    results["Standard Error of Cramer V"] = round(Standard_deviation_Cramer_V, 4)
    results["Standard Error of Contingency Coefficient"] = round(Standard_deviation_Contingency_Coefficient, 4)
    results["Standard Error of Tschuprow's T"] = round(Standard_deviation_Tschuprows_T, 4)
    results["Asymptotic CI Cramer V"] = f"({round(ci_lower_cramer, 4)}, {round(ci_upper_cramer, 4)})"
    results["Asymptotic CI Contingency Coefficient"] = f"({round(ci_lower_cc, 4)}, {round(ci_upper_cc, 4)})"
    results["Asymptotic Tschuprow's T"] = f"({round(ci_lower_tschuprows, 4)}, {round(ci_upper_tschuprows, 4)})"
    results["NCP CI Cramer V"] = f"({round(ci_ncp_lower_cramer, 4)}, {round(ci_ncp_upper_cramer, 4)})"
    results["NCP CI Contingency Coefficient"] = f"({round(ci_lower_cc_ncp, 4)}, {round(ci_upper_cc_ncp, 4)})"
    results["NCP CI Tschuprow's T"] = f"({round(ci_ncp_lower_Tschuprow, 4)}, {round(ci_ncp_upper_Tschuprow, 4)})"
    formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
    results["Statistical Line Cramer's V"] = "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Cramer's \033[3mV\033[0m = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(int(degrees_of_freedom_chi_square), sample_size, chi_squared, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(Cramer_V,3), confidence_level*100 ,round(ci_ncp_lower_cramer, 3),round(ci_ncp_upper_cramer,3))
    results["Statistical Line Contingency Coefficient"] = "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, \033[3mC\033[0m = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(int(degrees_of_freedom_chi_square), sample_size, chi_squared, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(Pearsons_Contingency_Coefficient,3), confidence_level*100 ,round(ci_lower_cc_ncp, 3),round(ci_upper_cc_ncp,3))
    results["Statistical Line Tschuprow's T"] = "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Tschuprow's \033[3mT\033[0m = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(int(degrees_of_freedom_chi_square), sample_size, chi_squared, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(Tschuprows_T,3), confidence_level*100 ,round(ci_ncp_lower_Tschuprow, 3),round(ci_ncp_upper_Tschuprow,3))
    
    # Table 2B - Bias Corrected Measures (Adjusted)
    results["____________________________________________"] = ''  
    results["Adjusted Cramer's V"] = round(Bias_Corrected_Cramer_V, 7)
    results["Adjusted Tschuprow's T"] = round(Bias_Corrected_Tschuprows_T, 7)
    results["Adjusted Contingency Coefficient"] = round(Bias_Corrected_Contingency_Coefficeint, 4)
    results["Standard Error of Bias Corrected Cramer's V"] = round(Bias_Corrected_Cramer_V_standard_deviation, 4)
    results["Standard Error of Bias Corrected Tschuprows T"] = round(Bias_Corrected_Tschuprows_T_Standard_deviation, 4)
    results["Standard Error of Bias Corrected Contingency Coefficient"] = round(Bias_Corrected_Contingency_Coefficient_Standard_deviation, 4)
    results["CI bias corrected Cramer V"] = f"({np.around(ci_lower_cramer_corrected, 4)}, {np.around(ci_upper_cramer_corrected, 4)})"
    results["CI bias corrected Tschuprows T"] = f"({np.around(ci_lower_tschuprows_corrected, 4)}, {np.around(ci_upper_tschuprows_corrected, 4)})"
    results["CI bias corrected Contingency Coefficient"] = f"({np.around(ci_lower_CC_corrected, 4)}, {np.around(ci_upper_CC_corrected, 4)})"
    results["NCP CI Cramer's V Bias Corrected"] = f"({np.around(ci_ncp_lower_cramer_bias_corrected, 4)}, {np.around(ci_ncp_upper_cramer_bias_corrected, 4)})"
    results["NCP CI Tschuprow's T Bias Corrected"] = f"({np.around(ci_ncp_lower_Tschuprow_bias_corrected, 4)}, {np.around(ci_ncp_upper_Tschuprow_bias_corrected, 4)})"
    results["NCP CI Contingency Coefficient Bias Corrected"] = f"({np.around(ci_lower_cc_bias_corrected_ncp, 4)}, {np.around(ci_upper_cc_bias_corrected_ncp, 4)})"
    results["Statistical Line Adjusted Cramer's V"] =              "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Cramer's \033[3mV\033[0m = {:.3f}, {}% CI(Pivotal) [{}, {}]".format(int(degrees_of_freedom_chi_square), sample_size, chi_squared, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(Bias_Corrected_Cramer_V               ,3), confidence_level*100 ,round(ci_ncp_lower_cramer_bias_corrected, 3),np.around(ci_ncp_upper_cramer_bias_corrected,3))
    results["Statistical Line Adjusted Contingency Coefficient"] = "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, \033[3mC\033[0m = {:.3f}, {}% CI(Pivotal) [{}, {}]".format(int(degrees_of_freedom_chi_square), sample_size, chi_squared, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(Bias_Corrected_Contingency_Coefficeint,3), confidence_level*100 ,round(ci_lower_cc_bias_corrected_ncp, 3), np.around(ci_upper_cc_bias_corrected_ncp,3))
    results["Statistical Line Adjusted Tschuprow's T"] = "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Tschuprow's \033[3mT\033[0m = {:.3f}, {}% CI(Pivotal) [{}, {}]".format(int(degrees_of_freedom_chi_square), sample_size, chi_squared, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(Bias_Corrected_Tschuprows_T,3), confidence_level*100 ,round(ci_ncp_lower_Tschuprow_bias_corrected, 3), np.around(ci_ncp_upper_Tschuprow_bias_corrected,3))

    results["Bias Corrected CI Testing"] = upper_ci_ncp_bias_corrected

    # Table 2C - Maximum Corrected Measures
    results["______________________________________________"] = ''  
    results["Maximum Corrected Cramers V (Berry, Mielke, Jhonston)"] = (Maximum_Corrected_cramers_v)
    results["Maximum Corrected Tschuprow's T"] = round(Maximum_Corrected_Tschuprows_T, 7)
    results["Maximum Corrected Contingency Coefficient (Sakoda, 1977)"] = round(Maximum_Corrected_Contingency_Coefficient, 4)
    results["Standard Error of Maximum Corrected Tschuprows T"] = round(Maximum_Corrected_Tscuprows_Standard_Error, 4)
    results["Standard Error of Maximum Corrected Contingency Coefficient"] = round(Maximum_Corrected_CC_Standard_Error, 4)
    results["CI Max corrected Tschuprows T"] = f"({np.around(ci_lower_Tschuprows_max_corrected, 4)}, {np.around(ci_upper_Tschuprows_max_corrected, 4)})"
    results["CI Max corrected Contingency Coefficient"] = f"({np.around(ci_lower_cc_max_corrected, 4)}, {np.around(ci_upper_cc_max_corrected, 4)})"
    results["NCP CI Cramer's V Max corrected"] = f"({round(ci_ncp_lower_cramer_max_corrected, 4)}, {np.around(ci_ncp_upper_cramer_max_corrected, 4)})"
    results["NCP CI Tschuprow's T Max corrected"] = f"({round(ci_ncp_lower_tschuprows_max_corrected, 4)}, {round(ci_ncp_upper_tschuprows_max_corrected, 4)})"
    results["NCP CI Contingency Coefficient Max corrected"] = f"({round(ci_lower_cc_max_corrected_ncp, 4)}, {round(ci_upper_cc_max_corrected_ncp, 4)})"
    results["Statistical Line Maximum Corrected Cramer's V"] = Maximum_Corrected_cramers_v if isinstance(Maximum_Corrected_cramers_v, str) else "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Maximum Corrected Cramer's \033[3mV\033[0m = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(int(degrees_of_freedom_chi_square), sample_size, chi_squared, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(Maximum_Corrected_cramers_v,3), confidence_level * 100 ,round(ci_ncp_lower_cramer_max_corrected, 3), np.around(ci_ncp_upper_cramer_max_corrected,3)) # type: ignore
    results["Statistical Line Maximum Corrected Contingency Coefficient"] = "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Maximum Corrected \033[3mC\033[0m = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(int(degrees_of_freedom_chi_square), sample_size, chi_squared, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(Maximum_Corrected_Contingency_Coefficient,3), confidence_level*100 ,round(ci_lower_cc_max_corrected_ncp, 3),round(ci_upper_cc_max_corrected_ncp,3))
    results["Statistical Line Maximum Corrected Tschuprow's T"] = "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Maximum Corrected Tschuprow's \033[3mT\033[0m = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(int(degrees_of_freedom_chi_square), sample_size, chi_squared, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(Maximum_Corrected_Tschuprows_T,3), confidence_level*100 ,round(ci_ncp_lower_tschuprows_max_corrected, 3),round(ci_ncp_upper_tschuprows_max_corrected,3))
  

    # Table 2D - Phi Square/ Cohens w
    results["__________________________________________________"] = ''  
    results["Phi / Cohen's w"] = round(phi, 4)
    results["Standard Deviation of Phi Square"] = round(standard_deviation_phi_square, 4)
    results["Adjusted Phi"] = round(max(Phi_Square_Bias_Corrected,0), 7)
    results["Asymptotic CI Phi / Cohens w"] = f"({round(ci_lower_phi, 4)}, {round(ci_upper_phi, 4)})"
    results["NCP CI's Phi / Cohens w"] = f"({round(np.sqrt(lower_ncp_phi_square), 4)}, {round(np.sqrt(upper_ncp_phi_square), 4)})"
    results["Ncp CI's Adjusted Phi / Cohens w"] = f"({np.around(np.sqrt(lower_ncp_phi_square_bias_corrected), 4)}, {np.around(np.sqrt(upper_ncp_phi_square_bias_corrected), 4)})"

    results["______________________________________________"] = ''  
    results["Statistical Line Cohen's w"] =  "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Cohen's \033[3mw\033[0m = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(int(degrees_of_freedom_chi_square), sample_size, chi_squared, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(phi,3) ,confidence_level*100, round(np.sqrt(lower_ncp_phi_square),3), round(np.sqrt(upper_ncp_phi_square),3))
    results["Statistical Line Adjusted Cohen's w"] =  "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Adjusted Cohen's \033[3mw\033[0m = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(int(degrees_of_freedom_chi_square), sample_size, chi_squared, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(Phi_Square_Bias_Corrected,3) ,confidence_level*100, np.around(np.sqrt(lower_ncp_phi_square_bias_corrected), 4), np.around(np.sqrt(upper_ncp_phi_square),3))

    # Uncertinty Coefficient
    results["Theil's Uncertainty Coefficient (Symmetric)"] = Uncertainty_Coefficient_Symmetric
    results["Theil's Uncertainty Coefficient (Rows)"] = Uncertainty_Coefficient_Rows
    results["Theil's Uncertainty Coefficient (Columns)"] = Uncertainty_Coefficient_Columns
    results["Theil's Uncertainty Coefficient Standard Error (Symmetric)"] = Standard_Error_Symmetric
    results["Theil's Uncertainty Coefficient Standard Error (Rows)"] = Standard_Error_Rows
    results["Theil's Uncertainty Coefficient Standard Error (Columns)"] = Standard_Error_Columns
    results["Theil's Uncertainty Coefficient Z-value (Symmetric)"] = Z_value_Symmetric
    results["Theil's Uncertainty Coefficient Z-value (Rows)"] = Z_value_Rows
    results["Theil's Uncertainty Coefficient Z-value (Columns)"] = Z_value_Columns
    results["Theil's Uncertainty Coefficient Confidence Intervals (Symmetric)"] =  f"({round(Symmetric_CI_lower, 4)}, {round(Symmetric_CI_upper, 4)})"
    results["Theil's Uncertainty Coefficient Confidence Intervals (Rows)"] =  f"({round(Rows_CI_lower, 4)}, {round(Rows_CI_upper, 4)})"
    results["Theil's Uncertainty Coefficient Confidence Intervals (Columns)"] =  f"({round(Columns_CI_lower, 4)}, {round(Columns_CIs_upper, 4)})"
    results["Statistical Line Theil's Uncertainty Coefficient Symmetric"] = "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Theil's \033[3mU\033[0m = {:.3f}, {}% CI [{:.3f}, {:.3f}]".format(int(degrees_of_freedom_chi_square), sample_size, chi_squared, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(Uncertainty_Coefficient_Symmetric,3), confidence_level*100 ,round(Symmetric_CI_lower, 3),round(Symmetric_CI_upper,3))
    results["Statistical Line Theil's Uncertainty Coefficient Rows"] = "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Theil's \033[3mU\033[0m = {:.3f}, {}% CI [{:.3f}, {:.3f}]".format(int(degrees_of_freedom_chi_square), sample_size, chi_squared, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(Uncertainty_Coefficient_Rows,3), confidence_level*100 ,round(Rows_CI_lower, 3),round(Rows_CI_upper,3))
    results["Statistical Line Theil's Uncertainty Coefficient Columns"] = "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Theil's \033[3mU\033[0m = {:.3f}, {}% CI [{:.3f}, {:.3f}]".format(int(degrees_of_freedom_chi_square), sample_size, chi_squared, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(Uncertainty_Coefficient_Columns,3), confidence_level*100 ,round(Columns_CI_lower, 3),round(Columns_CIs_upper,3))

    result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
    return result_str 



class NominalbyNominal():
    @staticmethod
    def Nominal_By_Nominal_from_Chi_Score(params: dict) -> dict:

        # Set params
        chi_score = params["Chi Sqaure Score"]
        sample_size = params["Sample Size"]
        degrees_of_freedom = params["Degrees of Freedom"]                
        confidence_level_percentages = params["Confidence Level"]

        # Calculate
        ###########
        confidence_level = confidence_level_percentages / 100

        cohensw = np.sqrt(chi_score/sample_size)
        cramerv = np.sqrt(chi_score/(degrees_of_freedom*sample_size))
        contingency_coefficeint = np.sqrt(chi_score / (chi_score + sample_size))

        lower_ci_chi, upper_ci_chi = ncp_ci(chi_score, degrees_of_freedom, confidence_level)

        lower_ci_cohensw = np.sqrt(lower_ci_chi/sample_size)
        upper_ci_cohensw = np.sqrt(upper_ci_chi/sample_size)
        lower_ci_cramerv = np.sqrt(lower_ci_chi/(degrees_of_freedom*sample_size))
        upper_ci_cramerv = np.sqrt(upper_ci_chi/(degrees_of_freedom*sample_size))
        lower_ci_contingency_coefficient = np.sqrt(lower_ci_chi / (lower_ci_chi + sample_size))
        upper_ci_contingency_coefficient = np.sqrt(upper_ci_chi / (upper_ci_chi + sample_size))
        p_value = chi2.sf((abs(chi_score)), degrees_of_freedom)

        # Set results
        results = {}
        results["Cohen's w / Phi"] = round(cohensw,4)
        results["Cramer's V"] = round(cramerv,4)
        results["Contigency Coefficient"] = round(contingency_coefficeint,4)
        
        results["Chi Square Score"] = round(chi_score, 4)
        results["Degrees of Freedom"] = round(degrees_of_freedom, 4)

        results["p-value"] = np.around(p_value,4)
        results["Cohen's w CI Lower"] = round(lower_ci_cohensw, 4)
        results["Cohen's w CI Upper"] = round(upper_ci_cohensw, 4)
        results["Cramer's V CI Lower"] = round(lower_ci_cramerv, 4)
        results["Cramer's V CI Upper"] = round(upper_ci_cramerv, 4)
        results["Contingency Coefficient CI Lower"] = round(lower_ci_contingency_coefficient, 4)
        results["Contingency Coefficient CI Upper"] = round(upper_ci_contingency_coefficient, 4)
        
        formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Cohen's w"] =  "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Cohen's  \033[3mw\033[0m = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(int(degrees_of_freedom), sample_size, chi_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(cohensw,3) ,confidence_level_percentages, round(lower_ci_cohensw,3), round(upper_ci_cohensw,3))
        results["Statistical Line Cramer's V"] = "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, Cramer's \033[3mV\033[0m = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(int(degrees_of_freedom), sample_size, chi_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(cramerv,3), confidence_level_percentages ,round(lower_ci_cramerv, 3),round(upper_ci_cramerv,3))
        results["Statistical Line Contingency Coefficient"] = "\033[3m\u03C7\u00B2\033[0m({}, N = {}) = {:.3f}, {}{}, \033[3mC\033[0m = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(int(degrees_of_freedom), sample_size, chi_score, '\033[3mp = \033[0m' if p_value >= 0.001 else '', formatted_p_value, round(contingency_coefficeint,3), confidence_level_percentages ,round(lower_ci_contingency_coefficient, 3),round(upper_ci_contingency_coefficient,3))

        return results

    @staticmethod
    def Nominal_By_Nominal_from_Contingency_Table(params: dict) -> dict:

        # Set params
        Contingency_Table = params["Contingency Table"]
        confidence_level_percentages = params["Confidence Level"]
        confidence_level = confidence_level_percentages / 100

        output_nominal_by_nominal = multilevel_contingency_tables(Contingency_Table, confidence_level)
        output_lambda = goodman_kruskal_lamda_correlation(Contingency_Table, confidence_level)
        output_Goodman_Kruskal_Tau = Goodman_Kruskal_Tau(Contingency_Table, confidence_level)
        
        results = {}

        results["Nominal by Nominal Association"] =  output_nominal_by_nominal
        results["Lambda Table"] =  output_lambda
        results["Tau Table"] =  output_Goodman_Kruskal_Tau

        return results
    

    @staticmethod
    def Nominal_By_Nominal_from_Data(params: dict) -> dict:

        # Set params
        column_1 = params["Column 1"]
        column_2 = params["Column 2"]
        confidence_level_percentages = params["Confidence Level"]

        confidence_level = confidence_level_percentages / 100
        Contingency_Table = Columns_to_Contingency(column_1, column_2)

        #output_nominal_by_nominal = multilevel_contingency_tables(Contingency_Table, confidence_level)
        output_lambda = goodman_kruskal_lamda_correlation(Contingency_Table, confidence_level)
        output_Goodman_Kruskal_Tau = Goodman_Kruskal_Tau(Contingency_Table, confidence_level)
        
        results = {}
        results["Contingency Table"] =  Contingency_Table
        #results["Nominal by Nominal Association"] =  output_nominal_by_nominal
        results["Lambda Table"] = output_lambda
        results["Tau Table"] =  output_Goodman_Kruskal_Tau

        return results

    # Things to Consider

    # 1. Consider Adding more Staff to the Inferntial Statistic Part
