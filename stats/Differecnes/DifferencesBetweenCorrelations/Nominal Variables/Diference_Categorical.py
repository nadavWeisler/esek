
#################################################################
###### Difference Between Categorical Association Measures ######
#################################################################

import numpy as np
from itertools import product
from scipy.stats import norm
import random
import pandas as pd

# Relevant functions


# 1. Lambda Correaltion
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

        return results


# 2. Goodman Kruskal tau Correlation
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

    return results

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



class Categorical_Association_Difference():
    @staticmethod
    def Categorical_Association_difference_from_data(params: dict) -> dict:
        
        column_1 = params["Column 1"]
        column_2 = params["Column 2"]
        column_3 = params["Column 3"]
        column_4 = params["Column 4"]

        confidence_level_percentages = params["Confidence Level"]

        confidence_level = confidence_level_percentages / 100
        Contingency_Table1 = Columns_to_Contingency(column_1, column_2)
        Contingency_Table2 = Columns_to_Contingency(column_3, column_4)

        output_lambda1 = (goodman_kruskal_lamda_correlation(Contingency_Table1, confidence_level))
        output_lambda2 = (goodman_kruskal_lamda_correlation(Contingency_Table2, confidence_level))

        output_Tau1 = Goodman_Kruskal_Tau(Contingency_Table1, confidence_level)
        output_Tau2 = Goodman_Kruskal_Tau(Contingency_Table2, confidence_level)

        Lambda_Rows1 = output_lambda1['Lambda Rows']
        Lambda_Columns1 = output_lambda1['Lambda Columns']
        Lambda_Symmetric1 = output_lambda1['Lambda Symmetric']
        Standard_Error_rows1 = output_lambda1['Random Method - Standard Error Rows']
        Standard_Error_columns1 = output_lambda1['Random Method - Standard Error Columns']
        Standard_Error_symmetric1 = output_lambda1['Random Method - Standard Error Symmetric']
        Lambda_Rows2 = output_lambda2['Lambda Rows']
        Lambda_Columns2 = output_lambda2['Lambda Columns']
        Lambda_Symmetric2 = output_lambda2['Lambda Symmetric']
        Standard_Error_rows2 = output_lambda2['Random Method - Standard Error Rows']
        Standard_Error_columns2 = output_lambda2['Random Method - Standard Error Columns']
        Standard_Error_symmetric2 = output_lambda2['Random Method - Standard Error Symmetric']

        Tau_Rows1 = output_Tau1['Goodman Kruskal Tau (Rows)']
        Tau_Columns1 = output_Tau1['Goodman Kruskal Tau (Columns)']
        Tau_Symmetric1 = output_Tau1['Goodman Kruskal Tau (Symmetric)']
        Standard_Error_rows1_tau = output_Tau1['Standard Error Rows']
        Standard_Error_columns1_tau = output_Tau1['Standard Error Coloumns']
        Standard_Error_symmetric1_tau = output_Tau1['Standard Error Symmetric']
        Tau_Rows2 = output_Tau2['Goodman Kruskal Tau (Rows)']
        Tau_Columns2 = output_Tau2['Goodman Kruskal Tau (Columns)']
        Tau_Symmetric2 = output_Tau2['Goodman Kruskal Tau (Symmetric)']
        Standard_Error_rows2_tau = output_Tau2['Standard Error Rows']
        Standard_Error_columns2_tau = output_Tau2['Standard Error Coloumns']
        Standard_Error_symmetric2_tau = output_Tau2['Standard Error Symmetric']
    

        # Significance for the Single Lambdas
        pvalue_rows_1 = output_lambda1["Random Method - P-Value Rows"]
        pvalue_rows_2 = output_lambda2["Random Method - P-Value Rows"]
        pvalue_columns_1 = output_lambda1["Random Method - P-Value Columns"]
        pvalue_columns_2 = output_lambda2["Random Method - P-Value Columns"]
        pvalue_symmetric_1 = output_lambda1["Random Method - P-Value Symmetric"]
        pvalue_symmetric_2 = output_lambda2["Random Method - P-Value Symmetric"]

        # Significance for the Single Taus
        pvalue_rows_1_tau = output_Tau1["p-value Goodman Kruskal Tau (Rows)"]
        pvalue_rows_2_tau = output_Tau2["p-value Goodman Kruskal Tau (Rows)"]
        pvalue_columns_1_tau = output_Tau1["p-value Goodman Kruskal Tau (Columns)"]
        pvalue_columns_2_tau = output_Tau2["p-value Goodman Kruskal Tau (Columns)"]
        pvalue_symmetric_1_tau = output_Tau1["p-value Goodman Kruskal Tau (Symmetric)"]
        pvalue_symmetric_2_tau = output_Tau2["p-value Goodman Kruskal Tau (Symmetric)"]


        Z_Statistic_Difference_rows = (Lambda_Rows1 - Lambda_Rows2) / np.sqrt(Standard_Error_rows1**2 + Standard_Error_rows2**2)
        p_value_difference_rows = norm.sf(abs(Z_Statistic_Difference_rows)) * 2 
        
        Z_Statistic_Difference_Columns = (Lambda_Columns1 - Lambda_Columns2) / np.sqrt(Standard_Error_columns1**2 + Standard_Error_columns2**2)
        p_value_difference_Columns = norm.sf(abs(Z_Statistic_Difference_Columns)) * 2 

        Z_Statistic_Difference_symmetric = (Lambda_Symmetric1 - Lambda_Symmetric2) / np.sqrt(Standard_Error_symmetric1**2 + Standard_Error_symmetric2**2)
        p_value_difference_symmetric = norm.sf(abs(Z_Statistic_Difference_symmetric)) * 2 

        Z_Statistic_Difference_rows_tau = (Tau_Rows1 - Tau_Rows2) / np.sqrt(Standard_Error_rows1_tau**2 + Standard_Error_rows2_tau**2)
        p_value_difference_rows_tau = norm.sf(abs(Z_Statistic_Difference_rows)) * 2 
        
        Z_Statistic_Difference_Columns_tau = (Tau_Columns1 - Tau_Columns2) / np.sqrt(Standard_Error_columns1_tau**2 + Standard_Error_columns2_tau**2)
        p_value_difference_Columns_tau = norm.sf(abs(Z_Statistic_Difference_Columns)) * 2 

        Z_Statistic_Difference_symmetric_tau = (Tau_Symmetric1 - Tau_Symmetric2) / np.sqrt(Standard_Error_symmetric1_tau**2 + Standard_Error_symmetric2_tau**2)
        p_value_difference_symmetric_tau = norm.sf(abs(Z_Statistic_Difference_symmetric)) * 2 

        results = {}
        results["Lambda Rows 1"] =  Lambda_Rows1
        results["Lambda Rows 2"] =  Lambda_Rows2
        results["p-value Lambda Rows 1"] = pvalue_rows_1
        results["p-value Lambda Rows 2"] = pvalue_rows_2

        results["Difference Between Lambda Rows"] =  (Lambda_Rows1 - Lambda_Rows2)
        results["Standard Error of the Difference Rows"] =   np.sqrt(Standard_Error_rows1**2 + Standard_Error_rows1**2)
        results["Z-Statistic of the Difference Rows"] =  Z_Statistic_Difference_rows
        results["p-value Rows"] =  p_value_difference_rows

        results["Lambda Columns 1"] =  Lambda_Columns1
        results["Lambda Columns 2"] =  Lambda_Columns2
        results["p-value Lambda Columns 1"] = pvalue_columns_1
        results["p-value Lambda Columns 2"] = pvalue_columns_2
        results["Difference Between Lambda Columns"] =  (Lambda_Columns1 - Lambda_Columns2)
        results["Standard Error of the Difference Columns"] =   np.sqrt(Standard_Error_columns1**2 + Standard_Error_columns1**2)
        results["Z-Statistic of the Difference Columns"] =  Z_Statistic_Difference_Columns
        results["p-value Columns"] =  p_value_difference_Columns

        results["Lambda Symmetric 1"] =  Lambda_Symmetric1
        results["Lambda Symmetric 2"] =  Lambda_Symmetric2
        results["p-value Lambda Symmetric 1"] = pvalue_symmetric_1
        results["p-value Lambda Symmetric 2"] = pvalue_symmetric_2
        results["Difference Between Lambda Symmetric"] =  (Lambda_Symmetric1 - Lambda_Symmetric2)
        results["Standard Error of the Difference Symmetric"] =   np.sqrt(Standard_Error_symmetric1**2 + Standard_Error_symmetric1**2)
        results["Z-Statistic of the Difference Symmetric"] =  Z_Statistic_Difference_symmetric
        results["p-value Symmetric"] =  p_value_difference_symmetric

        results["Tau Rows 1"] =  Tau_Rows1
        results["Tau Rows 2"] =  Tau_Rows2
        results["p-value Tau Rows 1"] = pvalue_rows_1
        results["p-value Tau Rows 2"] = pvalue_rows_2

        results["Difference Between Tau Rows"] =  (Tau_Rows1 - Tau_Rows2)
        results["Standard Error of the Difference Rows Tau"] =   np.sqrt(Standard_Error_rows1**2 + Standard_Error_rows1**2)
        results["Z-Statistic of the Difference Rows Tau"] =  Z_Statistic_Difference_rows_tau
        results["p-value Rows Tau"] =  p_value_difference_rows_tau

        results["Tau Columns 1"] =  Tau_Columns1
        results["Tau Columns 2"] =  Tau_Columns2
        results["p-value Tau Columns 1"] = pvalue_columns_1
        results["p-value Tau Columns 2"] = pvalue_columns_2
        
        results["Difference Between Tau Columns"] =  (Tau_Columns1 - Tau_Columns2)
        results["Standard Error of the Difference Columns Tau"] =   np.sqrt(Standard_Error_columns1**2 + Standard_Error_columns1**2)
        results["Z-Statistic of the Difference Columns Tau"] =  Z_Statistic_Difference_Columns_tau
        results["p-value Columns Tau"] =  p_value_difference_Columns_tau

        results["Tau Symmetric 1"] =  Tau_Symmetric1
        results["Tau Symmetric 2"] =  Tau_Symmetric2
        results["p-value Tau Symmetric 1"] = pvalue_symmetric_1
        results["p-value Tau Symmetric 2"] = pvalue_symmetric_2
        results["Difference Between Tau Symmetric Tau"] =  (Tau_Symmetric1 - Tau_Symmetric2)
        results["Standard Error of the Difference Symmetric Tau"] =   np.sqrt(Standard_Error_symmetric1**2 + Standard_Error_symmetric1**2)
        results["Z-Statistic of the Difference Symmetric Tau"] =  Z_Statistic_Difference_symmetric_tau
        results["p-value Symmetric Tau"] =  p_value_difference_symmetric_tau


        return results


    @staticmethod
    def Categorical_Association_difference_from_Contingency_Table(params: dict) -> dict:
        
        Contingency_Table1 = params["Contingency Table 1"]
        Contingency_Table2 = params["Contingency Table 2"]
        Contingency_Table2 = params["Confidence Level"]


        confidence_level_percentages = params["Confidence Level"]
        confidence_level = confidence_level_percentages / 100


        output_lambda1 = (goodman_kruskal_lamda_correlation(Contingency_Table1, confidence_level))
        output_lambda2 = (goodman_kruskal_lamda_correlation(Contingency_Table2, confidence_level))

        output_Tau1 = Goodman_Kruskal_Tau(Contingency_Table1, confidence_level)
        output_Tau2 = Goodman_Kruskal_Tau(Contingency_Table2, confidence_level)

        Lambda_Rows1 = output_lambda1['Lambda Rows']
        Lambda_Columns1 = output_lambda1['Lambda Columns']
        Lambda_Symmetric1 = output_lambda1['Lambda Symmetric']
        Standard_Error_rows1 = output_lambda1['Random Method - Standard Error Rows']
        Standard_Error_columns1 = output_lambda1['Random Method - Standard Error Columns']
        Standard_Error_symmetric1 = output_lambda1['Random Method - Standard Error Symmetric']
        Lambda_Rows2 = output_lambda2['Lambda Rows']
        Lambda_Columns2 = output_lambda2['Lambda Columns']
        Lambda_Symmetric2 = output_lambda2['Lambda Symmetric']
        Standard_Error_rows2 = output_lambda2['Random Method - Standard Error Rows']
        Standard_Error_columns2 = output_lambda2['Random Method - Standard Error Columns']
        Standard_Error_symmetric2 = output_lambda2['Random Method - Standard Error Symmetric']

        Tau_Rows1 = output_Tau1['Goodman Kruskal Tau (Rows)']
        Tau_Columns1 = output_Tau1['Goodman Kruskal Tau (Columns)']
        Tau_Symmetric1 = output_Tau1['Goodman Kruskal Tau (Symmetric)']
        Standard_Error_rows1_tau = output_Tau1['Standard Error Rows']
        Standard_Error_columns1_tau = output_Tau1['Standard Error Coloumns']
        Standard_Error_symmetric1_tau = output_Tau1['Standard Error Symmetric']
        Tau_Rows2 = output_Tau2['Goodman Kruskal Tau (Rows)']
        Tau_Columns2 = output_Tau2['Goodman Kruskal Tau (Columns)']
        Tau_Symmetric2 = output_Tau2['Goodman Kruskal Tau (Symmetric)']
        Standard_Error_rows2_tau = output_Tau2['Standard Error Rows']
        Standard_Error_columns2_tau = output_Tau2['Standard Error Coloumns']
        Standard_Error_symmetric2_tau = output_Tau2['Standard Error Symmetric']
    

        # Significance for the Single Lambdas
        pvalue_rows_1 = output_lambda1["Random Method - P-Value Rows"]
        pvalue_rows_2 = output_lambda2["Random Method - P-Value Rows"]
        pvalue_columns_1 = output_lambda1["Random Method - P-Value Columns"]
        pvalue_columns_2 = output_lambda2["Random Method - P-Value Columns"]
        pvalue_symmetric_1 = output_lambda1["Random Method - P-Value Symmetric"]
        pvalue_symmetric_2 = output_lambda2["Random Method - P-Value Symmetric"]

        # Significance for the Single Taus
        pvalue_rows_1_tau = output_Tau1["p-value Goodman Kruskal Tau (Rows)"]
        pvalue_rows_2_tau = output_Tau2["p-value Goodman Kruskal Tau (Rows)"]
        pvalue_columns_1_tau = output_Tau1["p-value Goodman Kruskal Tau (Columns)"]
        pvalue_columns_2_tau = output_Tau2["p-value Goodman Kruskal Tau (Columns)"]
        pvalue_symmetric_1_tau = output_Tau1["p-value Goodman Kruskal Tau (Symmetric)"]
        pvalue_symmetric_2_tau = output_Tau2["p-value Goodman Kruskal Tau (Symmetric)"]


        Z_Statistic_Difference_rows = (Lambda_Rows1 - Lambda_Rows2) / np.sqrt(Standard_Error_rows1**2 + Standard_Error_rows2**2)
        p_value_difference_rows = norm.sf(abs(Z_Statistic_Difference_rows)) * 2 
        
        Z_Statistic_Difference_Columns = (Lambda_Columns1 - Lambda_Columns2) / np.sqrt(Standard_Error_columns1**2 + Standard_Error_columns2**2)
        p_value_difference_Columns = norm.sf(abs(Z_Statistic_Difference_Columns)) * 2 

        Z_Statistic_Difference_symmetric = (Lambda_Symmetric1 - Lambda_Symmetric2) / np.sqrt(Standard_Error_symmetric1**2 + Standard_Error_symmetric2**2)
        p_value_difference_symmetric = norm.sf(abs(Z_Statistic_Difference_symmetric)) * 2 

        Z_Statistic_Difference_rows_tau = (Tau_Rows1 - Tau_Rows2) / np.sqrt(Standard_Error_rows1_tau**2 + Standard_Error_rows2_tau**2)
        p_value_difference_rows_tau = norm.sf(abs(Z_Statistic_Difference_rows)) * 2 
        
        Z_Statistic_Difference_Columns_tau = (Tau_Columns1 - Tau_Columns2) / np.sqrt(Standard_Error_columns1_tau**2 + Standard_Error_columns2_tau**2)
        p_value_difference_Columns_tau = norm.sf(abs(Z_Statistic_Difference_Columns)) * 2 

        Z_Statistic_Difference_symmetric_tau = (Tau_Symmetric1 - Tau_Symmetric2) / np.sqrt(Standard_Error_symmetric1_tau**2 + Standard_Error_symmetric2_tau**2)
        p_value_difference_symmetric_tau = norm.sf(abs(Z_Statistic_Difference_symmetric)) * 2 

        results = {}
        results["Lambda Rows 1"] =  Lambda_Rows1
        results["Lambda Rows 2"] =  Lambda_Rows2
        results["p-value Lambda Rows 1"] = pvalue_rows_1
        results["p-value Lambda Rows 2"] = pvalue_rows_2

        results["Difference Between Lambda Rows"] =  (Lambda_Rows1 - Lambda_Rows2)
        results["Standard Error of the Difference Rows"] =   np.sqrt(Standard_Error_rows1**2 + Standard_Error_rows1**2)
        results["Z-Statistic of the Difference Rows"] =  Z_Statistic_Difference_rows
        results["p-value Rows"] =  p_value_difference_rows

        results["Lambda Columns 1"] =  Lambda_Columns1
        results["Lambda Columns 2"] =  Lambda_Columns2
        results["p-value Lambda Columns 1"] = pvalue_columns_1
        results["p-value Lambda Columns 2"] = pvalue_columns_2
        results["Difference Between Lambda Columns"] =  (Lambda_Columns1 - Lambda_Columns2)
        results["Standard Error of the Difference Columns"] =   np.sqrt(Standard_Error_columns1**2 + Standard_Error_columns1**2)
        results["Z-Statistic of the Difference Columns"] =  Z_Statistic_Difference_Columns
        results["p-value Columns"] =  p_value_difference_Columns

        results["Lambda Symmetric 1"] =  Lambda_Symmetric1
        results["Lambda Symmetric 2"] =  Lambda_Symmetric2
        results["p-value Lambda Symmetric 1"] = pvalue_symmetric_1
        results["p-value Lambda Symmetric 2"] = pvalue_symmetric_2
        results["Difference Between Lambda Symmetric"] =  (Lambda_Symmetric1 - Lambda_Symmetric2)
        results["Standard Error of the Difference Symmetric"] =   np.sqrt(Standard_Error_symmetric1**2 + Standard_Error_symmetric1**2)
        results["Z-Statistic of the Difference Symmetric"] =  Z_Statistic_Difference_symmetric
        results["p-value Symmetric"] =  p_value_difference_symmetric

        results["Tau Rows 1"] =  Tau_Rows1
        results["Tau Rows 2"] =  Tau_Rows2
        results["p-value Tau Rows 1"] = pvalue_rows_1
        results["p-value Tau Rows 2"] = pvalue_rows_2

        results["Difference Between Tau Rows"] =  (Tau_Rows1 - Tau_Rows2)
        results["Standard Error of the Difference Rows Tau"] =   np.sqrt(Standard_Error_rows1**2 + Standard_Error_rows1**2)
        results["Z-Statistic of the Difference Rows Tau"] =  Z_Statistic_Difference_rows_tau
        results["p-value Rows Tau"] =  p_value_difference_rows_tau

        results["Tau Columns 1"] =  Tau_Columns1
        results["Tau Columns 2"] =  Tau_Columns2
        results["p-value Tau Columns 1"] = pvalue_columns_1
        results["p-value Tau Columns 2"] = pvalue_columns_2
        
        results["Difference Between Tau Columns"] =  (Tau_Columns1 - Tau_Columns2)
        results["Standard Error of the Difference Columns Tau"] =   np.sqrt(Standard_Error_columns1**2 + Standard_Error_columns1**2)
        results["Z-Statistic of the Difference Columns Tau"] =  Z_Statistic_Difference_Columns_tau
        results["p-value Columns Tau"] =  p_value_difference_Columns_tau

        results["Tau Symmetric 1"] =  Tau_Symmetric1
        results["Tau Symmetric 2"] =  Tau_Symmetric2
        results["p-value Tau Symmetric 1"] = pvalue_symmetric_1
        results["p-value Tau Symmetric 2"] = pvalue_symmetric_2
        results["Difference Between Tau Symmetric Tau"] =  (Tau_Symmetric1 - Tau_Symmetric2)
        results["Standard Error of the Difference Symmetric Tau"] =   np.sqrt(Standard_Error_symmetric1**2 + Standard_Error_symmetric1**2)
        results["Z-Statistic of the Difference Symmetric Tau"] =  Z_Statistic_Difference_symmetric_tau
        results["p-value Symmetric Tau"] =  p_value_difference_symmetric_tau


        return results






