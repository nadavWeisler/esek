
import numpy as np
import pandas as pd
from scipy.stats import norm
from pingouin import kruskal 
import math 


def Freemans_Theta(Contingency_Table):
    contingency_table = pd.DataFrame(Contingency_Table)
    row_names = contingency_table.index
    vectors = {}
    contrasts = {}

    for i in range(len(row_names)):
        for j in range(i + 1, len(row_names)):
            row1 = contingency_table.loc[row_names[i]].values
            row2 = contingency_table.loc[row_names[j]].values
            vector = np.multiply.outer(row1, row2)
            key = f'{row_names[i]}_{row_names[j]}'
            vectors[key] = vector

    for i in range(len(row_names)):
        for j in range(i + 1, len(row_names)):
            vector = vectors[f'{row_names[i]}_{row_names[j]}']
            contrast = np.sum(np.triu(vector)) - np.sum(np.tril(vector))
            key = f'{row_names[i]}_{row_names[j]}'
            contrasts[key] = contrast

    Delta = np.sum(np.abs(list(contrasts.values())))

    row_sums = contingency_table.sum(axis=1).values
    T2 = np.sum(np.triu(np.outer(row_sums, row_sums), k=1)) # type: ignore
    Theta = Delta / T2

    return Theta

def Rank_Biserial_Correlation(nominal_variable, ordinal_variable, confidence_level):
    unique_values = set(nominal_variable)
    if len(unique_values) != 2:
        return "There are more than two values, and therefore rank biserial correlation could not be calculated."
    else:
        value_map = {value: idx for idx, value in enumerate(unique_values)}
        binary_nominal = [value_map[item] for item in nominal_variable]
        count_A_greater_than_B = sum(1 for a, b in zip(binary_nominal, ordinal_variable) if a == 0 for a2, b2 in zip(binary_nominal, ordinal_variable) if a2 == 1 and b < b2)
        count_B_greater_than_A = sum(1 for a, b in zip(binary_nominal, ordinal_variable) if a == 0 for a2, b2 in zip(binary_nominal, ordinal_variable) if a2 == 1 and b > b2)
        Sample_Size = len(nominal_variable)
        Sample_Size_1 = binary_nominal.count(0)
        Sample_Size_2 = binary_nominal.count(1)
        Number_of_Comparisons = Sample_Size_1 * Sample_Size_2
        
        # Rank Biserial Correlation and its confidence Intervals
        Rank_Biserial_Corrlation = ((count_A_greater_than_B / Number_of_Comparisons) - (count_B_greater_than_A / Number_of_Comparisons)) / 2
        Standrd_Error_RBC = np.sqrt((Sample_Size_1+Sample_Size_2+1)/(3*Sample_Size_1+Sample_Size_2)) #see totser package formulae for paired data as well
        Z_Critical_Value = norm.ppf((1-confidence_level) + ((confidence_level) / 2))
        Lower_CI_Rank_Biserial_Correlation = max(math.tanh(math.atanh(Rank_Biserial_Corrlation) - Z_Critical_Value * Standrd_Error_RBC),-1)
        Upper_CI_Rank_Biserial_Correlation = min(math.tanh(math.atanh(Rank_Biserial_Corrlation) + Z_Critical_Value * Standrd_Error_RBC),1)
 
        return Rank_Biserial_Corrlation, Lower_CI_Rank_Biserial_Correlation, Upper_CI_Rank_Biserial_Correlation

class Nominal_by_Ordinal():

    @staticmethod
    def Nominal_By_Ordinal_from_Contingency_Table(params: dict) -> dict:
        
        # Set Params:
        Contingency_Table = params["Contingency Table"]
        n_boots = params["Number of Bootstrapping Samples"]
        confidence_level_percentages = params["Confidence Level"]
        
        # Preperations: 
        confidence_level = confidence_level_percentages / 100

        # Converting the contingency table into a ordinal nominal table format
        df = pd.DataFrame({'Group': np.repeat(np.arange(Contingency_Table.shape[0]), Contingency_Table.shape[1]),'Response': Contingency_Table.flatten()})
        df['Group'] = np.array([chr(97 + group) for group in df['Group']])
        df['Cumulative Frequency'] = df.groupby('Group').cumcount() + 1
        df = pd.DataFrame(df[['Group', 'Cumulative Frequency']].values.repeat(df['Response'], axis=0),columns=['Nominal_Variable', 'Ordinal_Variable'])
        df['Ordinal_Variable'] = pd.to_numeric(df['Ordinal_Variable'])

        sample_size = len(df['Ordinal_Variable'])
        Number_of_Levels_Nominal = len(np.unique(df['Nominal_Variable']))

        # 1. Epsilon Square and Kruskal-Wallis Test
        Kruskal_Wallis_Results = kruskal(data=df, dv='Ordinal_Variable', between='Nominal_Variable').values
        df_kruskal = Kruskal_Wallis_Results[0,1]
        H_Statistic = Kruskal_Wallis_Results[0,2]
        Kruskal_Wallis_p_Value = Kruskal_Wallis_Results[0,3]
        Epsilon_Sqaure = H_Statistic / (sample_size - 1)

        # Bootstrapping Including CI's
        Krushkal_Stats_Bootstrapping = [kruskal(data=df.sample(frac=1, replace=True), dv='Ordinal_Variable', between='Nominal_Variable').values[0,2] for _ in range(n_boots)]
        Upper_ci_Epsilon = np.percentile(Krushkal_Stats_Bootstrapping, 100 - ((100 - confidence_level_percentages) / 2))
        Lower_ci_Epsilon = np.percentile(Krushkal_Stats_Bootstrapping, ((100 - confidence_level_percentages) / 2))
        Epsilon_Upper_CI = Upper_ci_Epsilon / (sample_size - 1)
        Epsilon_Lower_CI = Lower_ci_Epsilon / (sample_size - 1)

        # 2. Freeman's Theta and Bootstrapping CI's
        FreemansTheta = Freemans_Theta(Contingency_Table)
        
        fremmans_theta_bootstrapp = []
        for _ in range(n_boots):
            bootstrap_sample = df.sample(frac=1, replace=True)
            contingency_table = pd.crosstab(bootstrap_sample['Nominal_Variable'], bootstrap_sample['Ordinal_Variable'])
            theta = Freemans_Theta(contingency_table)
            fremmans_theta_bootstrapp.append(theta)
        Lower_CI_Freemans_Theta = np.percentile(fremmans_theta_bootstrapp, (100 - confidence_level_percentages) / 2) 
        Upper_CI_Freemans_Theta = np.percentile(fremmans_theta_bootstrapp, 100 - ((100 - confidence_level_percentages) / 2) )

        Upper_ci_Epsilon = np.percentile(Krushkal_Stats_Bootstrapping, 100 - ((100 - confidence_level_percentages) / 2))
        Lower_ci_Epsilon = np.percentile(Krushkal_Stats_Bootstrapping, ((100 - confidence_level_percentages) / 2))
        Epsilon_Upper_CI = Upper_ci_Epsilon / (sample_size - 1)
        Epsilon_Lower_CI = Lower_ci_Epsilon / (sample_size - 1)        
        
        # 3. Rank Biserial Correlation
        if Number_of_Levels_Nominal == 2:
           Rank_Biserial_Corrlation, Lower_CI_Rank_Biserial_Correlation, Upper_CI_Rank_Biserial_Correlation = Rank_Biserial_Correlation(df['Nominal_Variable'], df["Ordinal_Variable"], confidence_level)

        results = {}
        results["H Statistic"] =  H_Statistic
        results["Degrees of Freedom of the Kruskal Wallis Test"] =  df_kruskal
        results["p-value of the Kruskal Wallis Test"] =  Kruskal_Wallis_p_Value
        results["Epsilon Square"] =  Epsilon_Sqaure
        results["Lower Ci of Epsilon Square"] =  Epsilon_Lower_CI
        results["Upper Ci of Epsilon Square"] =  Epsilon_Upper_CI
        results["Freeman's Theta"] =  FreemansTheta
        results["Freeman's Theta Lower CI"] =  Lower_CI_Freemans_Theta
        results["Freeman's Theta Upper Ci"] =  Upper_CI_Freemans_Theta
        formatted_p_value_kruskal = "{:.3f}".format(Kruskal_Wallis_p_Value).lstrip('0') if Kruskal_Wallis_p_Value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Epsilon Square"] = " \033[3mH\033[0m({}) = {}, {}{}, \033[3m\u03B5\u00B2\033[0m = {}, {}% CI(bootsrapping) [{}, {}]".format(df_kruskal, ((int( round(H_Statistic,3)) if float(round(H_Statistic,3)).is_integer() else round(H_Statistic,3))), '\033[3mp = \033[0m' if Kruskal_Wallis_p_Value >= 0.001 else '', formatted_p_value_kruskal, (('-' if str(Epsilon_Sqaure).startswith('-') else '') + str(round(Epsilon_Sqaure,3)).lstrip('-').lstrip('0') or '0'), (confidence_level_percentages), (('-' if str(Epsilon_Lower_CI).startswith('-') else '') + str(round(Epsilon_Lower_CI,3)).lstrip('-').lstrip('0') or '0'), (('-' if str(Epsilon_Upper_CI).startswith('-') else '') + str(round(Epsilon_Upper_CI,3)).lstrip('-').lstrip('0') or '0'))

        if Number_of_Levels_Nominal == 2:
            results["Rank Biserial Correlation"] =  Rank_Biserial_Corrlation
            results["Lower CI Rank Biserial Correlation"] =  Lower_CI_Rank_Biserial_Correlation
            results["Upper CI Rank Biserial Correlation"] =  Upper_CI_Rank_Biserial_Correlation
    
        return results


    @staticmethod
    def Nominal_By_Ordinal_from_Data(params: dict) -> dict:

        # Set params
        Nominal_Var = params["Nominal Variable"]
        Ordinal_Var = params["Ordinal Variable"]
        n_boots = params["Number of Bootstrapping Samples"]
        confidence_level_percentages = params["Confidence Level"]

        # Preperations: 
        confidence_level = confidence_level_percentages / 100
        df = pd.DataFrame({'Ordinal_Variable': Ordinal_Var, 'Nominal_Variable': Nominal_Var})
        Contingency_Table = pd.crosstab(df['Nominal_Variable'], df["Ordinal_Variable"])
        sample_size = len(df['Ordinal_Variable'])
        Number_of_Levels = len(np.unique(df['Nominal_Variable']))

        # 1. Epsilon Square and Kruskal-Wallis Test
        Kruskal_Wallis_Results = kruskal(data=df, dv='Ordinal_Variable', between='Nominal_Variable').values
        df_kruskal = Kruskal_Wallis_Results[0,1]
        H_Statistic = Kruskal_Wallis_Results[0,2]
        Kruskal_Wallis_p_Value = Kruskal_Wallis_Results[0,3]
        Epsilon_Sqaure = H_Statistic / (sample_size - 1)

        # Bootstrapping Including CI's
        Krushkal_Stats_Bootstrapping = [kruskal(data=df.sample(frac=1, replace=True), dv='Ordinal_Variable', between='Nominal_Variable').values[0,2] for _ in range(n_boots)]
        Upper_ci_Epsilon = np.percentile(Krushkal_Stats_Bootstrapping, 100 - ((100 - confidence_level_percentages) / 2))
        Lower_ci_Epsilon = np.percentile(Krushkal_Stats_Bootstrapping, ((100 - confidence_level_percentages) / 2))
        Epsilon_Upper_CI = Upper_ci_Epsilon / (sample_size - 1)
        Epsilon_Lower_CI = Lower_ci_Epsilon / (sample_size - 1)

        # 2. Freeman's Theta and Bootstrapping CI's
        FreemansTheta = Freemans_Theta(Contingency_Table)
        
        fremmans_theta_bootstrapp = []
        for _ in range(n_boots):
            bootstrap_sample = df.sample(frac=1, replace=True)
            contingency_table = pd.crosstab(bootstrap_sample['Nominal_Variable'], bootstrap_sample['Ordinal_Variable'])
            theta = Freemans_Theta(contingency_table)
            fremmans_theta_bootstrapp.append(theta)
        Lower_CI_Freemans_Theta = np.percentile(fremmans_theta_bootstrapp, (100 - confidence_level_percentages) / 2) 
        Upper_CI_Freemans_Theta = np.percentile(fremmans_theta_bootstrapp, 100 - ((100 - confidence_level_percentages) / 2) )

        Upper_ci_Epsilon = np.percentile(Krushkal_Stats_Bootstrapping, 100 - ((100 - confidence_level_percentages) / 2))
        Lower_ci_Epsilon = np.percentile(Krushkal_Stats_Bootstrapping, ((100 - confidence_level_percentages) / 2))
        Epsilon_Upper_CI = Upper_ci_Epsilon / (sample_size - 1) 
        Epsilon_Lower_CI = Lower_ci_Epsilon / (sample_size - 1)

        # 3. Rank Biserial Correlation
        if Number_of_Levels == 2:
           Rank_Biserial_Corrlation, Lower_CI_Rank_Biserial_Correlation, Upper_CI_Rank_Biserial_Correlation  = Rank_Biserial_Correlation(df['Nominal_Variable'], df["Ordinal_Variable"], confidence_level)

        results = {}
        results["H Statistic"] =  H_Statistic
        results["Degrees of Freedom of the Kruskal Wallis Test"] =  df_kruskal
        results["p-value of the Kruskal Wallis Test"] =  Kruskal_Wallis_p_Value
        results["Epsilon Square"] =  Epsilon_Sqaure
        results["Lower Ci of Epsilon Square"] =  Epsilon_Lower_CI
        results["Upper Ci of Epsilon Square"] =  Epsilon_Upper_CI
        results["Freeman's Theta"] =  FreemansTheta
        results["Freeman's Theta Lower CI"] =  Lower_CI_Freemans_Theta
        results["Freeman's Theta Upper Ci"] =  Upper_CI_Freemans_Theta
        formatted_p_value_kruskal = "{:.3f}".format(Kruskal_Wallis_p_Value).lstrip('0') if Kruskal_Wallis_p_Value >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Epsilon Square"] = " \033[3mH\033[0m({}) = {}, {}{}, \033[3m\u03B5\u00B2\033[0m = {}, {}% CI(bootsrapping) [{}, {}]".format(df_kruskal, ((int( round(H_Statistic,3)) if float(round(H_Statistic,3)).is_integer() else round(H_Statistic,3))), '\033[3mp = \033[0m' if Kruskal_Wallis_p_Value >= 0.001 else '', formatted_p_value_kruskal, (('-' if str(Epsilon_Sqaure).startswith('-') else '') + str(round(Epsilon_Sqaure,3)).lstrip('-').lstrip('0') or '0'), (confidence_level_percentages), (('-' if str(Epsilon_Lower_CI).startswith('-') else '') + str(round(Epsilon_Lower_CI,3)).lstrip('-').lstrip('0') or '0'), (('-' if str(Epsilon_Upper_CI).startswith('-') else '') + str(round(Epsilon_Upper_CI,3)).lstrip('-').lstrip('0') or '0'))

        if Number_of_Levels == 2:
            results["Rank Biserial Correlation"] =  Rank_Biserial_Corrlation
            results["Lower CI Rank Biserial Correlation"] =  Lower_CI_Rank_Biserial_Correlation
            results["Upper CI Rank Biserial Correlation"] =  Upper_CI_Rank_Biserial_Correlation
    

        return results
