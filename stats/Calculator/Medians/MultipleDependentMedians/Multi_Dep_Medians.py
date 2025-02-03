

###############################################
# Effect Size for Multiple Dependent Medians ##
###############################################

import numpy as np
import math
from scipy.stats import norm, nct, t, trim_mean, median_abs_deviation
import pandas as pd


# Relevant functions for multiple dependent groups

# 1. Winsorized Variance
def WinsorizedVariance(x, trimming_level=0.2):
    y = np.sort(x)
    n = len(x)
    ibot = int(np.floor(trimming_level * n)) + 1
    itop = n - ibot + 1
    xbot = y[ibot-1] 
    xtop = y[itop-1]
    y = np.where(y <= xbot, xbot, y)
    y = np.where(y >= xtop, xtop, y)
    winvar = np.std(y, ddof=1)**2
    return winvar

class Multiple_Dependent_Medians():
    @staticmethod
    def Multiple_Dependent_Medians_From_Data (params: dict) -> dict:
        
        # Set params
        confidence_level_percentages = params["Confidence Level"]
        trimmimg_level = params["Trimming Level"]

        df = pd.concat([pd.DataFrame(value, columns=[key]) for key, value in params.items() if key.startswith("Column")], axis = 1)
        
        # Descreptive Statistics 
        trimmed_mean_values = [trim_mean(df[col], proportiontocut=0.2) for col in df.columns]

        iqr_values = {col: np.percentile(df[col], 75) - np.percentile(df[col], 25) for col in df.columns}
        median_values = {col: np.median(df[col]) for col in df.columns}
        mad_values = {col: median_abs_deviation(df[col]) for col in df.columns}

        # Make a nice descreotie statistics table

        table_data = {
            'Column': list(df.columns),
            'IQR': [iqr_values[col] for col in df.columns],
            'MAD': [mad_values[col] for col in df.columns],
            'Median': [median_values[col] for col in df.columns],
            'Trimmed Mean': trimmed_mean_values
}

        Descreptive_Statistic_Table = pd.DataFrame(table_data)

        # One Way Robust Repeatd Measures ANOVA

        sample_size = df.shape[0]
        g = np.floor(sample_size*trimmimg_level)
        h = sample_size - 2* g
        J = df.shape[1]

        grand_trim_mean = np.mean(np.array(trimmed_mean_values))
        gc = h * np.sum((np.array(grand_trim_mean) - trimmed_mean_values)**2)

        results = {}
        
        results['Descreptive Statistics Table'] = ""
        results[''] = Descreptive_Statistic_Table

        results['_____________________________'] = ""
        results['Sample Size'] = sample_size
        results['Number '] = g
        results['Number 2 '] = J
        results['Mean of the Trimmed Means'] = trimmed_mean_values
        results['Gc'] = gc
        results["df"] = df



        return results

