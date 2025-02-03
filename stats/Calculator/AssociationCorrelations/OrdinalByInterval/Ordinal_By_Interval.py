

### Relevant Fnctions for Ordinal By Ordinal Correlation
from collections import Counter
import numpy as np
from scipy.stats import norm, spearmanr, rankdata, weightedtau, median_abs_deviation, t
import math
import pandas as pd
import pingouin as pg

# 1. Skipped Correlation (From pinguine)
def skipped_Correlation(x, y, confidence_level):
    from pingouin.utils import _is_sklearn_installed

    _is_sklearn_installed(raise_error=True)
    from scipy.stats import chi2
    from sklearn.covariance import MinCovDet

    sample_size = len(x)
    X = np.column_stack((x, y))
    nrows, ncols = X.shape
    gval = np.sqrt(chi2.ppf(0.975, 2))
    center = MinCovDet(random_state=42).fit(X).location_
    B = X - center
    bot = (B**2).sum(axis=1)
    dis = np.zeros(shape=(nrows, nrows))
    for i in np.arange(nrows):
        if bot[i] != 0:  # Avoid division by zero error
            dis[i, :] = np.linalg.norm(B.dot(B[i, :, None]) * B[i, :] / bot[i], axis=1)

    def ideal_forth_interquartile_range(x):
        n = len(x)
        j = int(np.floor(n / 4 + 5 / 12))
        y = np.sort(x)
        g = (n / 4) - j + (5 / 12)
        low = (1 - g) * y[j - 1] + g * y[j]
        k = n - j + 1
        up = (1 - g) * y[k - 1] + g * y[k - 2]
        return up - low

    # One can either use the MAD or the IQR (see Wilcox 2012)
    iqr = np.apply_along_axis(ideal_forth_interquartile_range, 1, dis)
    thresh_IQR = np.median(dis, axis=1) + gval * iqr
    outliers_iqr = np.apply_along_axis(np.greater, 0, dis, thresh_IQR).any(axis=0)
    mad = np.apply_along_axis(median_abs_deviation, 1, dis)
    thresh_mad = np.median(dis, axis=1) + gval * mad
    outliers_mad = np.apply_along_axis(np.greater, 0, dis, thresh_mad).any(axis=0)

    
    # Compute correlation on remaining data
    Spearman_Correlation_IQR, pval_Skipped_Correlation_IQR = spearmanr(X[~outliers_iqr, 0], X[~outliers_iqr, 1])
    Spearman_Correlation_MAD, pval_Skipped_Correlation_MAD = spearmanr(X[~outliers_mad, 0], X[~outliers_mad, 1])

    # Confidence Intervals using Bonette Write Procedure
    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
    sample_size_skipped_iqr = len(X[~outliers_iqr, 0])
    sample_sie_skipped_mad = len(X[~outliers_mad, 0])
    Zr_IQR = 0.5 * np.log((1 + Spearman_Correlation_IQR**2) / (1 - Spearman_Correlation_IQR))
    Standard_Error_Spearman_IQR_Bonette_Wright = np.sqrt((1 + Spearman_Correlation_IQR**2 / 2) / (sample_size_skipped_iqr - 3))
    Zr_Upper_IQR = Zr_IQR + zcrit * Standard_Error_Spearman_IQR_Bonette_Wright
    Zr_lower_IQR = Zr_IQR - zcrit * Standard_Error_Spearman_IQR_Bonette_Wright

    Zr_MAD = 0.5 * np.log((1 + Spearman_Correlation_MAD**2) / (1 - Spearman_Correlation_MAD))
    Standard_Error_Spearman_MAD_Bonette_Wright = np.sqrt((1 + Spearman_Correlation_MAD**2 / 2) / (sample_sie_skipped_mad - 3))
    Zr_Upper_MAD = Zr_MAD + zcrit * Standard_Error_Spearman_MAD_Bonette_Wright
    Zr_lower_MAD = Zr_MAD - zcrit * Standard_Error_Spearman_MAD_Bonette_Wright

    lower_ci_skipped_iqr = (math.exp(2 * Zr_lower_IQR) - 1) / (math.exp(2 * Zr_lower_IQR) + 1)
    upper_ci_skipped_iqr = (math.exp(2 * Zr_Upper_IQR) - 1) / (math.exp(2 * Zr_Upper_IQR) + 1)
    lower_ci_skipped_mad = (math.exp(2 * Zr_lower_MAD) - 1) / (math.exp(2 * Zr_lower_MAD) + 1)
    upper_ci_skipped_mad = (math.exp(2 * Zr_Upper_MAD) - 1) / (math.exp(2 * Zr_Upper_MAD) + 1)

    results = {}

    results["Skipped Correlation IQR based"] = Spearman_Correlation_IQR
    results["Skipped Correlation MAD based"] = Spearman_Correlation_MAD
    results["Skipped Correlation IQR based p-value"] = pval_Skipped_Correlation_IQR
    results["Skipped Correlation MAD based p-value"] = pval_Skipped_Correlation_MAD
    results["Skipped Correlation IQR based CI's"] = [lower_ci_skipped_iqr, upper_ci_skipped_iqr]
    results["Skipped Correlation MAD based CI's"] = [lower_ci_skipped_mad, upper_ci_skipped_mad]

    result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
    return result_str


# 2. Spearman Correlation
def Spearman_Correlation(x,y,confidence_level): 

    # Preperation
    ranked_x = rankdata(x)
    ranked_y = rankdata(y)
    mean_sample_1 = np.mean(ranked_x)
    mean_sample_2 = np.mean(ranked_y)
    sample_size = len(x)

    # Spearman Correlation 
    spearman_rho_p_value = spearmanr(x,y)[1]
    Spearman = (np.sum ((ranked_x- mean_sample_1) * (ranked_y - mean_sample_2))) / (np.sqrt(np.sum((ranked_x- mean_sample_1)**2)) * np.sqrt(np.sum((ranked_y- mean_sample_2)**2)))
    fisher_Zrho = math.atanh(Spearman)

    # Calculate the Standard Errors
    Fieller_Standard_Error = np.sqrt(1.06 / (sample_size-3)) # 1957
    Caruso_and_Cliff = np.sqrt(1 / (sample_size - 2)) +  (abs(fisher_Zrho)/(6*sample_size + 4 *np.sqrt(sample_size)))# 1997
    Bonett_Wright_Standard_Error = (1 + Spearman / 2) / (sample_size - 3) # 2000 - I use this as the default

    # Confidence Interval Based on the Different Standard Errors (Default is Bonette Wright)
    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
    tcrit = t.ppf(1 - (1 - confidence_level) / 2, sample_size-2)
    Lower_ci_BW = math.tanh(fisher_Zrho - zcrit * Bonett_Wright_Standard_Error)
    Upper_ci_BW = math.tanh(fisher_Zrho + zcrit * Bonett_Wright_Standard_Error)
    Lower_ci_Fieller = math.tanh(fisher_Zrho - zcrit * Fieller_Standard_Error)
    Upper_ci_Fieller = math.tanh(fisher_Zrho + zcrit * Fieller_Standard_Error)
    Lower_ci_CC = math.tanh(fisher_Zrho - zcrit * Caruso_and_Cliff)
    Upper_ci_CC = math.tanh(fisher_Zrho + zcrit * Caruso_and_Cliff)
    Lower_ci_Woods = math.tanh(fisher_Zrho - tcrit * (np.sqrt(1/(sample_size-3))))
    Upper_ci_Woods = math.tanh(fisher_Zrho + tcrit * (np.sqrt(1/(sample_size-3))))
    Lower_ci_Fisher = math.tanh(fisher_Zrho - zcrit * (np.sqrt(1/(sample_size-3))))
    Upper_ci_Fisher = math.tanh(fisher_Zrho + zcrit * (np.sqrt(1/(sample_size-3))))

    # Spearman Corrected for ties Oyeka and Nwankwo Chike 2014 
    Expected_ranks_x = x.argsort().argsort() + 1
    Expected_ranks_y = y.argsort().argsort() + 1
    Difference_Rank_x = Expected_ranks_x - ranked_x
    Difference_Rank_y = Expected_ranks_y - ranked_y
    Product_x = Expected_ranks_x * Difference_Rank_x
    Product_y = Expected_ranks_y * Difference_Rank_y
    Di_x = Difference_Rank_x**2 /2
    Di_y = Difference_Rank_y**2 /2
    pi_x = non_zero_count = sum(1 for element in Difference_Rank_x if element != 0) / sample_size
    pi_y = non_zero_count = sum(1 for element in Difference_Rank_y if element != 0) / sample_size
    Multi_Ranked = ranked_x * ranked_y
    term_x = (((sample_size*(sample_size**2-1))/12) - (2*pi_x * (sum(Product_x)-sum(Di_x))))
    term_y = (((sample_size*(sample_size**2-1))/12) - (2*pi_y * (sum(Product_y)-sum(Di_y))))
    Numerator = np.sum(Multi_Ranked) - ((sample_size*(sample_size+1)**2) / 4)  
    Denomirator = np.sqrt(term_x*term_y)
    Corrected_Spearman = Numerator / Denomirator
    p_value_Oyeka = t.sf(abs((Corrected_Spearman*((sample_size-2) / (1 - Corrected_Spearman**2)))), (sample_size-2))

    # Confidence Intervals for Spearman Corrected Oyeka et al., 
    Bonett_Wright_Standard_Error_Oyeka = (1 + Corrected_Spearman / 2) / (sample_size - 3) 
    Lower_ci_Oyeka = math.tanh(math.atanh(Corrected_Spearman) - zcrit * Bonett_Wright_Standard_Error_Oyeka)
    Upper_ci_Oyeka = math.tanh(math.atanh(Corrected_Spearman) + zcrit * Bonett_Wright_Standard_Error_Oyeka)

    # Spearman Corrected for ties Taylor (1964)
    d_square = np.sum((ranked_x - ranked_y)**2)
    frequency_vector_x = np.array([count for count in Counter(x).values() if count > 1]) # return only repeating frequencies for ties correction 
    frequency_vector_y = np.array([count for count in Counter(y).values() if count > 1]) # return only repeating frequencies for ties correction 
    Tx = sum(frequency_vector_x**3-frequency_vector_x) / 12
    Ty = sum(frequency_vector_y**3-frequency_vector_y) / 12
    Spearman_Corrected_Taylor = 1 - (6 * (np.sum(d_square)+Tx+Ty)) / (sample_size*(sample_size**2-1))
    p_value_taylor = t.sf(abs((Corrected_Spearman*((sample_size-2) / (1 - Spearman_Corrected_Taylor**2)))), (sample_size-2))

    # Confidence Intervals for Spearman Corrected Taylor
    Bonett_Wright_Standard_Error_Taylor = (1 + Spearman_Corrected_Taylor / 2) / (sample_size - 3) 
    Lower_ci_Taylor = math.tanh(math.atanh(Spearman_Corrected_Taylor) - zcrit * Bonett_Wright_Standard_Error_Taylor)
    Upper_ci_Taylor = math.tanh(math.atanh(Spearman_Corrected_Taylor) + zcrit * Bonett_Wright_Standard_Error_Taylor)

    results = {}
    results["Spearman"] = Spearman
    results["Spearman p-value"] = spearman_rho_p_value
    results["Standard Error (Bonett & Wright)"] = Bonett_Wright_Standard_Error
    results["Standard Error (Fieller)"] = Fieller_Standard_Error
    results["Standard Error (Caruso and_Cliff)"] = Caruso_and_Cliff
    results["Confidence Intervals (Bonett Wright)"] = f"({round(Lower_ci_BW, 4)}, {round(Upper_ci_BW, 4)})"
    results["Confidence Intervals (Fieller)"] = f"({round(Lower_ci_Fieller, 4)}, {round(Upper_ci_Fieller, 4)})"
    results["Confidence Intervals (Caruso and Cliff)"] = f"({round(Lower_ci_CC, 4)}, {round(Upper_ci_CC, 4)})"
    results["Confidence Intervals (Woods)"] = f"({round(Lower_ci_Woods, 4)}, {round(Upper_ci_Woods, 4)})" # Woods 2007
    results["Confidence Intervals (Fisher)"] = f"({round(Lower_ci_Fisher, 4)}, {round(Upper_ci_Fisher, 4)})" 
    results["Ties Corrected Spearman (Oyeka and Nwankwo Chike, 2014) "] = Corrected_Spearman
    results["Ties Corrected Spearman p-value (Oyeka and Nwankwo Chike, 2014) "] = p_value_Oyeka
    results["Ties Corrected Spearman Standard Error (Oyeka and Nwankwo Chike, 2014) "] = Bonett_Wright_Standard_Error_Oyeka
    results["Ties Corrected Spearman CI's(Oyeka and Nwankwo Chike, 2014) "] = f"({round(Lower_ci_Oyeka, 4)}, {round(Upper_ci_Oyeka, 4)})"
    results["Ties Corrected Spearman (Taylor, 1964)"] = Spearman_Corrected_Taylor
    results["Ties Corrected Spearman p-value  (Taylor, 1964)"] = p_value_taylor
    results["Ties Corrected Spearman Standard Error (Taylor, 1964)"] = Bonett_Wright_Standard_Error_Taylor
    results["Ties Corrected Spearman CI's (Taylor, 1964)"] = f"({round(Lower_ci_Taylor, 4)}, {round(Upper_ci_Taylor, 4)})"
    formatted_p_value = "{:.3f}".format(spearman_rho_p_value).lstrip('0') if spearman_rho_p_value >= 0.001 else "\033[3mp\033[0m < .001"
    formatted_p_value_Oyeka = "{:.3f}".format(p_value_Oyeka).lstrip('0') if p_value_Oyeka >= 0.001 else "\033[3mp\033[0m < .001"
    formatted_p_value_Taylor = "{:.3f}".format(p_value_taylor).lstrip('0') if p_value_taylor >= 0.001 else "\033[3mp\033[0m < .001"

    results["Statistical Line Spearman "] = " \033[3mr\033[0m = {}, {}{}, {}% CI [{}, {}]".format(Spearman, '\033[3mp = \033[0m' if spearman_rho_p_value >= 0.001 else '', formatted_p_value, confidence_level, round(Lower_ci_BW,3), round(Upper_ci_BW,3)) 
    results["Statistical Line Corrected Spearamn (Oyeka et al.)"] = " {}{}, {}% CI [{}, {}]".format(Corrected_Spearman, '\033[3mp = \033[0m' if p_value_Oyeka >= 0.001 else '', formatted_p_value_Oyeka, confidence_level, round(Lower_ci_Oyeka,3), round(Upper_ci_Oyeka,3))
    results["Statistical Line (Taylor)"] = " \033[3mr\033[0m = {}, {}{}, {}% CI [{}, {}]".format(Spearman_Corrected_Taylor, '\033[3mp = \033[0m' if p_value_taylor >= 0.001 else '', formatted_p_value_Taylor, confidence_level, round(Lower_ci_Taylor,3), round(Upper_ci_Taylor,3))

    result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
    return result_str


# 3. Gaussian Rank Correlation
def Gausian_Rank_Correlation(x,y, confidence_level = 0.95): 
    Normalized_X = norm.ppf((rankdata(x) / (len(x) + 1)))
    Normalized_Y = norm.ppf((rankdata(y) / (len(y) + 1)))
    Gaussian_Rank_Correlation, GRC_pvalue = spearmanr(Normalized_X, Normalized_Y)
    sample_size = len(x)

    # Confidence Intervals
    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
    Bonett_Wright_Standard_Error = (1 + Gaussian_Rank_Correlation / 2) / (sample_size - 3) # 2000 - I use this as the default
    Lower_ci_BW = math.tanh(math.atanh(Gaussian_Rank_Correlation) - zcrit * Bonett_Wright_Standard_Error)
    Upper_ci_BW = math.tanh(math.atanh(Gaussian_Rank_Correlation) + zcrit * Bonett_Wright_Standard_Error)

    results = {}
    results["Gaussian Rank Correlation"] = Gaussian_Rank_Correlation
    results["Gaussian Rank Correlation p-value"] = GRC_pvalue
    results["Gaussian Rank Correlation CI's"] = [Lower_ci_BW, Upper_ci_BW]

    result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
    return result_str


# 4. Sheperd's Pi
def bsmahal(a, b, n_boot=200):
        n, m = b.shape
        MD = np.zeros((n, n_boot))
        nr = np.arange(n)
        xB = np.random.choice(nr, size=(n_boot, n), replace=True)
        for i in np.arange(n_boot):
            s1 = b[xB[i, :], 0]
            s2 = b[xB[i, :], 1]
            X = np.column_stack((s1, s2))
            mu = X.mean(0)
            _, R = np.linalg.qr(X - mu)
            sol = np.linalg.solve(R.T, (a - mu).T)
            MD[:, i] = np.sum(sol**2, 0) * (n - 1)
        return MD.mean(1)

def shepherd(x, y, n_boot=200, confidence_level = 0.95):
        sample_size = len(x)
        X = np.column_stack((x, y))
        m = bsmahal(X, X, n_boot)
        outliers = m >= 6
        r_shepherd, pval_shepherd = spearmanr(x[~outliers], y[~outliers])

        # Confidence Intervals
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
        Bonett_Wright_Standard_Error = (1 + r_shepherd / 2) / (sample_size - 3) # 2000 - I use this as the default
        Lower_ci_shepherd = math.tanh(math.atanh(r_shepherd) - zcrit * Bonett_Wright_Standard_Error)
        Upper_ci_shepherd = math.tanh(math.atanh(r_shepherd) + zcrit * Bonett_Wright_Standard_Error)

        results = {}
        results["Sheperd's Pi"] = r_shepherd
        results["Sheperd's Pi p-value"] = pval_shepherd
        results["Sheperd's Pi CI's"] = [Lower_ci_shepherd, Upper_ci_shepherd]

        result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
        return result_str

# 5. Ginni's Gamma
def ginis_gamma(x, y, confidence_level=0.95):
    ranked_x = rankdata(x)
    ranked_y = rankdata(y)
    sample_size = len(x)

    term1 = np.sum(np.abs((sample_size+1 - ranked_x) - ranked_y) - np.abs(ranked_x - ranked_y))
    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)

    if sample_size % 2 == 0:
        index1 = 1 / (sample_size**2 / 2)
        gamma = term1 * index1
        ASE = np.sqrt((2 * (sample_size**2 + 2)) / (3 * (sample_size - 1) * (sample_size**2)))
        Z = gamma / ASE
        p_value = 2 * (1 - norm.cdf(np.abs(Z)))  # Two-tailed test
        result_ASE = ASE
        lower_ci = gamma - ASE*zcrit
        upper_ci = gamma + ASE*zcrit
    else:
        index2 = 1 / ((sample_size**2 - 1) / 2)
        gamma = term1 * index2
        ASE = np.sqrt((2 * (sample_size**2 + 3)) / (3 * (sample_size - 1) * (sample_size**2 - 1)))
        Z = gamma / ASE
        p_value = 2 * (1 - norm.cdf(np.abs(Z)))  
        result_ASE = ASE
        lower_ci = gamma - ASE*zcrit
        upper_ci = gamma + ASE*zcrit

    # Get Results
    results = {}

    results["Ginni's Gamma"] = gamma
    results["Ginni's Gamma p-value"] = p_value
    results["Ginni's Gamma Standard Error"] = [result_ASE]
    results["Ginni's Gamma Standard Error"] = [lower_ci, upper_ci]

    result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
    return result_str


class OrdinalbyInterval():

    @staticmethod
    def Ordinal_By_Interval_from_Contingency_Table(params: dict) -> dict:
        
        # Set Params:
        Contingency_Table = params["Contingency Table"]
        confidence_level_percentages = params["Confidence Level"]
        n_boot = params["Number Of Bootstraps Samples"]

        # Preperations: 

        # Convrt the variables into vectors
        confidence_level = confidence_level_percentages/ 100
        Variable1, Variable2 = [j + 1 for i in range(Contingency_Table.shape[0]) for j in range(Contingency_Table.shape[1]) for _ in range(Contingency_Table[i, j])], [i + 1 for i in range(Contingency_Table.shape[0]) for j in range(Contingency_Table.shape[1]) for _ in range(Contingency_Table[i, j])]

        # Calculate Effect Sizes using All Functions        
        skipped_Correlation_measures = skipped_Correlation(Variable1, Variable2, confidence_level)
        GRC_output = Gausian_Rank_Correlation(Variable1,Variable2,confidence_level)        
        Ginnis_Gamma_Output = ginis_gamma(Variable1, Variable2,confidence_level)
        shepherd_output = shepherd(np.array(Variable1), np.array(Variable2), n_boot, confidence_level = confidence_level)
        Spearman_Correlation_Output = Spearman_Correlation(np.array(Variable1), np.array(Variable2), confidence_level)

        results = {}
        results["Contingency Table"] =  Contingency_Table
        results["Skipped Corrleation "] =  skipped_Correlation_measures
        results["Gausian Rank Correlation "] =  GRC_output
        results["Ginni's Gamma"] =  Ginnis_Gamma_Output
        results["Shepherd's Pi"] =  shepherd_output
        results["Spearman Correlation"] =  Spearman_Correlation_Output


        return results
    
    @staticmethod
    def Ordinal_By_Interval_from_Data(params: dict) -> dict:
        
        # Set Params:
        Variable1 = params["Variable 1"]
        Variable2 = params["Variable 2"]
        confidence_level_percentages = params["Confidence Level"]
        n_boot = params["Number Of Bootstraps Samples"]

        # Convrt the variables into vectors
        confidence_level = confidence_level_percentages/ 100

        # Calculate Effect Sizes using All Functions        
        skipped_Correlation_measures = skipped_Correlation(Variable1, Variable2, confidence_level)
        GRC_output = Gausian_Rank_Correlation(Variable1,Variable2,confidence_level)        
        Ginnis_Gamma_Output = ginis_gamma(Variable1, Variable2,confidence_level)
        shepherd_output = shepherd(np.array(Variable1), np.array(Variable2), n_boot, confidence_level = confidence_level)
        Spearman_Correlation_Output = Spearman_Correlation(np.array(Variable1), np.array(Variable2), confidence_level)

        results = {}
        results["Skipped Corrleation "] =  skipped_Correlation_measures
        results["Gausian Rank Correlation "] =  GRC_output
        results["Ginni's Gamma"] =  Ginnis_Gamma_Output
        results["Shepherd's Pi"] =  shepherd_output
        results["Spearman Correlation"] =  Spearman_Correlation_Output


        return results









