

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
    #formatted_p_value_skipped_iqr = "{:.3f}".format(spearman_rho_p_value).lstrip('0') if spearman_rho_p_value >= 0.001 else "\033[3mp\033[0m < .001"
    #formatted_p_value_skipped_ = "{:.3f}".format(p_value_Oyeka).lstrip('0') if p_value_Oyeka >= 0.001 else "\033[3mp\033[0m < .001"
    #results["Statistical Line Spearman"] = "\033[3mr\033[0m({}) = {}{}, {}{}, {}{}% CI [{}{}, {}{}]".format((sample_size - 2), ('-' if np.round(Spearman,3) < 0 else ''), str(np.abs(np.round(Spearman,3))).lstrip('0').rstrip(''), '\033[3mp = \033[0m' if spearman_rho_p_value >= 0.001 else '', formatted_p_value, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(Lower_ci_BW,3) < 0 else ''), str(np.abs(np.round(Lower_ci_BW,3))).lstrip('0').rstrip(''), ('-' if np.round(Upper_ci_BW,3) < 0 else ''), str(np.abs(np.round(Upper_ci_BW,3))).lstrip('0').rstrip(''))
    #results["Statistical Line Corrected Spearamn (Oyeka et al.)"] = "\033[3mr\033[0m({}) = {}{}, {}{}, {}{}% CI [{}{}, {}{}]".format((sample_size - 2), ('-' if np.round(Corrected_Spearman,3) < 0 else ''), str(np.abs(np.round(Corrected_Spearman,3))).lstrip('0').rstrip(''), '\033[3mp = \033[0m' if p_value_Oyeka >= 0.001 else '', formatted_p_value_Oyeka, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(Lower_ci_Oyeka,3) < 0 else ''), str(np.abs(np.round(Lower_ci_Oyeka,3))).lstrip('0').rstrip(''), ('-' if np.round(Upper_ci_Oyeka,3) < 0 else ''), str(np.abs(np.round(Upper_ci_Oyeka,3))).lstrip('0').rstrip(''))
    #results["Statistical Line Corrected Spearamn (Taylor.)"] = "\033[3mr\033[0m({}) = {}{}, {}{}, {}{}% CI [{}{}, {}{}]".format((sample_size - 2), ('-' if np.round(Spearman_Corrected_Taylor,3) < 0 else ''), str(np.abs(np.round(Spearman_Corrected_Taylor,3))).lstrip('0').rstrip(''), '\033[3mp = \033[0m' if p_value_taylor >= 0.001 else '', formatted_p_value_Taylor, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(Lower_ci_Taylor,3) < 0 else ''), str(np.abs(np.round(Lower_ci_Taylor,3))).lstrip('0').rstrip(''), ('-' if np.round(Upper_ci_Taylor,3) < 0 else ''), str(np.abs(np.round(Upper_ci_Taylor,3))).lstrip('0').rstrip(''))

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
    results["Statistical Line Spearman"] = "\033[3mr\033[0m({}) = {}{}, {}{}, {}{}% CI [{}{}, {}{}]".format((sample_size - 2), ('-' if np.round(Spearman,3) < 0 else ''), str(np.abs(np.round(Spearman,3))).lstrip('0').rstrip(''), '\033[3mp = \033[0m' if spearman_rho_p_value >= 0.001 else '', formatted_p_value, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(Lower_ci_BW,3) < 0 else ''), str(np.abs(np.round(Lower_ci_BW,3))).lstrip('0').rstrip(''), ('-' if np.round(Upper_ci_BW,3) < 0 else ''), str(np.abs(np.round(Upper_ci_BW,3))).lstrip('0').rstrip(''))
    results["Statistical Line Corrected Spearamn (Oyeka et al.)"] = "\033[3mr\033[0m({}) = {}{}, {}{}, {}{}% CI [{}{}, {}{}]".format((sample_size - 2), ('-' if np.round(Corrected_Spearman,3) < 0 else ''), str(np.abs(np.round(Corrected_Spearman,3))).lstrip('0').rstrip(''), '\033[3mp = \033[0m' if p_value_Oyeka >= 0.001 else '', formatted_p_value_Oyeka, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(Lower_ci_Oyeka,3) < 0 else ''), str(np.abs(np.round(Lower_ci_Oyeka,3))).lstrip('0').rstrip(''), ('-' if np.round(Upper_ci_Oyeka,3) < 0 else ''), str(np.abs(np.round(Upper_ci_Oyeka,3))).lstrip('0').rstrip(''))
    results["Statistical Line Corrected Spearamn (Taylor.)"] = "\033[3mr\033[0m({}) = {}{}, {}{}, {}{}% CI [{}{}, {}{}]".format((sample_size - 2), ('-' if np.round(Spearman_Corrected_Taylor,3) < 0 else ''), str(np.abs(np.round(Spearman_Corrected_Taylor,3))).lstrip('0').rstrip(''), '\033[3mp = \033[0m' if p_value_taylor >= 0.001 else '', formatted_p_value_Taylor, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(Lower_ci_Taylor,3) < 0 else ''), str(np.abs(np.round(Lower_ci_Taylor,3))).lstrip('0').rstrip(''), ('-' if np.round(Upper_ci_Taylor,3) < 0 else ''), str(np.abs(np.round(Upper_ci_Taylor,3))).lstrip('0').rstrip(''))

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


# 4. Sheperd's Pi - Schwarzkopf, D.S., De Haas, B., Rees, G., 2012. Better ways to improve standards in brain-behavior correlation analysis
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
        formatted_p_value = "{:.3f}".format(pval_shepherd).lstrip('0') if pval_shepherd >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Sheperd's Pi"] = "Shepherd's \033[3mpi\033[0m({}) = {}{}, {}{}, {}{}% CI [{}{}, {}{}]".format((sample_size - 2), ('-' if np.round(r_shepherd,3) < 0 else ''), str(np.abs(np.round(r_shepherd,3))).lstrip('0').rstrip(''), '\033[3mp = \033[0m' if pval_shepherd >= 0.001 else '', formatted_p_value, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(Lower_ci_shepherd,3) < 0 else ''), str(np.abs(np.round(Lower_ci_shepherd,3))).lstrip('0').rstrip(''), ('-' if np.round(Upper_ci_shepherd,3) < 0 else ''), str(np.abs(np.round(Upper_ci_shepherd,3))).lstrip('0').rstrip(''))

        result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
        return result_str

# 5. Ginnis Gamma
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
    results["Confidence Interval's Ginni's Gamma"] = [lower_ci, upper_ci]
    formatted_p_value = "{:.3f}".format(p_value).lstrip('0') if p_value >= 0.001 else "\033[3mp\033[0m < .001"
    results["Statistical Line Ginni's Gamma"] = "Ginni's \033[3m\u03BB\033[0m({}) = {}{}, {}{}, {}{}% CI [{}{}, {}{}]".format((sample_size - 2), ('-' if np.round(gamma,3) < 0 else ''), str(np.abs(np.round(gamma,3))).lstrip('0').rstrip(''), '\033[3mp = \033[0m' if p_value >= 0.001 else '', p_value, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(lower_ci,3) < 0 else ''), str(np.abs(np.round(lower_ci,3))).lstrip('0').rstrip(''), ('-' if np.round(upper_ci,3) < 0 else ''), str(np.abs(np.round(upper_ci,3))).lstrip('0').rstrip(''))

    result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
    return result_str

# 6. Gamma Family: Kendall tau, Stuart Tau Goodman & Krushkal Gamma, Wilson e, Somer's Delta
def Gamma_Family_Measures(coloumn_1,coloumn_2, confidence_level_percentages):
    
    # Preperations of the Data
    confidence_level = confidence_level_percentages / 100
    Data_Frame = pd.DataFrame({'coloumn_1': coloumn_1, 'coloumn_2': coloumn_2})
    Contingency_Table = pd.crosstab(Data_Frame['coloumn_1'], Data_Frame['coloumn_2'])
    Sample_Size = np.sum(np.sum(Contingency_Table, axis=1))
    number_of_rows, number_of_columns = Contingency_Table.shape

    q = min(number_of_rows, number_of_columns)
    Final_Data = Contingency_Table.reset_index().melt(id_vars='coloumn_1', var_name='coloumn_2', value_name='Nij')
    Final_Data['Ni'] = Final_Data['coloumn_1'].map(Data_Frame['coloumn_1'].value_counts())
    Final_Data['Nj'] = Final_Data['coloumn_2'].map(Data_Frame['coloumn_2'].value_counts())
    Final_Data['P'] = Final_Data['Concordant Pairs'] = 0
    Final_Data['Q'] = Final_Data['Disconcordant Pairs'] = 0

    for i, row in Final_Data.iterrows():
        x_val = row['coloumn_1']
        y_val = row['coloumn_2']
        Concordant = 0
        Disconcordant = 0
        
        # Iterate through rows with different x values
        for index, other_row in Final_Data[Final_Data['coloumn_1'] != x_val].iterrows():
            count1 = other_row['Nij']  # Frequency of the other values group
            if (x_val > other_row['coloumn_1'] and y_val > other_row['coloumn_2']) or (x_val < other_row['coloumn_1'] and y_val < other_row['coloumn_2']):
                Concordant += count1
        for index, other_row in Final_Data[Final_Data['coloumn_1'] != x_val].iterrows():
            count2 = other_row['Nij']  # Frequency of the other values group
            if (x_val > other_row['coloumn_1'] and y_val < other_row['coloumn_2']) or (x_val < other_row['coloumn_1'] and y_val > other_row['coloumn_2']):
                Disconcordant += count2
        
        Final_Data.at[i, 'Concordant Pairs'] = Concordant
        Final_Data.at[i, 'Disconcordant Pairs'] = Disconcordant
        Final_Data.at[i, '(C-D)^2*Nij'] = (Concordant - Disconcordant)**2 * row['Nij']
        Final_Data.at[i, 'Concordant Pairs * Frequency'] = Concordant * row['Nij']
        Final_Data.at[i, 'Disconcordant Pairs * Frequency'] = Disconcordant * row['Nij']
        Final_Data.at[i, 'P'] = Final_Data.at[i, 'Concordant Pairs * Frequency']
        Final_Data.at[i, 'Q'] = Final_Data.at[i, 'Disconcordant Pairs * Frequency']
        Final_Data.at[i, 'Cij - Dij'] = Concordant - Disconcordant
        Final_Data.at[i, 'vij'] = (Sample_Size**2 - np.sum(np.sum(Contingency_Table, axis = 1)**2)) * row['Nj'] + (Sample_Size**2 -np.sum(np.sum(Contingency_Table, axis = 0)**2)) * row['Ni']

    # Calculation of Measures of Association and thier Standrd Errors
        
    P = Final_Data['P'].sum()
    Q = Final_Data['Q'].sum()
    D_var1 = Sample_Size**2 - np.sum(np.sum(Contingency_Table, axis = 1)**2)
    D_var2 = Sample_Size**2 - np.sum(np.sum(Contingency_Table, axis = 0)**2)

    # Measures of Association

    # 1. Tau Family: A. Kendall Tau-a (Kendall, 1938) B. Kendall Tau-b (Daniels, 1944; Kendall, 1945) C. Stuart Tau-C (Stuart, 1953)
    Kendall_Tau_a = (P-Q) / ((Sample_Size * (Sample_Size - 1)))
    Kendall_Tau_b = (P - Q) / np.sqrt(D_var2*D_var1)
    Stuart_Tau_c = q*(P - Q) / ((Sample_Size**2)*(q-1))
    Weighted_Tau, Weighted_Tau_p_value = weightedtau(coloumn_1, coloumn_2) # Grace & Shieh (1998)

    Term1 = (P-Q) * (Contingency_Table !=0)
    Term2 = np.sum(Term1[Term1 != 0]) / Sample_Size
    Term3 = 1 / (D_var1 * D_var2)
    Term4 = 2 * np.sqrt(D_var1 * D_var2)
    Term5 = Sample_Size**3*Kendall_Tau_b**2*(D_var1 + D_var2)**2

    Ci = Final_Data['Cij - Dij'] 
    C_ = np.sum(Ci) / Sample_Size

    ASE1_Tau_a = np.sqrt(2/(Sample_Size * (Sample_Size - 1)) * ((2 * (Sample_Size - 2))/(Sample_Size * (Sample_Size - 1)**2) * sum((Ci - C_)**2) + 1 - Kendall_Tau_a**2))
    ASE1_Tau_b = Term3 * np.sqrt(np.sum( Final_Data['Nij']*((Term4)*Final_Data['Cij - Dij'] + Final_Data['vij']*Kendall_Tau_b)**2) - Term5)
    ASE1_Tau_c = ((2*q) / ((q-1)*Sample_Size**2)) * np.sqrt((np.sum(Final_Data['Nij'] * (Final_Data['Disconcordant Pairs'] - Final_Data['Concordant Pairs'])**2)) - (1 / Sample_Size * (P-Q)**2)) # Note that for Stuart C ASE0 and ASE1 are the same

    ASE0_Tau_a = np.sqrt(2/(Sample_Size * (Sample_Size - 1)) * ((2 * (Sample_Size - 2))/(Sample_Size * (Sample_Size - 1)**2) * sum((Ci - C_)**2) + 1 - Kendall_Tau_a**2)) # I need to find the real one...it is now the ASE1
    ASE0_Tau_b =  np.sqrt(   ((np.sum(Final_Data['Nij'] * (Final_Data['Disconcordant Pairs'] - Final_Data['Concordant Pairs'])**2)) - (1 / Sample_Size * (P-Q)**2)) / (D_var1*D_var2)   )*2
    ASE0_Tau_c = ((2*q) / ((q-1)*Sample_Size**2)) * np.sqrt((np.sum(Final_Data['Nij'] * (Final_Data['Disconcordant Pairs'] - Final_Data['Concordant Pairs'])**2)) - (1 / Sample_Size * (P-Q)**2)) # Note that for Stuart C ASE0 and ASE1 are the same

    # Sommers Delta
    Sommers_Delta_Symmetric = (P - Q)/(0.5*(D_var2+D_var1))
    Sommers_Delta_Var1 = (P - Q) / D_var1
    Sommers_Delta_Var2 = (P - Q) / D_var2

    ASE0_Delta_Symmetric = (4/ (D_var1+D_var2)) *np.sqrt((np.sum(Final_Data['Nij'] * (Final_Data['Disconcordant Pairs'] - Final_Data['Concordant Pairs'])**2)) - (1 / Sample_Size * (P-Q)**2))
    ASE0_Delta_Var1 = 2/ D_var1 *np.sqrt((np.sum(Final_Data['Nij'] * (Final_Data['Disconcordant Pairs'] - Final_Data['Concordant Pairs'])**2)) - (1 / Sample_Size * (P-Q)**2))
    ASE0_Delta_Var2 = 2/ D_var2 *np.sqrt((np.sum(Final_Data['Nij'] * (Final_Data['Disconcordant Pairs'] - Final_Data['Concordant Pairs'])**2)) - (1 / Sample_Size * (P-Q)**2))

    ASE1_Delta_Symmetric = ((ASE1_Tau_b *2 / (D_var1 + D_var2) ) * np.sqrt(D_var1*D_var2))
    ASE1_Delta_Var1 = 2/ D_var1 *np.sqrt((np.sum(Final_Data['Nij'] * (Final_Data['Disconcordant Pairs'] - Final_Data['Concordant Pairs'])**2)) - (1 / Sample_Size * (P-Q)**2))
    ASE1_Delta_Var2 = (2 / D_var2**2) * np.sqrt(np.sum(Final_Data['Nij'] * (D_var2 * (Final_Data['Concordant Pairs'] - Final_Data['Disconcordant Pairs']) - (P - Q) *(Sample_Size - Final_Data['Nj']))**2))

    # Goodman Krushkal Gamma
    Gamma_Correlation = (P - Q) / (P + Q)
    ASE1_Gamma = (4 / (P + Q)**2) * (np.sqrt(np.sum(Final_Data['Nij'] * np.float64(((Final_Data['Disconcordant Pairs'] * P) - (Final_Data['Concordant Pairs'] * Q)) **2))))
    ASE0_Gamma = (2/ (P+Q)) * np.sqrt((np.sum(Final_Data['Nij'] * (Final_Data['Disconcordant Pairs'] - Final_Data['Concordant Pairs'])**2)) - (1 / Sample_Size * (P-Q)**2))
    #Standard_Error_Gamma_H0_1 = 1 / np.sqrt((P + Q) / (Sample_Size* (1 - Gamma_Correlation**2))) # This is the formula that appears in Sheskin, 2003


    # Wilson's e
    Wilsons_e = (2 * (np.sum(Final_Data['Concordant Pairs'] * Final_Data['Nij'] / 2) - np.sum(Final_Data['Disconcordant Pairs'] * Final_Data['Nij'] / 2))) / (Sample_Size**2  - np.sum(np.sum(Contingency_Table**2)))
    ASE_Wilson_Term1 = 4 * np.sum(Final_Data['Nij']*(Final_Data['Concordant Pairs'] - Final_Data['Disconcordant Pairs'])**2) - 4/Sample_Size *((np.sum(Final_Data['Concordant Pairs'] * Final_Data['Nij']) / 2)-(np.sum(Final_Data['Disconcordant Pairs'] * Final_Data['Nij']) / 2))**2
    ASE_Wilson_Term2  =  (Sample_Size**2  - np.sum(Final_Data['Nij']**2))**2
    ASE_Wilson = np.sqrt(ASE_Wilson_Term1 / ASE_Wilson_Term2)

    # Confidence Intervals and Significance
    
    # Statistics
    Z_Statistic_tau_a = Kendall_Tau_a / ASE0_Tau_a
    Z_Statistic_tau_b = Kendall_Tau_b / ASE0_Tau_b
    Z_Statistic_tau_c = Stuart_Tau_c / ASE0_Tau_c
    Z_Statistic_sommers_delta_var1 = Sommers_Delta_Var1 / ASE0_Delta_Var1
    Z_Statistic_sommers_delta_var2 = Sommers_Delta_Var2 / ASE0_Delta_Var2
    Z_Statistic_sommers_delta_symmetric = Sommers_Delta_Symmetric / ASE0_Delta_Symmetric
    Z_Statistic_Gamma = Gamma_Correlation / ASE0_Gamma
    Z_Statistic_Wilson = Wilsons_e / ASE_Wilson

    p_value_tau_a = norm.sf(Z_Statistic_tau_a)
    p_value_tau_b = norm.sf(Z_Statistic_tau_b)
    p_value_tau_c = norm.sf(Z_Statistic_tau_c)
    p_value_somer_var1 = norm.sf(Z_Statistic_sommers_delta_var1)
    p_value_somer_var2 = norm.sf(Z_Statistic_sommers_delta_var2)
    p_value_somer_symmetric = norm.sf(Z_Statistic_sommers_delta_symmetric)
    p_value_gamma = norm.sf(Z_Statistic_Gamma)
    p_value_wilson = norm.sf(Z_Statistic_Wilson)

    # Confidence Levels
    critical_z_value = norm.ppf(1 - ((1-confidence_level)/2))

    CI_lower_tau_a = Kendall_Tau_a - critical_z_value * ASE1_Tau_a
    CI_upper_tau_a = Kendall_Tau_a + critical_z_value * ASE1_Tau_a
    CI_lower_tau_b = Kendall_Tau_b - critical_z_value * ASE1_Tau_b
    CI_upper_tau_b = Kendall_Tau_b + critical_z_value * ASE1_Tau_b
    CI_lower_tau_c = Stuart_Tau_c - critical_z_value * ASE1_Tau_c
    CI_upper_tau_c = Stuart_Tau_c + critical_z_value * ASE1_Tau_c
    CI_lower_somer_var1 = Sommers_Delta_Var1 - critical_z_value * ASE1_Delta_Var1
    CI_upper_somer_var1 = Sommers_Delta_Var1 + critical_z_value * ASE1_Delta_Var1
    CI_lower_somer_var2 = Sommers_Delta_Var2 - critical_z_value * ASE1_Delta_Var2
    CI_upper_somer_var2 = Sommers_Delta_Var2 + critical_z_value * ASE1_Delta_Var2
    CI_lower_somer_symmetric = Sommers_Delta_Symmetric - critical_z_value * ASE1_Delta_Symmetric
    CI_upper_somer_symmetric = Sommers_Delta_Symmetric + critical_z_value * ASE1_Delta_Symmetric
    CI_lower_gamma = Gamma_Correlation - critical_z_value * ASE1_Gamma
    CI_upper_gamma = Gamma_Correlation + critical_z_value * ASE1_Gamma
    CI_lower_wilson = Wilsons_e - critical_z_value * ASE_Wilson
    CI_upper_wilson = Wilsons_e + critical_z_value * ASE_Wilson

    
    results = {}

    results["Sommer's Delta Symmetric"] = np.array([Sommers_Delta_Symmetric, ASE1_Delta_Symmetric, ASE0_Delta_Symmetric])
    results["Sommer's Delta Variable 1:"]= np.array([Sommers_Delta_Var1, ASE1_Delta_Var1, ASE0_Delta_Var1])
    results["Sommers Delta Variabl 2"] = np.array([Sommers_Delta_Var2, ASE1_Delta_Var2, ASE0_Delta_Var2])
    results["Kendall's Tau A"] = np.array([Kendall_Tau_a, ASE1_Tau_a, ASE1_Tau_a])
    results["Kendall's Tau B"] = np.array([Kendall_Tau_b, ASE1_Tau_b, ASE0_Tau_b])
    results["Stuart Tau C"]= np.array([Stuart_Tau_c, ASE1_Tau_c, ASE0_Tau_c])
    results["Gamma Correlation"]= np.array([Gamma_Correlation, ASE1_Gamma, ASE0_Gamma])
    results["Wilson's e "]= ([Wilsons_e, ASE_Wilson_Term1, ASE_Wilson])
    results["Weighted Tau"] = np.array(Weighted_Tau)
    results["Weighted Tau p-value"] = np.array(Weighted_Tau_p_value)

    # Z statistics
    results["Z-Statistic_tau_a"] = Z_Statistic_tau_a
    results["Z-Statistic_tau_b"] = Z_Statistic_tau_b
    results["Z-Statistic_tau_c"] = Z_Statistic_tau_c
    results["Z-Statistic_sommers_delta_var1"] = Z_Statistic_sommers_delta_var1
    results["Z-Statistic_sommers_delta_var2"] = Z_Statistic_sommers_delta_var2
    results["Z-Statistic_sommers_delta_symmetric"] = Z_Statistic_sommers_delta_symmetric
    results["Z-Statistic_Gamma"] = Z_Statistic_Gamma
    results["Z-Statistic_Wilson"] = Z_Statistic_Wilson

    # p-values
    results["p-value_tau_a"] = p_value_tau_a
    results["p-value_tau_b"] = p_value_tau_b
    results["p-value_tau_c"] = p_value_tau_c
    results["p-value_somer_var1"] = p_value_somer_var1
    results["p-value_somer_var2"] = p_value_somer_var2
    results["p-value_somer_symmetric"] = p_value_somer_symmetric
    results["p-value_gamma"] = p_value_gamma
    results["p-value_wilson"] = p_value_wilson

    # Confidence Intervals
    results["CI_tau_a"] = [CI_lower_tau_a, CI_upper_tau_a]
    results["CI_tau_b"] = [CI_lower_tau_b, CI_upper_tau_b]
    results["CI_tau_c"] = [CI_lower_tau_c, CI_upper_tau_c]
    results["CI_somer_var1"] = [CI_lower_somer_var1, CI_upper_somer_var1]
    results["CI_somer_var2"] = [CI_lower_somer_var2, CI_upper_somer_var2]
    results["CI_somer_symmetric"] = [CI_lower_somer_symmetric, CI_upper_somer_symmetric]
    results["CI_gamma"] = [CI_lower_gamma, CI_upper_gamma]
    results["CI_wilson"] = [CI_lower_wilson, CI_upper_wilson]

    result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])

    return result_str

class OrdinalbyOrdinal():

    @staticmethod
    def Ordinal_By_Ordinal_from_Contingency_Table(params: dict) -> dict:
        
        # Set Params:
        Contingency_Table = params["Contingency Table"]
        confidence_level_percentages = params["Confidence Level"]
        n_boot = params["Number Of Bootstraps Samples"]
        confidence_level = confidence_level_percentages/ 100


        # Convert the contingency table into vectors
        Variable1, Variable2 = [j + 1 for i in range(Contingency_Table.shape[0]) for j in range(Contingency_Table.shape[1]) for _ in range(Contingency_Table[i, j])], [i + 1 for i in range(Contingency_Table.shape[0]) for j in range(Contingency_Table.shape[1]) for _ in range(Contingency_Table[i, j])]

        # Calculate Effect Sizes using All Functions        
        skipped_Correlation_measures = skipped_Correlation(Variable1, Variable2, confidence_level)
        GRC_output = Gausian_Rank_Correlation(Variable1,Variable2,confidence_level)        
        Ginnis_Gamma_Output = ginis_gamma(Variable1, Variable2,confidence_level)
        shepherd_output = shepherd(np.array(Variable1), np.array(Variable2), n_boot, confidence_level = confidence_level)
        Gamma_Family_Output = Gamma_Family_Measures(Variable1, Variable2, confidence_level)
        Spearman_Correlation_Output = Spearman_Correlation(np.array(Variable1), np.array(Variable2), confidence_level)

        results = {}
        results["Contingency Table"] =  Contingency_Table
        results["Skipped Corrleation "] =  skipped_Correlation_measures
        results["Gausian Rank Correlation "] =  GRC_output
        results["Ginni's Gamma"] =  Ginnis_Gamma_Output
        results["Shepherd's Pi"] =  shepherd_output
        results["The Gamma Family Measures"] =  Gamma_Family_Output
        results["Spearman Correlation"] =  Spearman_Correlation_Output


        return results
    
    @staticmethod
    def Ordinal_By_Ordinal_from_Data(params: dict) -> dict:
        
        # Set Params:
        Variable1 = params["Variable 1"]
        Variable2 = params["Variable 2"]
        n_boot = params["Number Of Bootstraps Samples"]
        confidence_level_percentages = params["Confidence Level"]
        confidence_level = confidence_level_percentages/ 100


        # Calculate Effect Sizes using All Functions        
        skipped_Correlation_measures = skipped_Correlation(Variable1, Variable2, confidence_level)
        GRC_output = Gausian_Rank_Correlation(Variable1,Variable2,confidence_level)        
        Ginnis_Gamma_Output = ginis_gamma(Variable1, Variable2,confidence_level)
        shepherd_output = shepherd((np.array(Variable1)), (np.array(Variable2)), n_boot, confidence_level = confidence_level) # For some Data with small N, many bootstraps samples can cause an error..Need to chack that
        Gamma_Family_Output = Gamma_Family_Measures(np.array(Variable1), np.array(Variable2), confidence_level)
        Spearman_Correlation_Output = Spearman_Correlation(np.array(Variable1), np.array(Variable2), confidence_level)

        results = {}
        results["Skipped Corrleation "] =  skipped_Correlation_measures
        results["Gausian Rank Correlation "] =  GRC_output
        results["Ginni's Gamma"] =  Ginnis_Gamma_Output
        results["Shepherd's Pi"] =  shepherd_output
        results["The Gamma Family Measures"] =  Gamma_Family_Output
        results["Spearman Correlation"] =  Spearman_Correlation_Output


        return results









