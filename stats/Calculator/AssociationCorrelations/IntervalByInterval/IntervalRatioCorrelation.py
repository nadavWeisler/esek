
import numpy as np
import math
import scipy.optimize as opt
from scipy.stats import pearsonr, norm, t, rankdata, median_abs_deviation, ncf,f, bootstrap
import scipy.special as special
import pandas as pd

# Relevant Fnctions for Interval/Ratio Correlations

### 0. The non Central F function 
def Non_Central_CI_F(F_Statistic, df1, df2, confidence_level):
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
    Lower_NCF_CI_Final = Lower_CI(F_Statistic, df1, df2, Upper_Limit, Lower_CI_Difference_Value)[0]
    Upper_NCF_CI_Final = Upper_CI(F_Statistic, df1, df2, Lower_Limit, Lower_CI_Difference_Value)[0]

    return Lower_NCF_CI_Final, Upper_NCF_CI_Final


# 1. Main Function Pearson Correlation
def pearson_correlation(x,y,confidence_level = 0.95):
    Pearson_Correlation, Pearson_pvalue =  pearsonr(x,y)

    sample_size = len(x)
    Sample_1_Mean = np.mean(x)    
    Sample_2_Mean = np.mean(y)
    Sample_1_Standard_Deviation = np.std(x, ddof = 1)    
    Sample_2_Standard_Deviation = np.std(y, ddof = 1)
    Slope = (Sample_2_Standard_Deviation/Sample_1_Standard_Deviation) * Pearson_Correlation
    constant = -(Slope * Sample_1_Mean - Sample_2_Mean)
    Predicted_Values = Slope*np.array(x) + constant
    Sum_of_square_deviations_from_mean_y = np.sum((y - Sample_2_Mean)**2) ##Sstotal
    Sum_of_square_deviations_from_mean_x = np.sum((x - Sample_1_Mean)**2) ##Ssx
    Sum_of_residuals = np.sum((y-Predicted_Values)**2) #SSres
    sum_of_regression = Sum_of_square_deviations_from_mean_y - Sum_of_residuals #SSreg
    Standard_error_slope = np.sqrt((1/(sample_size-2)) * (Sum_of_residuals/Sum_of_square_deviations_from_mean_x)) 
    t_score_of_the_slope = Slope / Standard_error_slope
    Rsquare = (sum_of_regression/Sum_of_square_deviations_from_mean_y)
    Approximated_r = Pearson_Correlation + (Pearson_Correlation*(1-Rsquare)) / (2*(sample_size-3))

    # Approximated Standrd Errors - Gnambs, 2023
    Fisher_SE = (1-Rsquare) / np.sqrt(sample_size*(1+Rsquare))  #Fisher, 1896
    Fisher_Filon_SE = (1-Rsquare) / np.sqrt(sample_size)  #Fisher & Filon, 1898
    Soper_1_SE = ((1-Rsquare) / np.sqrt(sample_size-1))  # Soper, 1913, Large Sample
    Soper_2_SE = ((1-Rsquare)/np.sqrt(sample_size)) * ( 1 + (1+5.5*Rsquare) / (2*sample_size))  # Soper, 1913
    Soper_3_SE = ((1-Rsquare)/np.sqrt(sample_size-1)) * (1 + (11*Rsquare) / (4*(sample_size-1)))  # Soper, 1913
    Hoteling = ((1-Rsquare)/np.sqrt(sample_size-1)) * (1 + ((11*Rsquare) / (4*(sample_size-1))) + ((-192*Rsquare+479*Rsquare**2)/ (32*(sample_size-1)**2)))  # Hoteling, 1953
    Ghosh_SE = np.sqrt((1 - ((sample_size - 2) * (1 - Rsquare) / (sample_size - 1) * special.hyp2f1(1, 1, (sample_size + 1) / 2, Rsquare)) - ((2 / (sample_size - 1) * (math.gamma(sample_size / 2) / math.gamma((sample_size - 1) / 2))**2) * np.sqrt(Rsquare) * special.hyp2f1(0.5, 0.5, (sample_size + 1) / 2, Rsquare))**2))
    Hedges_SE = np.sqrt((Pearson_Correlation * special.hyp2f1(0.5, 0.5, (sample_size - 2) / 2, 1 - Rsquare))**2 - (1 - (sample_size - 3) * (1 - Rsquare) * special.hyp2f1(1, 1, sample_size / 2, 1 - Rsquare) / (sample_size - 2)))
    Bonett_SE = (1-Rsquare) / np.sqrt(sample_size - 3)
    Regression_SE = np.sqrt((1-Rsquare) / (sample_size - 2))
 
    # Different Confidence Intervals for Pearson Correlation:
    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
    tcrit = t.ppf(1 - (1 - confidence_level) / 2, sample_size-2)

    # 1. Fisher Based CI's (1921)
    Zr = 0.5 * np.log((1+Pearson_Correlation) / (1-Pearson_Correlation)) # Fisher's Transformation
    Standard_Error_ZR =  1 / (np.sqrt(sample_size-3)) #Fisher (1921) Approximation of the Variance
    Z_Lower = Zr - zcrit * Standard_Error_ZR
    Z_Upper = Zr + zcrit * Standard_Error_ZR
    Pearosn_Fisher_CI_lower = (np.exp(2*Z_Lower) - 1) / (np.exp(2*Z_Lower) + 1)
    Pearosn_Fisher_CI_upper = (np.exp(2*Z_Upper) - 1) / (np.exp(2*Z_Upper) + 1)

    # 2. Olivoto et al., 2018 - Half Widthe Confidecne Interval
    Half_Width = (0.45304**abs(Pearson_Correlation)) * 2.25152 * (sample_size**-0.50089)
    Lower_Olivoto = Pearson_Correlation - Half_Width
    Upperr_Olivoto = Pearson_Correlation + Half_Width

    # 3. Olkin & Fin, 1995
    Standard_Error_R_Sqaure = np.sqrt((4 * Rsquare * (1- Rsquare)**2*(sample_size-2)**2) / ((sample_size**2 -1)*(sample_size+3)))# Olkin & Finn, 1995
    Pearson_CI_Olkin_Fin_Lower = np.sqrt(Rsquare - zcrit * Standard_Error_R_Sqaure)
    Pearson_CI_Olkin_Fin_Upper = np.sqrt(Rsquare + zcrit * Standard_Error_R_Sqaure)

    # 4. Non-Central CI's based on F distribtion
    Lowerc_NCP_CI, Upper_NCP_CI = Non_Central_CI_F(t_score_of_the_slope**2, 1, sample_size-2, confidence_level)
    Pearson_CI_Eta_lower = np.sqrt(Lowerc_NCP_CI / (Lowerc_NCP_CI + (sample_size-2)))
    Pearson_CI_Eta_upper = np.sqrt(Upper_NCP_CI / (Upper_NCP_CI + (sample_size-2)))

    # 5. Bootstrapping CI's
    def pearson_r(x, y):
        return pearsonr(x, y)[0]
    Bootstrap_Sample = bootstrap((x, y), pearson_r, n_resamples = 1000, vectorized=False, paired=True, random_state=np.random.default_rng(), confidence_level = confidence_level)
    Pearosn_Bootstrapping_CI = Bootstrap_Sample.confidence_interval

    # 6. Bonett 2008 Procedure
    Lower_ci_Bonett = math.tanh(math.atanh(Pearson_Correlation) - zcrit * Bonett_SE)
    Upper_ci_Bonett= math.tanh(math.atanh(Pearson_Correlation) + zcrit * Bonett_SE)

    # 7. Null Counter-Null Interval
    Lower_ci_Bonett = math.tanh(math.atanh(Pearson_Correlation) - zcrit * Bonett_SE)
    Upper_ci_Bonett= math.tanh(math.atanh(Pearson_Correlation) + zcrit * Bonett_SE)


    # More Alternative Effect Sizes for Pearson Correlation 
    Common_Language_Effect_Size_Dunlap = np.arcsin(Pearson_Correlation) / math.pi + 0.
    Counter_Null_EffectSize = np.sqrt(4*Rsquare/(1+3*Rsquare))    
    Absolute_R_Square = 1 - (sum_of_regression / (np.sum((y - np.median(y))**2))) # Bertsimas et al., 2008

    # Confidence Intervals for other Effect Sizes
    Absolute_R_Square_SE = (1-Absolute_R_Square**2) / np.sqrt(sample_size - 3)

    T_Statistic_CLES = Common_Language_Effect_Size_Dunlap * np.sqrt((sample_size-2) / (1 - Common_Language_Effect_Size_Dunlap**2))


    Lower_ci_CLES = np.arcsin(Lower_ci_Bonett) / math.pi + 0.
    Upper_ci_CLES = np.arcsin(Upper_ci_Bonett) / math.pi + 0.
    #Lower_ci_Counter_Null = np.sqrt(4*Lower_ci_Bonett/(1+3*Lower_ci_Bonett))
    #Upper_ci_Counter_Null = np.sqrt(4*Upper_ci_Bonett/(1+3*Upper_ci_Bonett))
    Lower_ci_Absolute_R = math.tanh(math.atanh(Absolute_R_Square_SE) - zcrit * Absolute_R_Square_SE)
    Upper_ci_Absolute_R = math.tanh(math.atanh(Absolute_R_Square_SE) + zcrit * Absolute_R_Square_SE)

 
    results = {}
    results["Pearson Correlation"] = (Pearson_Correlation)
    results["t score"] = round(t_score_of_the_slope, 4)
    results["Degrees of Freedom"] = round(sample_size-2, 4)
    results["Pearson Correlation P-value"] = round(Pearson_pvalue, 4)
    results["Standrd Error of the Slope"] = round(Standard_error_slope, 4)
    results["Constant"] = round(constant, 4)
    results["Slope"] = round(Slope, 4)

    results["Standard Error Fisher "] = round(Fisher_SE, 4)
    results["Standard Error Fisher & Filon"] = round(Fisher_Filon_SE, 4)
    results["Standard Error Soper-I"] = round(Soper_1_SE, 4)
    results["Standard Error Soper-II"] = round(Soper_2_SE, 4)
    results["Standard Error Soper-III"] = round(Soper_3_SE, 4)
    results["Standard Error Hoteling"] = round(Hoteling, 4)
    results["Standard Error Ghosh"] = round(Ghosh_SE, 4)
    results["Standard Error Hedges"] = round(Hedges_SE, 4)
    results["Standard Error Bonett"] = round(Bonett_SE, 4)    
    results["Standard Error From Regression"] = round(Regression_SE, 4)

    results["Fisher's Zr"] = (Zr)
    results["Standard Error of Zr"] = (Standard_Error_ZR)
    results["Confidence Intervals Fisher (1921)"] = f"({round(Pearosn_Fisher_CI_lower, 4)}, {round(Pearosn_Fisher_CI_upper, 4)})"
    results["Confidence Intervals Olivoto"] = f"({round(Lower_Olivoto, 4)}, {round(Upperr_Olivoto, 4)})"
    results["Confidence Intervals for R-square (Olkin & Fin)"] = f"({round(Pearson_CI_Olkin_Fin_Lower, 4)}, {round(Pearson_CI_Olkin_Fin_Upper, 4)})"

    results["Common Language Effect Size (Dunlap, 1994)"] = (Common_Language_Effect_Size_Dunlap)
    results["Approximated r (Hedges & Olkin, 1985)"] = (Approximated_r)
    results["Counter_Null_EffectSize"] = (Counter_Null_EffectSize)
    results["Absolute R Square"] = (Absolute_R_Square)
    results["Absolute R Square SE"] = round(Absolute_R_Square_SE, 4)

    results["Confidence Intervals CLES"] = f"({round(Lower_ci_CLES, 4)}, {round(Upper_ci_CLES, 4)})"
    results["Confidence Intervals Absolute R"] = f"({round(Lower_ci_Absolute_R, 4)}, {round(Upper_ci_Absolute_R, 4)})"

    results["Confidence Intervals Bootstrapping"] = f"({round(Pearosn_Bootstrapping_CI[0], 4)}, {round(Pearosn_Bootstrapping_CI[1], 4)})"
    results["Confidence Intervals Bonett"] = f"({round(Lower_ci_Bonett, 4)}, {round(Upper_ci_Bonett, 4)})"
    results["Confidence Intervals Eta Lower"] = round(Pearson_CI_Eta_lower, 4)
    results["Confidence Intervals Eta Upper"] = round(Pearson_CI_Eta_upper, 4)

    result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
    return result_str

# 2. Different R_Square Estimators

def Rsquare_Estimation(Rsquare, sample_size, Number_Of_Predictors):
    dftotal = sample_size-1
    df_residual = sample_size - Number_Of_Predictors - 1
    df = sample_size - Number_Of_Predictors

    # Adjusted R square
    term1 = (sample_size - 3) * (1 - Rsquare) / df_residual
    Smith = 1 - (sample_size / (df)) * (1 - Rsquare) # Smith, 1929
    Ezekiel = 1 - (dftotal / df_residual) * (1 - Rsquare) #Ezekiel, 1930
    Wherry = 1 - (dftotal / df) * (1 - Rsquare) #Wherry, 1931
    olkin_pratt = 1 - term1 * special.hyp2f1(1, 1, (df + 1) / 2, 1 - Rsquare) #Olkin & Pratt, 1958
    olkin_pratt = 1 - term1 * special.hyp2f1(1, 1, (df + 1) / 2, 1 - Rsquare) #Olkin & Pratt, 1958

    Cattin = 1 - term1 * ((1 + (2 * (1 - Rsquare)) / df_residual) + ((8 * (1 - Rsquare) ** 2) / (df_residual * (df + 3)))) # Cattin, 1980 (Approximation to Olkin and Pratt)
    Pratt = 1 - (((sample_size - 3) * (1 - Rsquare)) / df_residual) * (1 + (2 * (1 - Rsquare)) / (df - 2.3)) # Pratt, 1964
    Herzberg = 1 - (((sample_size - 3) * (1 - Rsquare)) / df_residual) * (1 + (2 * (1 - Rsquare)) / (df+1)) # Herzberg, 1969
    Claudy =   1 - (((sample_size - 4) * (1 - Rsquare)) / df_residual) * (1 + (2 * (1 - Rsquare)) / (df+1)) # Herzberg, 1978
    def Alf_Graf_MLE(Rsquare, sample_size, Number_Of_Predictors):     # Alf and Graf 2002, MLE
        return opt.minimize_scalar(lambda rho: (1 - rho) ** (sample_size / 2) * (special.hyp2f1(0.5 * sample_size, 0.5 * sample_size, 0.5 * Number_Of_Predictors, rho * Rsquare)) * -1,bounds=(0, 1),method='bounded').x
    AlfGraf = Alf_Graf_MLE(Rsquare, sample_size, Number_Of_Predictors)

    # Squared Cross-Validity Coefficient
    Lord = 1 - (sample_size+Number_Of_Predictors+1) / (sample_size-Number_Of_Predictors-1) * (1-Rsquare) #Uhl & Eisenberg, 1970, also known as the Lord formula (it is most cited by this name)
    Lord_Nicholson = 1 - ((sample_size+Number_Of_Predictors +1) / sample_size) * (dftotal/df_residual) * (1-Rsquare) 
    Darlington_Stein = 1 - ((sample_size + 1) / sample_size) * (dftotal/df_residual) * ((sample_size-2) / (df-2)) * (1-Rsquare)
    Burket = (sample_size*Rsquare - Number_Of_Predictors) / (np.sqrt(Rsquare)*df) 
    Brown_Large_Sample = (((df - 3) * Ezekiel**2 +     Ezekiel)     / ((sample_size-2*Number_Of_Predictors - 2) * Ezekiel + Number_Of_Predictors))
    Brown_small_Sample = (((df - 3) * olkin_pratt**2 + olkin_pratt) / ((sample_size-2*Number_Of_Predictors - 2) * olkin_pratt + Number_Of_Predictors))
    Rozeboom = 1 - ((sample_size+Number_Of_Predictors) / df) * (1 - Rsquare) #Rozeboom, 1978
    Rozeboom2_Large_Sample = Ezekiel * ((1 + (Number_Of_Predictors/ (df-2)) * ((1-Ezekiel)/Ezekiel))**-1) #Rozeboom, 1981 
    Rozeboom2_small_Sample = olkin_pratt * ((1 + (Number_Of_Predictors/ (df-2)) * ((1-olkin_pratt)/olkin_pratt))**-1) #Rozeboom, 1981
    Claudy1_Large_Sample = (2* Ezekiel - (Rsquare))
    Claudy1_Small_Sample = (2* olkin_pratt - (Rsquare))

    results = {
        "Smith (1928)": Smith,
        "Ezekiel (1930)": Ezekiel,
        "Wherry (1931)": Wherry,
        "Olkin & Pratt (1958)": olkin_pratt,
        "Olkin & Pratt, Pratt's Approximation (1964)": Pratt,
        "Olkin & Pratt, Herzberg's Approximation (1968)": Herzberg,
        "Olkin & Pratt, Claudy's Approximation (1978)": Claudy,
        "Olkin & Pratt, Cattin's Approximation (1980)": Cattin,
        "Alf and Graf MLE (2002)": AlfGraf,
        "Lord(1950)": Lord, 
        "Lord_Nicholson(1960)": Lord_Nicholson,
        "Darlington/Stein (1967/1960)": Darlington_Stein,
        "Burket (1964)": Burket,
        "Brown_Large_Samples (1975)": Brown_Large_Sample,
        "Brown_Small_Samples (1975)": Brown_small_Sample,
        "Rozeboom I (1978)": Rozeboom,
        "Rozeboom II Large Samples- (1978)": Rozeboom2_Large_Sample,
        "Rozeboom II Small Samples (1978)": Rozeboom2_small_Sample, 
        "Claudy-I, Large Samples (1978)": Claudy1_Large_Sample,
        "Claudy-I, Small Samples (1978)": Claudy1_Small_Sample,   
    }

    result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
    return result_str



# 3. Alternative Robust Measures of Pearson Correlation (Including Skipped Correlation)
def Robust_Measures_Interval(x,y, confidence_level = 0.95): 
    
    # A. Percentage Bend Correlation
    sample_size = len(x)
    Omega_Hat_X= sorted((abs(np.array(x) - np.median(x))))[math.floor((1 - 0.2**2) * sample_size) - 1]
    psix = (x-np.array(np.median(x)))/Omega_Hat_X
    i1 = sum((x_i - np.median(x)) / Omega_Hat_X < -1 for x_i in x)
    i2 = sum((x_i - np.median(np.median(x))) / Omega_Hat_X > 1 for x_i in x)
    Sx = np.sum(np.where(np.logical_or(psix < -1, psix > 1), 0, x))
    Phi_x = (Omega_Hat_X * (i2 - i1) + Sx) / (sample_size - i1 - i2)
    Ai = np.clip((x - Phi_x) / Omega_Hat_X, -1, 1)

    Omega_Hat_Y = sorted((abs(np.array(y) - np.median(y))))[math.floor((1 - 0.2**2) * sample_size) - 1]
    psiy = (y - np.array(np.median(y))) / Omega_Hat_Y
    i1_y = sum((y_i - np.median(y)) / Omega_Hat_Y < -1 for y_i in y)
    i2_y = sum((y_i - np.median(np.median(y))) / Omega_Hat_Y > 1 for y_i in y)
    Sy = np.sum(np.where(np.logical_or(psiy < -1, psiy > 1), 0, y))
    Phi_y = (Omega_Hat_Y * (i2_y - i1_y) + Sy) / (sample_size - i1_y - i2_y)
    Bi = np.clip((y - Phi_y) / Omega_Hat_Y, -1, 1)

    Percentage_Bend_Correlation = np.sum(Ai * Bi)/np.sqrt(np.sum(Ai**2) * np.sum(Bi**2))

    # B. Winsorized Correlation
    lower_items = int(np.floor(0.2 * sample_size)) + 1
    upper_items = len(x) - lower_items + 1
    sorted_x = np.sort(x)
    sorted_y = np.sort(y)
    
    Winzorized_X = np.where((x <= sorted_x[lower_items - 1]) | (x >= sorted_x[upper_items - 1]),np.where(x <= sorted_x[lower_items - 1], sorted_x[lower_items - 1], sorted_x[upper_items - 1]),x)
    Winzorized_Y = np.where((y <= sorted_y[lower_items - 1]) | (y >= sorted_y[upper_items - 1]),np.where(y <= sorted_y[lower_items - 1], sorted_y[lower_items - 1], sorted_y[upper_items - 1]),y)
    Winzorized_Correlation = np.corrcoef(Winzorized_X, Winzorized_Y)[0, 1]

    # C. Biweight Midcorrelation (Verified with asbio for r)
    Ui = (x - np.median(x)) / (9 * t.ppf(0.75, 100000) * 1.4826*np.median(np.abs(x- np.median(x))))
    Vi = (y - np.median(y)) / (9 * t.ppf(0.75, 100000) * 1.4826*np.median(np.abs(y - np.median(y))))
    Ai = np.where((Ui <= -1) | (Ui >= 1), 0, 1)
    Bi = np.where((Vi <= -1) | (Vi >= 1), 0, 1)

    Sxx = (np.sqrt(sample_size) * np.sqrt(sum((Ai * ((x - np.median(x))**2)) * ((1 - Ui**2)**4))) /  abs(sum(Ai * (1 - Ui**2) * (1 - 5 * Ui**2))))**2
    Syy = (np.sqrt(sample_size) * np.sqrt(sum((Bi * ((y - np.median(y))**2)) * ((1 - Vi**2)**4))) / (abs(sum(Bi * (1 - Vi**2) * (1 - 5 * Vi**2)))))**2
    Sxy = sample_size * sum((Ai *(x - np.median(x))) * ((1-Ui**2)**2) * (Bi* (y-np.median(y))) * ((1-Vi**2)**2)) / (sum((Ai* (1-Ui**2)) * (1-5*Ui**2)) * sum((Bi* (1-Vi**2)) * (1 - 5 * Vi**2)))
    Biweight_midcorrelation = Sxy / (np.sqrt(Sxx * Syy))

    # D. Gausian Rank Correlation 
    Normalized_X = norm.ppf((np.argsort(x) + 1) / (len(x) + 1))
    Normalized_Y = norm.ppf((np.argsort(y) + 1) / (len(y) + 1))
    Gaussian_Rank_Correlation = pearsonr(Normalized_X, Normalized_Y)
    zcrit = t.ppf(1 - (1 - confidence_level) / 2, 100000)
    Normalized_X = norm.ppf((rankdata(x) / (len(x) + 1)))
    Normalized_Y = norm.ppf((rankdata(y) / (len(y) + 1)))
    Gaussian_Rank_Correlation, GRS_pvalue = pearsonr(Normalized_X, Normalized_Y)

    # E. Robust correlational median estimator (10% trimming)
    trimmedx = x.astype(float)
    trimmedy = y.astype(float)
    trimmedx[np.logical_or(trimmedx <= np.percentile(trimmedx, 10), trimmedx >= np.percentile(trimmedx, 90))] = np.nan
    trimmedy[np.logical_or(trimmedy <= np.percentile(trimmedy, 10), trimmedy >= np.percentile(trimmedy, 90))] = np.nan
    trimmed_data = pd.DataFrame({'x': trimmedx, 'y': trimmedy}).dropna()

    median_trimmed_x = np.median(trimmed_data['x'])
    median_trimmed_y = np.median(trimmed_data['y'])

    Trimmed_Covariance = np.sum((np.array(trimmed_data['x']) - median_trimmed_x) * (np.array(trimmed_data['y']) - median_trimmed_y))
    Squared_Deviation_Median_x = np.sum((median_trimmed_x -   np.array(trimmed_data['x']))**2)
    Squared_Deviation_Median_y = np.sum((median_trimmed_y - np.array(trimmed_data['y']))**2)
    rCME = float(((Trimmed_Covariance)) / (np.sqrt((Squared_Deviation_Median_x)*Squared_Deviation_Median_y)))

    # F. MAD correlation Coefficient
    Median_x = np.median(x)
    Median_y = np.median(y)
    MAD_x = np.median(abs(x - Median_x))
    MAD_y = np.median(abs(y - Median_y))

    u = ((x - np.median(x)) / (MAD_x*np.sqrt(2))) + ((y - np.median(y)) / (MAD_y*np.sqrt(2)))
    v = ((x - np.median(x)) / (MAD_x*np.sqrt(2))) - ((y - np.median(y)) / (MAD_y*np.sqrt(2)))
    median_u= np.median(u)
    median_v= np.median(v)

    rmad = ((np.median(abs(u - median_u)))**2 -  (np.median(abs(v - median_v)))**2) / ((np.median(abs(u - median_u)))**2 + (np.median(abs(v - median_v)))**2) 
    rmed = ((np.median(abs(u)))**2 - (np.median(abs(v)))**2) / ((np.median(abs(u)))**2 + (np.median(abs(v)))**2)

    # Statistics and p-values 
    degrees_of_freedom = sample_size-2
    degrees_of_freedom_WC = sample_size - 2 * int(np.floor(0.2 * sample_size)) - 2
    degrees_of_freedom_rCME = len(np.array(trimmed_data['x'])) - 2
  
    T_Statistic_PB = Percentage_Bend_Correlation * np.sqrt((degrees_of_freedom) / (1 - Percentage_Bend_Correlation**2))
    T_Statistic_WC = Winzorized_Correlation * np.sqrt((degrees_of_freedom_WC - 2) / (1 - Winzorized_Correlation**2))
    T_statistic_BM = Biweight_midcorrelation* np.sqrt((degrees_of_freedom - 2) / (1 - Biweight_midcorrelation**2))
    T_Statistic_rCME = rCME * np.sqrt((degrees_of_freedom_rCME - 2) / (1 - rCME**2))
    T_Statistic_rmed = rmed * np.sqrt((degrees_of_freedom - 2) / (1 - rmed**2))
    T_statistic_rmad = rmad* np.sqrt((degrees_of_freedom - 2) / (1 - rmad**2))
    T_statistic_Gaussian_Rank_Correlation = Gaussian_Rank_Correlation * np.sqrt((degrees_of_freedom - 2) / (1 - Gaussian_Rank_Correlation**2))

    P_value_BM =        t.sf(abs(T_statistic_BM), degrees_of_freedom)    
    P_Value_Winzoried = t.sf(abs(T_Statistic_WC), degrees_of_freedom_WC)
    P_value_PB  =       t.sf(abs(T_Statistic_PB), degrees_of_freedom)
    P_value_rmed =      t.sf(abs(T_Statistic_rmed), degrees_of_freedom)    
    P_Value_rmad =      t.sf(abs(T_statistic_rmad), degrees_of_freedom)
    P_value_rCME  =     t.sf(abs(T_Statistic_rCME), degrees_of_freedom_rCME)

    # Confidence Intervals (Fisher Based)
    SE_Bonett_BM = (1 - Biweight_midcorrelation**2) / np.sqrt(sample_size-3)
    SE_Bonett_GRC = (1 - Gaussian_Rank_Correlation**2) / np.sqrt(sample_size-3)
    SE_Bonett_Winsorized = (1 - Winzorized_Correlation**2) / np.sqrt(degrees_of_freedom_WC-1)
    SE_Bonett_PB = (1 - Percentage_Bend_Correlation**2) / np.sqrt(sample_size-3)
    SE_Bonett_rmed = (1 - rmed**2) / np.sqrt(sample_size-3)
    SE_Bonett_rmad = (1 - rmad**2) / np.sqrt(sample_size-3)
    SE_Bonett_rCME = ((1 - rCME**2) / np.sqrt(degrees_of_freedom_rCME-1))

    Lower_ci_BM = math.tanh(math.atanh(Biweight_midcorrelation) - zcrit * SE_Bonett_BM)
    Upper_ci_BM = math.tanh(math.atanh(Biweight_midcorrelation) + zcrit * SE_Bonett_BM)
    Lower_ci_GRC = math.tanh(math.atanh(Gaussian_Rank_Correlation) - zcrit * SE_Bonett_GRC)
    Upper_ci_GRC = math.tanh(math.atanh(Gaussian_Rank_Correlation) + zcrit * SE_Bonett_GRC)
    Lower_ci_Winsorized = math.tanh(math.atanh(Winzorized_Correlation) - zcrit * SE_Bonett_Winsorized)
    Upper_ci_Winsorized = math.tanh(math.atanh(Winzorized_Correlation) + zcrit * SE_Bonett_Winsorized)
    Lower_ci_PB = math.tanh(math.atanh(Percentage_Bend_Correlation) - zcrit * SE_Bonett_PB)
    Upper_ci_PB = math.tanh(math.atanh(Percentage_Bend_Correlation) + zcrit * SE_Bonett_PB)
    #Lower_ci_rmed = math.tanh(math.atanh(rmed) - zcrit * SE_Bonett_rmed)
    #Upper_ci_rmed = math.tanh(math.atanh(rmed) + zcrit * SE_Bonett_rmed)
    #Lower_ci_rmad = math.tanh(math.atanh(rmad) - zcrit * SE_Bonett_rmad)
    #Upper_ci_rmad = math.tanh(math.atanh(rmad) + zcrit * SE_Bonett_rmad)
    #Lower_ci_rCME = math.tanh(math.atanh(rCME) - zcrit * SE_Bonett_rCME)
    #Upper_ci_rCME = math.tanh(math.atanh(rCME) + zcrit * SE_Bonett_rCME)

    results = {
    "Trimmed Data": trimmed_data,
    "Robust Correlation Median Estimator": rCME,
    "Robust Correlational median estimator SE": "The Trimmed Data is Too Small, Can't Calculate the Variance" if degrees_of_freedom_rCME <=1 else SE_Bonett_rCME,
    "Robust Correlational median Estimator Stat": "The Trimmed Data is Too Small" if degrees_of_freedom_rCME <=1 else T_Statistic_rCME ,
    "Robust Correlational median estimator p-value": "The Trimmed Data is Too Small" if degrees_of_freedom_rCME <=1 else P_value_rCME,
    "Robust Correlational median estimator Degrees of Freedom": degrees_of_freedom_rCME ,
    #"Robust Correlational median estimator CI": (Lower_ci_rCME, Upper_ci_rCME),
    
    "Biweight Midcorrelation": Biweight_midcorrelation,
    "Biweight Midcorrelation SE": SE_Bonett_BM,
    "Biweight Midcorrelation Statistic": T_statistic_BM,
    "Biweight Midcorrelation p-value": P_value_BM,
    "Biweight Midcorrelation CI": (Lower_ci_BM, Upper_ci_BM),
    
    "Gaussian Rank Correlation": Gaussian_Rank_Correlation,
    "Gaussian Rank Correlation SE": SE_Bonett_GRC,
    "Gaussian Rank Correlation Statistic": T_statistic_Gaussian_Rank_Correlation,
    "Gaussian Rank Correlation p-value": GRS_pvalue,
    "Gaussian Rank Correlation CI": (Lower_ci_GRC, Upper_ci_GRC),
    
    "Winsorized Correlation": Winzorized_Correlation,
    "Winsorized Correlation SE": SE_Bonett_Winsorized,
    "Winsorized Correlation Statistic": T_Statistic_WC,
    "Winsorized Correlation p-value": P_Value_Winzoried,
    "Winsorized Correlation CI": (Lower_ci_Winsorized, Upper_ci_Winsorized),
    
    "Percentage Bend Correlation": Percentage_Bend_Correlation,
    "Percentage Bend Correlation SE": SE_Bonett_PB,
    "Percentage Bend Correlation Statistic": T_Statistic_PB,
    "Percentage Bend Correlation p-value": P_value_PB,
    "Percentage Bend Correlation CI": (Lower_ci_PB, Upper_ci_PB),
    
    "MAD correlation Coefficient": rmad,
    "MAD correlation Coefficient SE": SE_Bonett_rmad,
    "MAD correlation Coefficient Statistic": T_statistic_rmad,
    "MAD correlation Coefficient p-value": P_Value_rmad,
    #"MAD correlation Coefficient CI": (Lower_ci_rmad, Upper_ci_rmad),
    
    "Robust correlational median estimator": rmed,
    "Robust correlational median estimator SE": SE_Bonett_rmed,
    "Robust correlational median estimator Statistic": T_Statistic_rmed,
    "Robust correlational median estimator p-value": P_value_rmed,
    #"Robust correlational median estimator CI": (Lower_ci_rmed, Upper_ci_rmed)
     }

    result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
    return result_str


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
    Pearson_Skipped_Correlation_IQR, pval_Skipped_Correlation_IQR = pearsonr(X[~outliers_iqr, 0], X[~outliers_iqr, 1])
    Pearson_Skipped_Correlation_MAD, pval_Skipped_Correlation_MAD = pearsonr(X[~outliers_mad, 0], X[~outliers_mad, 1])

    # Confidence Intervals using Bonett (2008)
    zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
    sample_size_Skipped_iqr = len(X[~outliers_iqr, 0])
    sample_size_Skipped_mad = len(X[~outliers_mad, 0])

    SE_Bonett_Skipped_Correlation_IQR = (1 - Pearson_Skipped_Correlation_IQR**2) / np.sqrt(sample_size_Skipped_iqr-3)
    SE_Bonett_Skipped_Correlation_MAD = (1 - Pearson_Skipped_Correlation_MAD**2) / np.sqrt(sample_size_Skipped_mad-3)

    Lower_ci_Skipped_IQR = math.tanh(math.atanh(Pearson_Skipped_Correlation_IQR) - zcrit * SE_Bonett_Skipped_Correlation_IQR)
    Upper_ci_Skipped_IQR= math.tanh(math.atanh(Pearson_Skipped_Correlation_IQR) + zcrit * SE_Bonett_Skipped_Correlation_IQR)
    Lower_ci_Skipped_MAD = math.tanh(math.atanh(Pearson_Skipped_Correlation_MAD) - zcrit * SE_Bonett_Skipped_Correlation_MAD)
    Upper_ci_Skipped_MAD = math.tanh(math.atanh(Pearson_Skipped_Correlation_MAD) + zcrit * SE_Bonett_Skipped_Correlation_MAD)

    results = {}

    results["Skipped Correlation IQR based"] = Pearson_Skipped_Correlation_IQR
    results["Skipped Correlation MAD based"] = Pearson_Skipped_Correlation_MAD
    results["Skipped Correlation IQR based p-value"] = pval_Skipped_Correlation_IQR
    results["Skipped Correlation MAD based p-value"] = pval_Skipped_Correlation_MAD
    results["Skipped Correlation IQR based CI's"] = [Lower_ci_Skipped_IQR, Upper_ci_Skipped_IQR]
    results["Skipped Correlation MAD based CI's"] = [Lower_ci_Skipped_MAD, Upper_ci_Skipped_MAD]

    result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
    return result_str


class Int_Ratio_Correlation():
    @staticmethod
    def IntervaRatio_from_Data(params: dict) -> dict:
        
        # Set Params:
        Variable1 = params["Variable 1"]
        Variable2 = params["Variable 2"]
        confidence_level_percentages = params["Confidence Level"]

        # Convrt the variables into vectors
        confidence_level = confidence_level_percentages/ 100

        R_Square = pearsonr(Variable1,Variable2)[0]**2
        sample_size = len(Variable1)
        Rsqare_Estiamtion_Output = Rsquare_Estimation(R_Square, sample_size, 1)

        # Calculate Effect Sizes using All Functions        
        skipped_Correlation_measures = skipped_Correlation(Variable1, Variable2, confidence_level)
        Robust_Correaltions_Output = Robust_Measures_Interval(Variable1,Variable2,confidence_level)        
        Main_Procedure_Output = pearson_correlation(Variable1, Variable2,confidence_level)

        results = {}
        results["Skipped Corrleation "] =  skipped_Correlation_measures
        results["_______________________"] = ""
        results["Robust Correlation"] =  Robust_Correaltions_Output
        results["______________________"] = ""
        results["Rsqare Estiamtion Output"] =  Rsqare_Estiamtion_Output
        results["________________________"] = ""
        results["General Procedre Pearson"] =  Main_Procedure_Output

        return results



