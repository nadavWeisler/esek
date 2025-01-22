
#########################################
##### Multiple Correlation R-Square #####
#########################################

from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import f, norm, ncf
import scipy.special as special
import scipy.optimize as opt


def Non_Central_CI_F(F_Statistic, df1, df2, confidence_level):
    Upper_Limit = 1 - (1 - confidence_level) / 2
    Lower_Limit = 1 - Upper_Limit
    Lower_CI_Difference_Value = 1

    def Lower_CI(F_Statistic, df1, df2, Upper_Limit, Lower_CI_Difference_Value):
        Lower_Bound = [0.001, F_Statistic / 2, F_Statistic]       
        while ncf.cdf(F_Statistic, df1, df2, Lower_Bound[0]) < Upper_Limit:
            return [0, f.cdf(F_Statistic, df1, df2)] if f.cdf(F_Statistic, df1, df2) < Upper_Limit else None
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
    Lower_CI_Final = Lower_CI(F_Statistic, df1, df2, Upper_Limit, Lower_CI_Difference_Value)[0]
    Upper_CI_Final = Upper_CI(F_Statistic, df1, df2, Lower_Limit, Lower_CI_Difference_Value)[0]

    return Lower_CI_Final, Upper_CI_Final


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


class R_Squared():
    @staticmethod
    def R_Squared_From_Data(params: dict) -> dict:
        
        Data = params["Data"]
        Conidence_Level_Percentages = params["Confidence Level"]

        # Preperation
        Conidence_Level = Conidence_Level_Percentages / 100
        Y = np.array(Data["Predicted"])
        Predictors = Data[[col for col in Data.columns if col.startswith('x')]].values
        r_squared = LinearRegression().fit(Predictors, Y).score(Predictors, Y)
        Number_Of_Predictors = Predictors.shape[1]  # Number of columns in predictors
        Sample_Size = len(Y)
        df1 = Number_Of_Predictors
        df2 = Sample_Size - Number_Of_Predictors
        F_Score = (r_squared/Number_Of_Predictors) / ((1-r_squared) / (Sample_Size-Number_Of_Predictors-1))
        p_value = f.sf(F_Score, df1, df2)

        R_Square_Estimation_output = Rsquare_Estimation(r_squared, Sample_Size, Number_Of_Predictors)

        # Confidence Intervals for R Square 
        Z_crit = norm.ppf(1 - (1-Conidence_Level)/2)

        # Method 1 - Wald Type Method
        ASE_Wishart = np.sqrt((4*r_squared*(1-r_squared)**2*(Sample_Size-Number_Of_Predictors-1)**2) / ((Sample_Size**2-1)*(Sample_Size+3))) # This Approximation originally appeared in Wishart 1931
        lower_ci_wald_type = r_squared - Z_crit * ASE_Wishart
        upper_ci_wald_type = r_squared + Z_crit * ASE_Wishart

        # Method 2 - Fisher
        ASE_Algina = np.sqrt(4/Sample_Size)
        Zr_square = np.log( (1+np.sqrt(r_squared)) / (1-np.sqrt(r_squared)) ) #Algina 1999 Approximation
        lower_ci_Fisher_Zr_squared = (Zr_square - Z_crit * ASE_Algina)
        upper_ci_Fisher_Zr_squared = (Zr_square + Z_crit * ASE_Algina)
        lower_ci_Fisher = ((np.exp(lower_ci_Fisher_Zr_squared) - 1) / (np.exp(lower_ci_Fisher_Zr_squared) + 1))**2
        upper_ci_Fisher = ((np.exp(upper_ci_Fisher_Zr_squared) - 1) / (np.exp(upper_ci_Fisher_Zr_squared) + 1))**2

        # Method 3 - Scaled Central

        # Method 4 - Scaled Non-Central
        


        results = {}
        results["R_Squared"] = r_squared
        results["Number Of Predictors"] = Number_Of_Predictors
        results["Sample_Size"] = Sample_Size
        results["Degrees of Freedom 1"] = df1
        results["Degrees of Freedom 2"] = df2
        results["F score"] = F_Score
        results["p_value"] = p_value
        results["R Square Estimation"] = R_Square_Estimation_output


        return results






