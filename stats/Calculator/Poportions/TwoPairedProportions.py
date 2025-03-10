
#########################################################
# Effect Size for Two Paired/Dependent Proportions#######
#########################################################

import numpy as np
import math
from scipy.stats import norm, chi2, beta
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

def Main_Two_Dep_Proportions_From_Parameters (proportion_sample_1, proportion_sample_2, proportion_of_yes_yes, sample_size, confidence_level_percentages, difference_in_population): 
        
        confidence_level = confidence_level_percentages / 100

        ############################
        # 1. Descreptive Statistics#
        ############################
        
        yes_yes = proportion_of_yes_yes * sample_size
        yes_no = (proportion_sample_1 * sample_size) - yes_yes
        no_yes = (proportion_sample_2 * sample_size) - yes_yes
        no_no = sample_size - (yes_yes + yes_no + no_yes)

        sample_size = yes_yes + yes_no + no_yes + no_no
        Delta = 0.95     #This is just for Debugging

        n1total = yes_yes + yes_no
        n2total = yes_yes + no_yes
        
        Proportion_Matching = (yes_yes + no_no) / sample_size
        Proportion_Not_Matching = 1 - Proportion_Matching
        contingency_table = [[yes_yes, yes_no], [no_yes, no_no]]
        x = np.array(contingency_table).flatten()
        p1 = yes_yes+yes_no; p2=yes_yes+no_yes; q1 = no_yes+no_no; q2=yes_no+no_no
        difference_between_proportions = proportion_sample_1 - proportion_sample_2

        #############################################################
        # 2. Inferntial Statistics ##################################
        #############################################################
        
        # Significance # Mcnemar Test (2x2)
        yes_yes_rounded = round(proportion_of_yes_yes * sample_size)
        yes_no_rounded = round((proportion_sample_1 * sample_size) - yes_yes)
        no_yes_rounded = round((proportion_sample_2 * sample_size) - yes_yes)
        no_no_rounded = round(sample_size - (yes_yes + yes_no + no_yes))

        contingency_table_rounded = [[yes_yes_rounded, yes_no_rounded], [no_yes_rounded, no_no_rounded]]
        exact_mcnemar = mcnemar(contingency_table_rounded, exact=True)
        mcnemar_chi_square = ((abs(yes_no - no_yes))**2) / (yes_no+no_yes)
        mcnemar_chi_square_corrected = ((abs(yes_no - no_yes) - 1)**2) / (yes_no+no_yes)
        mcnemar_pvalue = 1 - chi2.cdf(mcnemar_chi_square, 1)
        mcnemar_pvalue_corrected = 1 - chi2.cdf(mcnemar_chi_square_corrected, 1)


        #############################################################
        # 3. Difference Between Paired Proportions ##################
        #############################################################
  
        # Confidence Intervals
        z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))

        # 1. Wald CI's
        Standard_Error_Wald = np.sqrt((yes_no + no_yes) - ((yes_no - no_yes)**2) / sample_size) / sample_size
        LowerCi_WALD = max(difference_between_proportions - z_critical_value *Standard_Error_Wald,-1 )
        UpperCi_WALD = min(difference_between_proportions + z_critical_value *Standard_Error_Wald,1 )

        # 2. Wald with CC correction (Fleiss et al., 2003)
        Standard_Error_Wald_Corrected = np.sqrt((yes_no + no_yes) - ((yes_no - no_yes)**2) / sample_size) / sample_size
        LowerCi_WALD_Corrected = max(difference_between_proportions - z_critical_value *Standard_Error_Wald_Corrected - (1/sample_size),-1 )
        UpperCi_WALD_Corrected = min(difference_between_proportions + z_critical_value *Standard_Error_Wald_Corrected + (1/sample_size),1 )
        
        # 3. Wald with Yates correction
        Standard_Error_Wald_Corrected_yates = np.sqrt((yes_no + no_yes) - ((yes_no - no_yes - 1)**2) / sample_size) / sample_size
        LowerCi_WALD_Corrected_Yates = max(difference_between_proportions - z_critical_value *Standard_Error_Wald_Corrected_yates,-1 )
        UpperCi_WALD_Corrected_Yates = min(difference_between_proportions + z_critical_value *Standard_Error_Wald_Corrected_yates,1 )
        
        # 4. Agresti & Min (2005)
        Standard_Error_AM =np.sqrt(((yes_no+0.5) + (no_yes+0.5)) - (((yes_no+0.5) - (no_yes+0.5))**2) / (sample_size+2)) / (sample_size+2)
        LowerCi_AM = max(((yes_no+0.5) - (no_yes+0.5)) /  (sample_size + 2) - z_critical_value * Standard_Error_AM,-1 )
        UpperCi_AM = min(((yes_no+0.5) - (no_yes+0.5)) /  (sample_size + 2) + z_critical_value * Standard_Error_AM, 1 )

        # 5. Bonett & Price (2005)
        p1_adjusted = (yes_no + 1) / (sample_size + 2)
        p2_adjusted = (no_yes + 1) / (sample_size + 2)
        Standard_Error_BP = np.sqrt((p1_adjusted+p2_adjusted-(p2_adjusted-p1_adjusted)**2)/(sample_size+2))
        LowerCi_BP = max(p1_adjusted - p2_adjusted - z_critical_value * Standard_Error_BP,-1 )
        UpperCi_BP = min(p1_adjusted - p2_adjusted + z_critical_value * Standard_Error_BP,1 )

        # 6. Newcomb, Square and Add
        A1 = (2 * sample_size * ((n1total)/sample_size) + z_critical_value**2) / (2 * sample_size + 2 * z_critical_value**2)
        B1 = (z_critical_value * np.sqrt(z_critical_value**2 + 4 * sample_size * (n1total/sample_size) * (1 - (n1total/sample_size)))) / (2 * sample_size + 2 * z_critical_value**2)  
        A2 = (2 * sample_size * ((n2total)/sample_size) + z_critical_value**2) / (2 * sample_size + 2 * z_critical_value**2)
        B2 = (z_critical_value * np.sqrt(z_critical_value**2 + 4 * sample_size * (n2total/sample_size) * (1 - (n2total/sample_size)))) / (2 * sample_size + 2 * z_critical_value**2)
        lower_p1 = A1 - B1
        upper_p1 = A1 + B1
        lower_p2 = A2 - B2
        upper_p2 = A2 + B2

        if n1total == 0 or n2total == 0 or (sample_size-n1total) == 0 or (sample_size - n2total) == 0:
                products_correction= 0
        else:
            marginals_product = n1total*n2total*(sample_size-n1total)*(sample_size-n2total)
            cells_product = yes_yes*no_no - no_yes*no_yes
            if cells_product > sample_size / 2:
                products_correction = (cells_product - sample_size / 2) / np.sqrt(marginals_product)
            elif cells_product >= 0 and cells_product <= sample_size / 2:
                products_correction = 0
            else:
                products_correction = cells_product / np.sqrt(marginals_product)

        LowerCi_newcomb = difference_between_proportions - np.sqrt((proportion_sample_1 - lower_p1)**2 + (upper_p2 - proportion_sample_2)**2 - 2 * products_correction * (proportion_sample_1 - lower_p1) * (upper_p2 - proportion_sample_2))
        UpperCi_newcomb = difference_between_proportions + np.sqrt((proportion_sample_2 - lower_p2)**2 + (upper_p1 - proportion_sample_1)**2 - 2 * products_correction * (proportion_sample_2 - lower_p2) * (upper_p1 - proportion_sample_1))
        
        
        # Optional - Can check this built in Functions for more version of the mcnemar test
        #data = [[p1, q1], [p2, q2]]
        #a = (mcnemar(data, exact=True))
        #b = (mcnemar(data, exact=False, correction=False))
        #c = (mcnemar(data, exact=False, correction=True))



        ##############################
        # 4. Matched Pairs Odds Ratio#
        ##############################
        
        # 1. Matched Pairs Odds Ratio
        Match_Pairs_Odds_Ratio = yes_no / no_yes
        
        # 2. adjusted Matched Pairs Odds Ratio (see Jewell, 1984)
        Match_Pairs_Odds_Ratio_Adjusted = (yes_no / (no_yes+1))
        
        # 3. Bias Corrected maximum likelihood - Matched Pairs Odds Ratio (Parr & Tolley, 1982)
        Odds_Ratio_Bias_Corrected = Match_Pairs_Odds_Ratio * (1-no_yes**-1)

        # 4. Jacknife odds ratio (Miller, 1974 & Jewell, 1984)
        Odds_Ratio_Matched_Pairs_Jackknife = Match_Pairs_Odds_Ratio - ((sample_size - 1) / (no_yes-1)) * (Match_Pairs_Odds_Ratio/sample_size)
                
        # Standard Errors of Odds Ratios
        Standrd_Error_Odds_ratio = np.sqrt(1/yes_no + 1/no_yes) #Ejigou and McHugh, 1977
        Adjusted_Standrd_Error_Odds_ratio = np.sqrt(1/(yes_no+1) + 1/(no_yes+1)) #Jewell, 1984/1986


        # Confidence Intervals for the matched pairs Odds Ratio
        #######################################################

        # 1. Binomial Based CI's (Altman, 2000, p.66) - The most popular method in Statistical Softwares
        # Also called the Clopper-Pearson Method (Fagerland, 2014) or Fisher Exact Method (openEpi) or Inferntial Model (IM) in Chen et al., 2021
        ci_lower, ci_upper = proportion_confint(count=yes_no, nobs=yes_no+no_yes, alpha=0.05, method="beta")
        CI_odds_ratio_binomial_lower = ci_lower/(1-ci_lower)
        CI_odds_ratio_binomial_upper = ci_upper/(1-ci_upper)

        # 2. Wald Logarithmic Method
        # This Method is also very popular and is called in different names such as the Wald Method (Fagerland, 2014), Taylor series confidence interval (Martin & Austin, 1991) or Delta method (Chen et al., 2021)
        # A taylor series approximation is applied to the varience and therfore the name 
        # Also see the reference here: (An Efficient Program for Computing Conditional Maximum Likelihood Estimates and Exact Confidence Limits for a Common Odds Ratio David Martin1 and Harland Austin, 1991) see also Flanders, 1985
        CI_odds_ratio_log_lower = Match_Pairs_Odds_Ratio * math.exp(-z_critical_value*Standrd_Error_Odds_ratio)
        CI_odds_ratio_log_upper = Match_Pairs_Odds_Ratio * math.exp(z_critical_value*Standrd_Error_Odds_ratio)
        
        # 3. Wald with Laplace adjustment
        Lower_Wald_Laplace = math.exp(np.log((yes_no+1)/(no_yes+1)) - z_critical_value * np.sqrt((1/(yes_no + 1)) + (1 / (no_yes + 1))))
        Upper_Wald_Laplace = math.exp(np.log((yes_no+1)/(no_yes+1)) + z_critical_value * np.sqrt((1/(yes_no + 1)) + (1 / (no_yes + 1))))

        # 4. Rigby & Robinsone (2000) Mcnemar Test Based CI's # RIGBY, A. S., & ROBINSONOE, M. B. (2000). Statistical methods in epidemiology. IV. Confounding and the matched pairs odds ratio. Disability and Rehabilitation, 22(6), 259–265. https://doi.org/10.1080/096382800296719
        CI_Mcnemar_odds_ratio_lower = Match_Pairs_Odds_Ratio**(1-z_critical_value/np.sqrt(mcnemar_chi_square))
        CI_Mcnemar_odds_ratio_upper = Match_Pairs_Odds_Ratio**(1+z_critical_value/np.sqrt(mcnemar_chi_square))
        
        # 5. Rigby & Robinsone (2000) Mcnemar corrected Test Based CI's
        CI_Mcnemar_corrected_odds_ratio_lower = Match_Pairs_Odds_Ratio**(1-z_critical_value/np.sqrt(mcnemar_chi_square_corrected))
        CI_Mcnemar_corrected_odds_ratio_upper = Match_Pairs_Odds_Ratio**(1+z_critical_value/np.sqrt(mcnemar_chi_square_corrected))
        
        # 6. The Score Method also known as the Wilson score Method (Fagerland, 2014)
        A = (2*yes_no*no_yes) + z_critical_value**2*(yes_no+no_yes)
        lower_ci_score_method = (A - np.sqrt(A**2 - ((2*yes_no*no_yes)**2))) / (2*no_yes**2)
        upper_ci_score_method = (A + np.sqrt(A**2 - ((2*yes_no*no_yes)**2))) / (2*no_yes**2)
        
        # Fiducial Intervals (Chen et al., 2021)
        beta_lower = beta.ppf((1-confidence_level) / 2, yes_no + 0.5, no_yes + 0.5)
        beta_upper = beta.ppf(1 - (1-confidence_level) / 2, yes_no + 0.5, no_yes + 0.5)
        Fiducial_lower = beta_lower / (1 - beta_lower)
        Fiducial_upper = beta_upper / (1 - beta_upper)
        
        
        # Check Where this function came from?     
        no_yes_adjusted = no_yes + 1
        A = (2*yes_no*no_yes_adjusted) + z_critical_value**2*(yes_no+no_yes_adjusted)
        lower_ci_score_adjusted_method = (A - np.sqrt(A**2 - ((2*yes_no*no_yes_adjusted)**2))) / (2*no_yes_adjusted**2)
        upper_ci_score_adjusted_method = (A + np.sqrt(A**2 - ((2*yes_no*no_yes_adjusted)**2))) / (2*no_yes_adjusted**2)


        ###################################################################################
        # 5. Matched Pairs Risk Ratio / Ratio of Prportions / Relative Risk  ##############
        ###################################################################################
        
        Matched_Pairs_Relative_Risk = (yes_yes+yes_no)/ (yes_yes+no_yes)
        sample_size = yes_yes+yes_no+no_no+no_yes


        # Confidence Intervals for the matched pairs Relative Risk/Risk Ratio
        
        # 1. Wald Method
        Standard_Error_Wald = math.sqrt((yes_no + no_yes) / (n1total * n2total))
        lower_wald_RR = Matched_Pairs_Relative_Risk * math.exp(-z_critical_value * Standard_Error_Wald)
        upper_wald_RR = Matched_Pairs_Relative_Risk * math.exp(z_critical_value * Standard_Error_Wald)
        
        # 2+3. Bonett & Price Wilson Score Based CI's with and withut Continuity Correction
        n_star = sample_size - no_no
        A = math.sqrt((yes_no + no_yes + 2) / ((n1total + 1) * (n2total + 1)))
        B = math.sqrt((1 - (n1total + 1) / (n_star + 2)) / (n1total + 1))
        C = math.sqrt((1 - (n2total + 1) / (n_star + 2)) / (n2total + 1))        
        z = A / (B + C) * z_critical_value
        
        def wilson_score_interval(n_success, n_star, z):
            center = 2 * n_success + z**2
            margin = z * math.sqrt(z**2 + 4 * n_success * (1 - n_success / n_star))
            denominator = 2 * (n_star + z**2)
            lower = (center - margin) / denominator
            upper = (center + margin) / denominator
            return lower, upper
        
        l1, u1 = wilson_score_interval(n1total, n_star, z)
        l2, u2 = wilson_score_interval(n2total, n_star, z)
        
        lower_bound_Bonett_Price = l1 / u2
        upper_bound_Bonett_Price = u1 / l2
        
        # Wilson Score Method for the Bonett & Price CI's with Continuity Correction
        def wilson_score_interval_lower(n_success, n_star, z):
            numerator = 2 * n_success + z**2 - 1 - z * math.sqrt(z**2 - 2 - (1/n_star) + 4*(n_success/n_star)  * (n_star- n_success + 1))
            denominator = 2 * (n_star + z**2)
            lower = numerator / denominator
            return lower

        def wilson_score_interval_upper(n_success, n_star, z):
            numerator = 2 * n_success + z**2 + 1 + z * math.sqrt(z**2 + 2 - (1/n_star) + 4*(n_success/n_star) * (n_star- n_success - 1))
            denominator = 2 * (n_star + z**2)
            upper = numerator / denominator
            return  upper

        l1_cc = wilson_score_interval_lower(n1total, n_star, z)
        u1_cc = wilson_score_interval_upper(n1total, n_star, z)
        l2_cc = wilson_score_interval_lower(n2total, n_star, z)
        u2_cc = wilson_score_interval_upper(n2total, n_star, z)

        lower_bound_Bonett_Price_CC = l1_cc / u2_cc
        upper_bound_Bonett_Price_CC = u1_cc / l2_cc

        # 4. MOVER Wilson based CI's (based n Tang et al., 2010)
        def wilson_score_MOVER(n_success, sample_size, z):
            center = 2 * n_success + z**2
            margin = z * math.sqrt(z**2 + 4 * n_success * (1 - n_success / sample_size))
            denominator = 2 * (sample_size + z**2)
            lower = (center - margin) / denominator
            upper = (center + margin) / denominator
            return lower, upper

        l1_MOVER, u1_MOVER = wilson_score_MOVER(n1total, sample_size, z_critical_value)
        l2_MOVER, u2_MOVER = wilson_score_MOVER(n2total, sample_size, z_critical_value)


        Correlation = (yes_yes * no_no - yes_no * no_yes) / math.sqrt((n1total) * (n2total) * (no_no + no_yes) * (no_no + yes_no))
        A = (proportion_sample_1 - l1_MOVER) * (u2_MOVER - proportion_sample_2) * Correlation
        B = (u1_MOVER - proportion_sample_1) * (proportion_sample_2 - l2_MOVER) * Correlation
        discriminant_lower = (A - proportion_sample_1 * proportion_sample_2) ** 2 - l1_MOVER * (2 * proportion_sample_1 - l1_MOVER) * u2_MOVER * (2 * proportion_sample_2 - u2_MOVER)
        discriminant_upper = (B - proportion_sample_1 * proportion_sample_2) ** 2 - u1_MOVER * (2 * proportion_sample_1 - u1_MOVER) * l2_MOVER * (2 * proportion_sample_2 - l2_MOVER)

        if discriminant_lower < 0:
            raise ValueError("Negative discriminant encountered. Check inputs or numerical approximations.")

        # Final calculation of lower bound
        lower_MOVER = (A - proportion_sample_1 * proportion_sample_2 + math.sqrt(discriminant_lower)) / (u2_MOVER * (u2_MOVER - 2 * proportion_sample_2))
        upper_MOVER = (B - proportion_sample_1 * proportion_sample_2 - math.sqrt(discriminant_upper)) / (l2_MOVER * (l2_MOVER - 2 * proportion_sample_2))


        # 5. Tang Method #Function based on ratesci
        def Tang_T1_function(x, Delta): 
            N = np.sum(x)
            Stheta = ((x[1] + x[0]) - (x[2] + x[0]) * Delta)
            A = N * (1 + Delta)
            B = (x[0] + x[2]) * Delta**2 - (x[0] + x[1] + 2 * x[2])
            C_ = x[2] * (1 - Delta) * (x[0] + x[1] + x[2]) / N
            num = -B + np.sqrt(B**2 - 4 * A * C_)
            q21 = num / (2 * A)
            Variance = np.maximum(0, N * (1 + Delta) * q21 + (x[0] + x[1] + x[2]) * (Delta - 1))
            Z_Score_Tang = Stheta / np.sqrt(Variance)
            return Z_Score_Tang, np.sqrt(Variance)
        
        def Tang_CI_Function(Function, Maximum_Iterations=100, CI = "lower_CI"):
            hi = 1; lo = -1; niter = 1
            while niter <= Maximum_Iterations:
                mid = max(-1, min(1, (hi + lo) / 2))
                scor = Function(np.tan(np.pi * (mid + 1) / 4))
                check = (scor <= 0) or np.isnan(scor)
                hi = mid if check else hi
                lo = mid if not check else lo
                niter += 1
            Optimize = lo if CI == "lower_CI" else hi
            return np.tan((Optimize + 1) * np.pi / 4)

        def myfun(Delta):
            return Tang_T1_function(Delta=Delta, x=x)[0]
        Statistic_Tang, Standard_Deviation_Tang = Tang_T1_function(Delta=Delta, x=x)
        Standard_Error_Tang = Standard_Deviation_Tang / np.sqrt((sample_size-no_no)**2)
        P_Val_Tang = norm.cdf(abs(Statistic_Tang))
        CI_Tang_lower = Tang_CI_Function(lambda Delta: myfun(Delta) - z_critical_value, CI="lower_CI")
        CI_Tang_Upper = Tang_CI_Function(lambda Delta: myfun(Delta) + z_critical_value, CI="upper_CI")


        # Other Relative Measures
        Population_Attributional_Risk = ((p1/(p1+p2)) * ((Matched_Pairs_Relative_Risk-1) /Matched_Pairs_Relative_Risk)) 
        Incidental_Rate_Exposed = proportion_sample_1
        Incidental_Rate_UnExposed = proportion_sample_2
        Incidental_Rate_Population = (p1+p2)/sample_size
        Risk_Differnce = proportion_sample_1 - proportion_sample_2
        Risk_Differnce_percentages = Risk_Differnce * 100
        Exposed_Attributable_Fraction =  Risk_Differnce / Incidental_Rate_Exposed
        Exposed_Attributable_Fraction_percentages = Exposed_Attributable_Fraction * 100
        Population_attributable_risk_percentages = Population_Attributional_Risk 
        Population_Attributable_Fraction = (Population_Attributional_Risk / Incidental_Rate_Population) / 100
        Population_Attributable_Fraction_percentages = Population_Attributable_Fraction 
        NNT = 1 / (proportion_sample_1-proportion_sample_2)


        results = {}
        
        # Descriptive Statistics
        results["Table 1 - Descriptive Statistics"] = ""
        results["--------------------------------"] = ""
        results["Proportion of Success of Variable 1"] = round(proportion_sample_1, 4)
        results["Proportion of Success of Variable 2"] = round(proportion_sample_2, 4)
        results["Joint Proportion of Success"] = round(Proportion_Matching, 4)
        results["Joint Proportion of Failure"] = round(Proportion_Not_Matching, 4)
        results["Yes Yes Frequency"] = round(yes_yes)
        results["Yes No Frequency"] = round(yes_no)
        results["No Yes Frequency"] = round(no_yes, 4)
        results["No No Frequency"] = round(no_no, 4)
        results["Difference Between Proportions"] = round(difference_between_proportions, 7)
        results["                                                                                                                                                       "] = ""

        # Inferential Statistics
        results["Table 2 - Inferntial Statistics"] = ""
        results["------------------------------"] = ""
        results["McNemar Chi Square Statistic"] = f"({round(mcnemar_chi_square, 4)})"
        results["McNemar Chi Square Statistic Corrected)"] = f"({round(mcnemar_chi_square_corrected, 4)})"
        results["McNemar pvalue"] = f"({(mcnemar_pvalue)})"
        results["McNemar pvalue corrected)"] = f"({(mcnemar_pvalue_corrected)})"
        results["McNemar Exact)"] = f"({(exact_mcnemar)})"
        results["                                                                                                                                                         "] = ""


        # Confidence Intervals for the Difference Between Paired Proportions
        results["Table 3 - Difference Between Paired Proportions CI's"] = ""
        results["----------------------------------------------------"] = ""
        results["Confidence Intervals Wald"] = f"({round(LowerCi_WALD, 4)}, {round(UpperCi_WALD, 4)})"
        results["Confidence Intervals Wald Corrected (Edwards)"] = f"({round(LowerCi_WALD_Corrected, 4)}, {round(UpperCi_WALD_Corrected, 4)})"
        results["Confidence Intervals Wald Corrected (Yates)"] = f"({round(LowerCi_WALD_Corrected_Yates, 4)}, {round(UpperCi_WALD_Corrected_Yates, 4)})"
        results["Confidence Intervals adjusted (Agresti & Min, 2005)"] = f"({round(LowerCi_AM, 4)}, {round(UpperCi_AM, 4)})"
        results["Confidence Intervals adjusted (Bonett & Price, 2012)"] = f"({round(LowerCi_BP, 4)}, {round(UpperCi_BP, 4)})"
        results["Confidence Intervals (Newcomb)"] = f"({round(LowerCi_newcomb, 4)}, {round(UpperCi_newcomb, 4)})"

        results["Standard Error Wald"] = f"({round(Standard_Error_Wald, 4)})"
        results["Standard Error Wald CC Correction"] = f"({round(Standard_Error_Wald_Corrected, 4)})"
        results["Standard Error Wald Yates Correction"] = f"({round( Standard_Error_Wald_Corrected_yates, 4)})"
        results["Standard Error Agresti-Min"] = f"({round(Standard_Error_AM, 4)})"
        results["Standard Error Bonett-Price"] = f"({round(Standard_Error_BP, 4)})"
        results["                                                                                                                                                                      "] = ""


        # Matched Pairs Odds Ratio
        results["Table 4 - Odds Ratios and their CI's"] = ""
        results["------------------------------------"] = ""        
        results["Matched Pairs Odds Ratio)"] = f"({round(Match_Pairs_Odds_Ratio, 4)})"
        results["Matched Pairs Odds Ratio (Adjusted - Jewell)"] = f"({round(Match_Pairs_Odds_Ratio_Adjusted, 4)})"
        results["Matched Pairs Odds Ratio (Jackknife Correction)"] = f"({round(Odds_Ratio_Matched_Pairs_Jackknife, 4)})"
        results["Matched Pairs Odds Ratio (Bias Corrected)"] = f"({round(Odds_Ratio_Bias_Corrected, 4)})"
        results["Standard Error Matched Pairs Odds Ratio"] = f"({round(Standrd_Error_Odds_ratio, 4)})"
        results["Adjusted Standard Error Matched Pairs Odds Ratio"] = f"({round(Adjusted_Standrd_Error_Odds_ratio, 4)})"

        results["Matched Pairs Odds Ratio CI's (Wald)"] = f"({round(CI_odds_ratio_log_lower, 4)}, {round(CI_odds_ratio_log_upper, 4)})"
        results["Matched Pairs Odds Ratio CI's (Binomial Method)"] = f"({round(CI_odds_ratio_binomial_lower, 4)}, {round(CI_odds_ratio_binomial_upper, 4)})"
        results["Matched Pairs Odds Ratio CI's (Wald with Laplace Correction)"] = f"({round(Lower_Wald_Laplace, 4)}, {round(Upper_Wald_Laplace, 4)})"        
        results["Matched Pairs Odds Ratio CI's (Rigby & Robinsone)"] = f"({round(CI_Mcnemar_odds_ratio_lower, 4)}, {round(CI_Mcnemar_odds_ratio_upper, 4)})"        
        results["Matched Pairs Odds Ratio Corrected CI's (Rigby & Robinsone)"] = f"({round(CI_Mcnemar_corrected_odds_ratio_lower, 4)}, {round(CI_Mcnemar_corrected_odds_ratio_upper, 4)})"        
        results["Matched Pairs Odds Ratio CI's Transformed Wilson Score Method"] = f"({round(lower_ci_score_method, 4)}, {round(upper_ci_score_method, 4)})"
        results["Matched Pairs Odds Ratio CI's Fiducial"] = f"({round(Fiducial_lower, 4)}, {round(Fiducial_upper, 4)})"
        results["                                                                                                                                                                         "] = ""



        # Matched Pairs Relative Risk/Risk Ratio
        results["Table 5 - Matched Pairs Relative Risk and their CI's"] = ""
        results["---------------------------------------------------"] = "" 
        results["Matched Pairs Relative Risk"] = f"({round(Matched_Pairs_Relative_Risk, 4)})"
        results["Matched Pairs Relative Risk CI's Wald"] = f"({round(lower_wald_RR, 4)}, {round(upper_wald_RR, 4)})"        
        results["Matched Pairs Relative Risk CI's (Tang Method)"] = f"({round(CI_Tang_lower, 4)}, {round(CI_Tang_Upper, 4)})"
        results["Matched Pairs Relative Risk CI's (Bonett-Price)"] = f"({round(lower_bound_Bonett_Price, 4)}, {round(upper_bound_Bonett_Price, 4)})"
        results["Matched Pairs Relative Risk CI's (Bonett-Price with CC)"] = f"({round(lower_bound_Bonett_Price_CC, 4)}, {round(upper_bound_Bonett_Price_CC, 4)})"
        results["Matched Pairs Relative Risk CI's (MOVER Wilson)"] = f"({round(lower_MOVER, 4)}, {round(upper_MOVER, 4)})"

  
  
        #More RiskMeasurments
        results["Table 6 - More Risk Measurments"] = ""
        results["-------------------------------"] = "" 
        results["Population Attributional Risk"] = round(Population_Attributional_Risk, 4)
        results["Number Needed to Treat (NNT)"] = round(NNT, 4)
        results["Incidental Rate Exposed"] = round(Incidental_Rate_Exposed, 4)
        results["Incidental Rate Unexposed"] = round(Incidental_Rate_UnExposed, 4)
        results["Incidental Rate Population"] = round(Incidental_Rate_Population, 4)
        results["Risk Difference (Absolute Risk Reduction)"] = round((proportion_sample_1 - proportion_sample_2), 4)
        results["Risk Difference (%)"] = round(abs(Risk_Differnce_percentages), 4)
        results["Exposed Attributable Fraction"] = round((Exposed_Attributable_Fraction), 4)
        results["Exposed_Attributable_Fraction (%)"] = round((abs(Exposed_Attributable_Fraction_percentages)), 4)
        results["Population Atributable Risk"] = round((Population_Attributional_Risk), 4)
        results["Population Atributable Risk (%) "] = round((abs(Population_attributable_risk_percentages*100)), 4)
        results["Population Atributable Fraction (Relative Risk Reduction) "] = round((Population_Attributable_Fraction), 4)
        results["Population Atributable Fraction (%) "] = round((abs(Population_Attributable_Fraction_percentages*100)), 4)

        return results



class two_dep_samples_proportions():
    @staticmethod
    def Two_Dep_Proportions_From_Parameters (params: dict) -> dict:

        # Set params
        confidence_level_percentages = params["Confidence Level"]
        proportion_sample_1 = params["Proportion Sample 1"]
        proportion_sample_2 = params["Proportion Sample 2"]
        Joint_Proportion = params["Joint Proportion"]

        sample_size = params["Sample Size"]
        difference_in_population = params["Difference in the Population"]
        Two_dependent_proportions_output = Main_Two_Dep_Proportions_From_Parameters(proportion_sample_1, proportion_sample_2, Joint_Proportion, sample_size, confidence_level_percentages, difference_in_population)

        results = {"Two Dependent Proportions From Parameters": Two_dependent_proportions_output}
    
        # Displaying results in separate rows
        for key, value in results.items():
            print(f"{key}:")
        
        for inner_key, inner_value in value.items():
            print(f"  {inner_key}: {inner_value}")

        return results



    @staticmethod
    def display_results(title: str, results: dict):
        """
        Function to display results in a readable format in the notebook output.
        """
        print(f"\n{'=' * 50}")
        print(title)
        print(f"{'=' * 50}")
        for key, value in results.items():
            print(f"{key}: {value}")

    @staticmethod
    def Two_Dependent_Proportions_From_data(params: dict) -> dict:
        # Set params
        confidence_level_percentages = params["Confidence Level"]
        column_1 = params["Column 1"]
        column_2 = params["Column 2"]
        population_proportion = params["Populations Difference"]
        defined_success = params["Defined Success Value"]

        df = pd.DataFrame({"Column_1": column_1, "Column_2": column_2})

        # Calculate sample size by counting non-empty rows in both columns
        sample_size = df.dropna(subset=["Column_1", "Column_2"]).shape[0]

        number_of_successes_1 = np.count_nonzero(column_1 == defined_success)
        proportion_sample_1 = number_of_successes_1 / sample_size
        number_of_successes_2 = np.count_nonzero(column_2 == defined_success)
        proportion_sample_2 = number_of_successes_2 / sample_size
        yes_yes = df[(df["Column_1"] == defined_success) & (df["Column_2"] == defined_success)].shape[0]
        proportion_of_yes_yes = yes_yes / sample_size

        Two_dependent_proportions_output = Main_Two_Dep_Proportions_From_Parameters(
            proportion_sample_1, proportion_sample_2, proportion_of_yes_yes,
            sample_size, confidence_level_percentages, population_proportion
        )

        # Display results
        two_dep_samples_proportions.display_results(
            "Two Dependent Proportions From Data", Two_dependent_proportions_output
        )

        return {"Two Dependent Proportions From Data": Two_dependent_proportions_output}

    @staticmethod
    def Two_dependent_Proportions_From_Frequencies(params: dict) -> dict:
        # Extract parameters
        yes_yes = params["Sucesses in two samples"]
        yes_no = params["Success only in first sample"]
        no_yes = params["Success only in second sample"]
        no_no = params["No Sucess"]
        confidence_level = params["Confidence Level"]
        population_diff = params["Difference in the Population"]

        # Calculate proportions
        sample_size = yes_yes + yes_no + no_yes + no_no
        proportion_sample_1 = (yes_no + yes_yes) / sample_size
        proportion_sample_2 = (no_yes + yes_yes) / sample_size
        proportion_of_joint = yes_yes / sample_size

        # Call the main computation function
        output = Main_Two_Dep_Proportions_From_Parameters(proportion_sample_1, proportion_sample_2, proportion_of_joint,sample_size, confidence_level, population_diff)

        # Display results
        two_dep_samples_proportions.display_results(
            "Two Dependent Proportions From Frequencies", output
        )

        return {"Two Dependent Proportions From Frequencies": output}

    
    # Consider
    #1. Adding a More rare adjusments for the matched pairs odds ratio (Jeweel, 1984) - fits small samples
    #2. Consider adding the log odd ratios 
    #3. Adding +0.5 to deal with zeros
    # Adding Person Years in the future to calcuate rate ratio
        
        
    #Check Panello 2010 for multiple Proportions
    
    
    ##### Functions for Rate Ratio
    
    """
    Standard_Error_Delta = np.sqrt((yes_yes+yes_no)*(yes_no+no_yes)/(yes_yes+no_yes)**3)
    Statistic_Delta = (Matched_Pairs_Relative_Risk - Delta) / Standard_Error_Delta
    P_Val_Delta = norm.cdf(abs(Statistic_Delta))
    CI_delta_lower = Matched_Pairs_Relative_Risk - Standard_Error_Delta * z_critical_value
    CI_delta_Upper = Matched_Pairs_Relative_Risk + Standard_Error_Delta * z_critical_value

    # 3. Delta Method Modified
    Standard_Error_Delta_modified = np.sqrt(Delta * (yes_no+no_yes)/(yes_yes+no_yes)**2)
    Statistic_Delta_modified = (Matched_Pairs_Relative_Risk - Delta) / Standard_Error_Delta_modified
    P_Val_Delta_modified = norm.cdf(abs(Statistic_Delta_modified))
    CI_delta_lower_modified = Matched_Pairs_Relative_Risk - Standard_Error_Delta_modified * z_critical_value
    CI_delta_Upper_modified = Matched_Pairs_Relative_Risk + Standard_Error_Delta_modified * z_critical_value

    # 4. Katz Logarithmic
    Standard_Error_Katz = np.sqrt((no_yes+yes_no)/((yes_yes+yes_no)*(yes_yes+no_yes)))
    Statistic_Katz = (math.log(yes_no+yes_yes) - math.log(yes_yes+no_yes) - math.log(Delta)) / Standard_Error_Katz
    P_Val_Katz = norm.cdf(abs(Statistic_Katz))
    CI_katz_lower = Matched_Pairs_Relative_Risk - Standard_Error_Katz * z_critical_value
    CI_katz_Upper = Matched_Pairs_Relative_Risk + Standard_Error_Katz * z_critical_value

    # 5. Katz logarithmic Modified
    Standard_Error_Katz_Modified = np.sqrt((yes_no + no_yes) / (Delta*(yes_yes+no_yes)**2))
    Statistic_Katz_Modified = (math.log(yes_no+yes_yes) - math.log(yes_yes+no_yes) - math.log(Delta))  / Standard_Error_Katz_Modified
    P_Val_Katz_Modified = norm.cdf(abs(Statistic_Katz_Modified))
    CI_katz_modified_lower = Matched_Pairs_Relative_Risk - Standard_Error_Katz_Modified * z_critical_value
    CI_katz_modified_Upper = Matched_Pairs_Relative_Risk + Standard_Error_Katz_Modified * z_critical_value

    # 6. Lachenbruch and Lynch
    Standard_Deviation_Lachenbruch_Lynch = np.sqrt((yes_no*(sample_size-yes_no) + yes_yes*(sample_size-yes_yes)*(1-Delta)**2 + no_yes*(sample_size-no_yes)*Delta**2 + 2*yes_no*no_yes*Delta-2*yes_yes*yes_no*(1-Delta) + 2*yes_yes*no_yes*Delta*(1-Delta))/ sample_size)
    Standard_Error_Lachenbruch_Lynch = Standard_Deviation_Lachenbruch_Lynch / np.sqrt((sample_size-no_no)**2)
    Statistic_Lachenbruch_Lynch = ((yes_yes+yes_no) - Delta*(yes_yes+no_yes)) / Standard_Deviation_Lachenbruch_Lynch
    P_Val_Lachenbruch_Lynce = norm.cdf(abs(Statistic_Lachenbruch_Lynch))
    CI_Lachenbruch_Lynce_lower = Matched_Pairs_Relative_Risk - Standard_Error_Lachenbruch_Lynch * z_critical_value
    CI_Lachenbruch_Lynce_Upper = Matched_Pairs_Relative_Risk + Standard_Error_Lachenbruch_Lynch * z_critical_value
    """
    
    