###############################################
# Effect Size for Two Independent Proportions##
###############################################

import numpy as np
import math
from scipy.stats import norm, beta, binom, chi2_contingency, barnard_exact, fisher_exact
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep
from scipy.optimize import newton, root_scalar
from scipy.stats.contingency import odds_ratio
import rpy2.robjects as robjects


def main_Two_sample_proportions (proportion_sample_1, proportion_sample_2, sample_size_1, sample_size_2, confidence_level_percentages, difference_in_population): 
        # Calculation
        #############
        confidence_level = confidence_level_percentages / 100

        p1 = proportion_sample_1 * sample_size_1
        p2 = proportion_sample_2 * sample_size_2
        q1 = sample_size_1 - p1
        q2 = sample_size_2 - p2
        sample_size = sample_size_1 + sample_size_2 
        samples_difference = proportion_sample_1-proportion_sample_2


        # Inferntial_Statistics
        #######################
        continuiety_correction = (1/sample_size_1 + 1/sample_size_2)/2

        #1. Wald Type Z-Statistic
        Standard_Error_Wald_sample_1 = (proportion_sample_1*(1-proportion_sample_1)) / sample_size_1
        Standard_Error_Wald_sample_2 = (proportion_sample_2*(1-proportion_sample_2)) / sample_size_2
        Standard_Error_Wald = np.sqrt(Standard_Error_Wald_sample_1 + Standard_Error_Wald_sample_2)
        Z_wald = (samples_difference - difference_in_population) / Standard_Error_Wald
        p_value_wald = norm.sf(abs(Z_wald))

        Z_Wald_corrected = (samples_difference - difference_in_population - continuiety_correction) / Standard_Error_Wald
        p_value_Wald_corrected = norm.sf(abs(Z_Wald_corrected))

        #2. Wald_H0 Type Z-Statistic
        Estimitated_Population_Proportion =  (proportion_sample_1*sample_size_1 + proportion_sample_2*sample_size_2) / (sample_size_1 + sample_size_2)
        Standard_error_Wald_H0 = np.sqrt(Estimitated_Population_Proportion * (1-Estimitated_Population_Proportion) * (1/sample_size_1 + 1/sample_size_2))
        Z_Wald_H0 = (samples_difference - difference_in_population) / Standard_error_Wald_H0
        p_value_Wald_H0 = norm.sf(abs(Z_Wald_H0))

        Z_Wald_H0_corrected = (samples_difference - difference_in_population - continuiety_correction) / Standard_error_Wald_H0
        p_value_Wald_H0_corrected = norm.sf(abs(Z_Wald_H0_corrected))

        # 3. Hauck and Anderson's Z Statistic
        correction_HA = 1/(2 * min(sample_size_1, sample_size_2))# Hauck and Anderson's correction
        Standard_Error_sample_1_HA = (proportion_sample_1*(1-proportion_sample_1)) / (sample_size_1 - 1)
        Standard_Error_sample_2_HA = (proportion_sample_2*(1-proportion_sample_2)) / (sample_size_2 - 1)
        Standard_error_HA = np.sqrt(Standard_Error_sample_1_HA + Standard_Error_sample_2_HA)
        z_score_HA = (samples_difference -  difference_in_population - correction_HA) / Standard_error_HA
        p_value_HA = norm.sf(abs(z_score_HA))

        # 4. Conditional Mantel Haenszel Test
        Standart_Error_MH = np.sqrt((sample_size_1*sample_size_2*(p1+p2)* (q1+q2)) / (sample_size**2 *(sample_size-1)))
        Mean_MH = (sample_size_1*(p1+p2)) / sample_size
        Z_MH = (p1 - Mean_MH) / Standart_Error_MH
        pval_MH = norm.sf(abs(Z_MH))

        # 5. Barnard Exact
        data_2x2 = [[p1, p2] , [q1,q2]]
        Barnard_Exact = barnard_exact(data_2x2)
        pvalue_barnard = Barnard_Exact.pvalue
        Barnard_D_Statistic = Barnard_Exact.statistic

        # 6. Fisher Exact
        data_2x2 = [[p1, p2] , [q1,q2]]
        Fisher_Exact_p_value = fisher_exact(data_2x2).pvalue

        
        # Effect Sizes
        ###################

        # 1. Cohen's h
        phi1 = 2*np.arcsin(np.sqrt(proportion_sample_1))
        phi2 = 2*np.arcsin(np.sqrt(proportion_sample_2))
        cohens_h =  abs(phi1 - phi2)

        
        # 2. Calculate Probit d and its Variance (see Wilson, 2017)
        Z_score_Proportion_1 =  norm.ppf(proportion_sample_1)
        Z_score_Proportion_2 =  norm.ppf(proportion_sample_2)
        Probit_d = Z_score_Proportion_1 - Z_score_Proportion_2
        #Add variance calculation for CI's


        # 3. Calculate Logit d
        logit_d = np.log((proportion_sample_1/(1-proportion_sample_1))) - np.log((proportion_sample_2/(1-proportion_sample_2)))
        # Add the variance for this


        # 4. Cohens W
        table = np.array([[p1,p2], [q1,q2]])
        chi_square = chi2_contingency(table).statistic
        Cohens_w = np.sqrt(chi_square/sample_size)

        # 5. Relative Risk / Risk Ratio
        Realtive_risk = proportion_sample_1/proportion_sample_2
        Realtive_risk_unbiased = (p1*(q1+q2+1)) / (((p1+p2) * (q1 + 1))) # This is the unbiased estimator of the relative risk (Jewell, 1986)
        Realtive_risk_adjusted_Walters = np.exp(np.log((p1+0.5) / (sample_size_1+0.5)) - np.log((p2+0.5) / (sample_size_2+0.5)) )

        # 6. Odds Ratio
        # A. Conditional MLE (Corenfield)
        odds_ratio_Conditional_test = odds_ratio([[int(p1), int(q1)], [int(p2), int(q2)]])
        odds_ratio_Conditional = odds_ratio_Conditional_test.statistic
        
        # B. Unconditional MLE (Wald)
        odds_ratio_wald = (proportion_sample_1*(1-proportion_sample_2)) / (proportion_sample_2* (1-proportion_sample_1))

        # C. Wald_Adjusted
        odds_ratio_wald_adjusted = ((p1 + 0.5) * (q2 + 0.5)) / ((p2 + 0.5) * (q1 + 0.5))

        # D. Wald Small Samples Correction
        odds_ratio_small_samples_adjusted = (p1 * q2) / ((p2 + 1) * (q1 + 1))

        # E. Median unbaised odds ratio (Mid-p method)
        robjects.r('''odds_ratio_median_unbaised = function(p1, q1, p2, q2, conf.level = 0.95, interval = c(0, 1000)) {
            alpha = 1 - conf.level
            x = matrix(c(p1, q1, p2, q2), 2, 2)
            oddsratio = uniroot(function(or_val) {fisher.test(x, or = or_val, alternative = "less")$p.value - fisher.test(x, or = or_val, alternative = "greater")$p.value},interval = interval)$root
            return(oddsratio)}''')
        
        odds_ratio_median_unbaised = robjects.r['odds_ratio_median_unbaised'](p1, p2, q1, q2)  # type: ignore
        
        
        # F chi_square based Odds Ratio (Leave it for now, Needs to verify this Method)
        chi2_odds_ratio =   (abs(p1 - ((p1+q1)*(p1+p2)/sample_size)))**2 / ((p1+q1)*(p1+p2)/sample_size) + \
                            (abs(q1 - ((p1+q1)*(q1+q2)/sample_size)))**2 / ((p1+q1)*(q1+q2)/sample_size) + \
                            (abs(p2 - ((p2+q2)*(p1+p2)/sample_size)))**2 / ((p2+q2)*(p1+p2)/sample_size) + \
                            (abs(q2 - ((p2+q2)*(q1+q2)/sample_size)))**2 / ((p2+q2)*(q1+q2)/sample_size) 

        chi2_odds_ratio_adjusted = (abs(p1 - ((p1+q1)*(p1+p2)/sample_size)) - 0.5)**2 / ((p1+q1)*(p1+p2)/sample_size) + \
                                (abs(q1 - ((p1+q1)*(q1+q2)/sample_size)) - 0.5)**2 / ((p1+q1)*(q1+q2)/sample_size) + \
                                (abs(p2 - ((p2+q2)*(p1+p2)/sample_size)) - 0.5)**2 / ((p2+q2)*(p1+p2)/sample_size) + \
                                (abs(q2 - ((p2+q2)*(q1+q2)/sample_size)) - 0.5)**2 / ((p2+q2)*(q1+q2)/sample_size)

    
    

        # 7. Yules Q Family
        Yules_Q = (p1*q2 - q1*p2) / (p1*q2 + q1*p2)
        Yules_Y = ((p1*q2)**0.5 - (q1*p2)**0.5) / ((p1*q2)**0.5 + (q1*p2)**0.5)
        Digbys_H = ((p1*q2)**0.75 - (q1*p2)**0.75) / ((p1*q2)**0.75 + (q1*p2)**0.75)

        # Yules Y* (Bonett and Price, 2007) 
        Marginal_Proportion_1 = (p1+p2)/sample_size
        Marginal_Proportion_2 = (p1+q1)/sample_size
        Marginal_Proportion_3 = (p2+q2)/sample_size
        Marginal_Proportion_4 = (q1+q2)/sample_size

        marginal_proportions = np.array([Marginal_Proportion_1, Marginal_Proportion_2, Marginal_Proportion_3, Marginal_Proportion_4])
        min_marginal_prooportion = min(marginal_proportions)
        c = 0.5 - (0.5 - min_marginal_prooportion) **2
        corrected_odds_ratio = ((p1+0.1) * (q2+0.1)) / ((p2+0.1) * (q1+0.1))
        Yules_Y_Star = (corrected_odds_ratio**c - 1) / (corrected_odds_ratio**c + 1)


        # 8. Number Needed to Treat   **Confidence Intervals here will be CI's for 1 / proportion difference  
        NNT = 1 / (proportion_sample_1-proportion_sample_2)


        # 9. Other Risk Measures 
        Population_Attributional_Risk = ((p1/(p1+p2)) * ((Realtive_risk-1) /Realtive_risk)) 
        Population_attributable_risk_percentages = Population_Attributional_Risk * 100
        Incidental_Rate_Exposed = proportion_sample_1
        Incidental_Rate_UnExposed = proportion_sample_2
        Incidental_Rate_Population = (p1+p2)/sample_size
        Exposed_Attributable_Fraction =  samples_difference / Incidental_Rate_Exposed
        Exposed_Attributable_Fraction_percentages = Exposed_Attributable_Fraction * 100
        Population_Attributable_Fraction = (Population_Attributional_Risk / Incidental_Rate_Population) / 100
        Population_Attributable_Fraction_percentages = Population_Attributable_Fraction 


        #Confidence Intervals:
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)


        # 1. Confidence Intervals for the Difference Between Independent Proportions
        #########################################################################
        Standard_Error_Wald_sample_1 = (proportion_sample_1*(1-proportion_sample_1)) / sample_size_1
        Standard_Error_Wald_sample_2 = (proportion_sample_2*(1-proportion_sample_2)) / sample_size_2
        Standard_Error_Wald = np.sqrt(Standard_Error_Wald_sample_1 + Standard_Error_Wald_sample_2)

        # Method 1 - Wald Method
        lower_ci_wald = samples_difference - zcrit*(Standard_Error_Wald)
        upper_ci_wald = samples_difference + zcrit*(Standard_Error_Wald)

        # Method 2 - Wald corrected Method
        lower_ci_wald_corrected = samples_difference - (0.5*(1/sample_size_1 + 1/sample_size_2) + zcrit*(Standard_Error_Wald))
        upper_ci_wald_corrected = samples_difference + (0.5*(1/sample_size_1 + 1/sample_size_2) + zcrit*(Standard_Error_Wald))

        # Method 3 - Haldane
        psi_Haldane = (proportion_sample_1 + proportion_sample_2) / 2
        v = (1/sample_size_1 - 1/sample_size_2) / 4
        mu = (1/sample_size_1 + 1/sample_size_2) / 4
        theta_haldane = ((proportion_sample_1 - proportion_sample_2) + zcrit**2*v*(1-2*psi_Haldane)) / (1 + zcrit**2*mu)
        w_haldane = (zcrit/ (1+zcrit**2*mu)) * np.sqrt(mu*(4*psi_Haldane*(1-psi_Haldane) - (proportion_sample_1 - proportion_sample_2)**2) + 2*v*(1-2*psi_Haldane) * (proportion_sample_1 - proportion_sample_2) + 4*zcrit**2*mu**2*(1-psi_Haldane)*psi_Haldane  + zcrit**2*v**2*(1-2*psi_Haldane)**2  )
        lower_ci_haldane = theta_haldane - w_haldane
        upper_ci_haldane = theta_haldane + w_haldane

        # Method 4 - Jeffreys-Perks
        psi_JP = 0.5*(((p1 + 0.5) / (sample_size_1+1)) + ((p2+ 0.5) / (sample_size_2+1))) 
        theta_JP = ((proportion_sample_1 - proportion_sample_2) + zcrit**2*v*(1-2*psi_JP)) / (1 + zcrit**2*mu)
        w_JP = (zcrit/ (1+zcrit**2*mu)) * np.sqrt(mu*(4*psi_JP*(1-psi_JP) - (proportion_sample_1 - proportion_sample_2)**2) + 2*v*(1-2*psi_JP) * (proportion_sample_1 - proportion_sample_2) + 4*zcrit**2*mu**2*(1-psi_JP)*psi_JP  + zcrit**2*v**2*(1-2*psi_JP)**2  )
        lower_ci_JP= theta_JP - w_JP
        upper_ci_JP = theta_JP + w_JP

        # Method 5+6 - MEE and Miettinen-Nurminen
        def Standart_Error_Calcualte(proportion_sample_1, sample_size_1, proportion_sample_2, sample_size_2, difference_in_population):
            sample_size_ratio = sample_size_2 / sample_size_1
            a = 1 + sample_size_ratio
            b = -(1 + sample_size_ratio + proportion_sample_1 + sample_size_ratio * proportion_sample_2 + difference_in_population * (sample_size_ratio + 2))
            c = difference_in_population * difference_in_population + difference_in_population * (2 * proportion_sample_1 + sample_size_ratio + 1) + proportion_sample_1 + sample_size_ratio * proportion_sample_2
            d = -proportion_sample_1 * difference_in_population * (1 + difference_in_population)
            v = (b / a / 3) ** 3 - b * c / (6 * a * a) + d / a / 2
            v = np.where(np.abs(v) < np.finfo(float).eps, 0, v)
            s = np.sqrt((b / a / 3) ** 2 - c / a / 3)
            u = np.where(v > 0, 1, -1) * s
            w = (np.pi + np.arccos(v / u ** 3)) / 3
            proportion_sample_hat_1 = 2 * u * np.cos(w) - b / a / 3
            proportion_sample_hat_2 = proportion_sample_hat_1 - difference_in_population
            n = sample_size_1 + sample_size_2
            Variance_Miettinen_Nurminen = (proportion_sample_hat_1 * (1 - proportion_sample_hat_1) / sample_size_1 + proportion_sample_hat_2 * (1 - proportion_sample_hat_2) / sample_size_2) * (n / (n - 1)) 
            Standart_Error_Miettinen_Nurminen = np.sqrt(Variance_Miettinen_Nurminen)
            Variance_Miettinen_MEE = (proportion_sample_hat_1 * (1 - proportion_sample_hat_1) / sample_size_1 + proportion_sample_hat_2 * (1 - proportion_sample_hat_2) / sample_size_2)
            Standart_Error_MEE = np.sqrt(Variance_Miettinen_MEE)


            return Standart_Error_Miettinen_Nurminen, Standart_Error_MEE

        def pval(proportion_sample_1, sample_size_1, proportion_sample_2, sample_size_2, difference_in_population):
            samples_difference = proportion_sample_1 - proportion_sample_2
            se_mn, se_mee = Standart_Error_Calcualte(proportion_sample_1, sample_size_1, proportion_sample_2, sample_size_2, difference_in_population)
            z_mn = (samples_difference - difference_in_population) / se_mn
            z_mee = (samples_difference - difference_in_population) / se_mee
            p_mn = 2 * np.minimum(norm.cdf(z_mn), 1 - norm.cdf(z_mn))
            p_mee = 2 * np.minimum(norm.cdf(z_mee), 1 - norm.cdf(z_mee))

            return p_mn, p_mee

        def confidence_interval(proportion_sample_1, sample_size_1, proportion_sample_2, sample_size_2, confidence_level):
            lower_bracket = [-1, proportion_sample_1 - proportion_sample_2]
            upper_bracket = [proportion_sample_1 - proportion_sample_2, 0.999999]

            def root_func(difference_in_population):
                return pval(proportion_sample_1, sample_size_1, proportion_sample_2, sample_size_2, difference_in_population)[0] - confidence_level

            CI_mn_lower = root_scalar(root_func, bracket=lower_bracket).root
            CI_mn_upper = root_scalar(root_func, bracket=upper_bracket).root

            def root_func_mee(difference_in_population):
                return pval(proportion_sample_1, sample_size_1, proportion_sample_2, sample_size_2, difference_in_population)[1] - confidence_level

            CI_mee_lower = root_scalar(root_func_mee, bracket=lower_bracket).root
            CI_mee_upper = root_scalar(root_func_mee, bracket=upper_bracket).root


            return (max(-1, CI_mn_lower), min(1, CI_mn_upper)), (max(-1, CI_mee_lower), min(1, CI_mee_upper))

        def z_score(proportion_sample_1, sample_size_1, proportion_sample_2, sample_size_2, Population_Difference):
            samples_difference = proportion_sample_1 - proportion_sample_2
            se_mn, se_mee = Standart_Error_Calcualte(proportion_sample_1, sample_size_1, proportion_sample_2, sample_size_2, Population_Difference)
            z_score_mn = (samples_difference - Population_Difference) / se_mn
            z_score_mee = (samples_difference - Population_Difference) / se_mee
            return z_score_mn, z_score_mee

        CI_mn, CI_mee = confidence_interval(proportion_sample_1, sample_size_1, proportion_sample_2, sample_size_2, 1 - confidence_level)
        #z_score_mn, z_score_mee = z_score(proportion_sample_1, sample_size_1, proportion_sample_2, sample_size_2, difference_in_population)
        #se_mn, se_mee = Standart_Error_Calcualte(proportion_sample_1, sample_size_1, proportion_sample_2, sample_size_2, difference_in_population)
        
        # Method 7 - Agresti_Caffo
        p1_agresti_caffo = (p1 +1) / (sample_size_1 +2)
        p2_agresti_caffo = (p2 +1) / (sample_size_2 +2)
        Standard_Error_sample_1_AC = (p1_agresti_caffo*(1-p1_agresti_caffo)) / (sample_size_1 + 2)
        Standard_Error_sample_2_AC = (p2_agresti_caffo*(1-p2_agresti_caffo)) / (sample_size_2 + 2)
        Standard_error_AC = np.sqrt(Standard_Error_sample_1_AC + Standard_Error_sample_2_AC)
        lower_ci_AC= (p1_agresti_caffo-p2_agresti_caffo) - zcrit* Standard_error_AC
        upper_ci_AC = (p1_agresti_caffo-p2_agresti_caffo) + zcrit* Standard_error_AC

        #Method 8 - Wilson
        epsilon_1, constant1 = (p1 + zcrit ** 2 / 2) / ( sample_size_1 + zcrit ** 2), zcrit * math.sqrt(sample_size_1) / (sample_size_1 + zcrit ** 2) * math.sqrt(proportion_sample_1 * (1 - proportion_sample_1) + zcrit ** 2 / (4 * sample_size_1))
        CI_wilson1_lower, CI_wilson1_upper = max(0, epsilon_1 - constant1), min(1, epsilon_1 + constant1)
        epsilon_2, constant2 = (p2 + zcrit ** 2 / 2) / ( sample_size_2 + zcrit ** 2), zcrit * math.sqrt(sample_size_2) / (sample_size_2 + zcrit ** 2) * math.sqrt(proportion_sample_2 * (1 - proportion_sample_2) + zcrit ** 2 / (4 * sample_size_2))
        CI_wilson2_lower, CI_wilson2_upper = max(0, epsilon_2 - constant2), min(1, epsilon_2 + constant2)

        CI_wilson_lower = samples_difference - zcrit * np.sqrt(CI_wilson1_lower*(1-CI_wilson1_lower)/sample_size_1 + CI_wilson2_upper*(1-CI_wilson2_upper)/sample_size_2)
        CI_wilson_upper = samples_difference + zcrit * np.sqrt(CI_wilson1_upper*(1-CI_wilson1_upper)/sample_size_1 + CI_wilson2_lower*(1-CI_wilson2_lower)/sample_size_2)

        #Method 9 - Wilson Corrected
        CI_wilsonc1_lower = (2 * p1 + zcrit**2 - 1 - zcrit * math.sqrt(zcrit**2 - 2 - 1/sample_size_1 + 4 * proportion_sample_1 * (sample_size_1 * (1 - proportion_sample_1) + 1))) / (2 * (sample_size_1 + zcrit**2))
        CI_wilsonc1_upper = (2 * p1 + zcrit**2 + 1 + zcrit * math.sqrt(zcrit**2 + 2 - 1/sample_size_1 + 4 * proportion_sample_1 * (sample_size_1 * (1 - proportion_sample_1) - 1))) / (2 * (sample_size_1 + zcrit**2))
        CI_wilsonc2_lower = (2 * p2 + zcrit**2 - 1 - zcrit * math.sqrt(zcrit**2 - 2 - 1/sample_size_2 + 4 * proportion_sample_2 * (sample_size_2 * (1 - proportion_sample_2) + 1))) / (2 * (sample_size_2 + zcrit**2))
        CI_wilsonc2_upper = (2 * p2 + zcrit**2 + 1 + zcrit * math.sqrt(zcrit**2 + 2 - 1/sample_size_2 + 4 * proportion_sample_2 * (sample_size_2 * (1 - proportion_sample_2) - 1))) / (2 * (sample_size_2 + zcrit**2))

        CI_wilsonc_lower = max(-1, samples_difference - np.sqrt((proportion_sample_1 - CI_wilsonc1_lower)**2 + (CI_wilsonc2_upper-proportion_sample_2)**2) )
        CI_wilsonc_upper = min( 1, samples_difference + np.sqrt((CI_wilsonc1_upper-proportion_sample_1)**2 + (proportion_sample_2-CI_wilsonc2_lower)**2) )

        #Method 10 - Hauck-Anderson
        correction_HA = 1/(2 * min(sample_size_1, sample_size_2))# Hauck and Anderson's correction
        Standard_Error_sample_1_HA = (proportion_sample_1*(1-proportion_sample_1)) / (sample_size_1 - 1)
        Standard_Error_sample_2_HA = (proportion_sample_2*(1-proportion_sample_2)) / (sample_size_2 - 1)
        Standard_error_HA = np.sqrt(Standard_Error_sample_1_HA + Standard_Error_sample_2_HA)
    
        CI_lower_HA = max(samples_difference - 1/(2 * min(sample_size_1,sample_size_2)) - zcrit * Standard_error_HA, -1)
        CI_upper_HA = min(samples_difference + 1/(2 * min(sample_size_1,sample_size_2)) + zcrit * Standard_error_HA, 1)

        #Method 11 - Brown, Li’s Jeffreys
        p1_BLJ = (p1 + 0.5) / (sample_size_1+1)
        p2_BLJ = (p2 + 0.5) / (sample_size_2+1)
        Standard_Error_BLJ = np.sqrt(p1_BLJ*(1-p1_BLJ)/sample_size_1 + p2_BLJ*(1-p2_BLJ)/sample_size_2)
        CI_BLJ_lower = (p1_BLJ - p2_BLJ) - zcrit * Standard_Error_BLJ
        CI_BLJ_upper = (p1_BLJ - p2_BLJ) + zcrit * Standard_Error_BLJ

        #method 12 - Gart Nam
        robjects.r('''scoretheta <- function(x1, n1, x2, n2, theta, level = 0.95) {Prop_Diff <- ((x1 / n1) - (x2 / n2)) - theta
                N <- n1 + n2
                a <- (n1 + 2 * n2) * theta - N - (x1 + x2)
                b <- (a / N / 3)^3 - a * ((n2 * theta - N - 2 * x2) * theta + (x1 + x2)) / (6 * N * N) + (x2 * theta * (1 - theta)) / N / 2
                c <- ifelse(b > 0, 1, -1) * sqrt(pmax(0, (a / N / 3)^2 - ((n2 * theta - N - 2 * x2) * theta + (x1 + x2)) / N / 3))
                p2d <- pmin(1, pmax(0, round(2 * c * cos(((pi + acos(pmax(-1, pmin(1, ifelse(c == 0 & b == 0, 0, b / c^3))))) / 3)) - a / N / 3, 10)))
                p1d <- pmin(1, pmax(0, p2d + theta))
                Variance <- pmax(0, (p1d * (1 - p1d) / n1 + p2d * (1 - p2d) / n2))
                scterm <- (p1d * (1 - p1d) * (1 - 2 * p1d) / (n1^2) - p2d * (1 - p2d) * (1 - 2 * p2d) / (n2^2)) / (6 * Variance^(3 / 2))
                score <- ifelse(scterm == 0, (Prop_Diff / sqrt(Variance)), (-1 + sqrt(pmax(0, 1^2 - 4 * scterm * -(Prop_Diff / sqrt(Variance) + scterm))) ) / (2 * scterm))
                return(score)}

                Binary_Search <- function(score_function, max.iter = 100, tail = "lower") {
                nstrat <- length(eval(score_function(1)))
                hi <- rep(1, nstrat)
                lo <- rep(-1, nstrat)
                niter <- 1
                while (niter <= max.iter && any(2 > 0.0000005 | is.na(hi))) {
                mid <- pmax(-1, pmin(1, round((hi + lo) / 2, 10)))
                scor <- score_function(mid)
                check <- (scor <= 0) | is.na(scor)
                hi[check] <- mid[check]
                lo[!check] <- mid[!check]
                niter <- niter + 1}
                ci <- if (tail == "lower") lo else hi
                return(ci)}

                gart_nam = function(x1, n1, x2, n2, level = 0.95) {
                zcrit <- qnorm(1 - (1 - level)/2)
                lower <- Binary_Search(score_function = function(theta) scoretheta(x1, n1, x2, n2,theta) - zcrit, tail = "lower")
                upper <- Binary_Search(score_function = function(theta) scoretheta(x1, n1, x2, n2,theta) + zcrit, tail = "upper")
                return(c(lower, upper))
                }''')

        Gart_Nam_CI = robjects.r['gart_nam'](p1, sample_size_1, p2, sample_size_2, confidence_level)  # type: ignore
        Newcomb_CI = confint_proportions_2indep(p1, sample_size_1, p2, sample_size_2, method = "newcomb", alpha = 1 - confidence_level)


        # 2. Confidence Intervals Odds Ratio
        ####################################
        odds_ratio_wald = (proportion_sample_1*(1-proportion_sample_2)) / (proportion_sample_2* (1-proportion_sample_1))
        standard_error_odds_ratio =np.sqrt((1 / (p1)) + (1 / (p2)) + (1 / (q1)) + (1 / (q2)))
        standard_error_odds_ratio_adjusted =np.sqrt((1 / (p1+0.5)) + (1 / (p2+0.5)) + (1 / (q1 + 0.5)) + (1 / (q2 + 0.5)))

        # Method 1 - Woolf (Aka Wald) - Logarithmic asymptotyc CI's (it is also called logit CI in statsmodels)
        upper_ci_woolf = np.exp(np.log(odds_ratio_wald) + zcrit* standard_error_odds_ratio)
        lower_ci_woolf = np.exp(np.log(odds_ratio_wald) - zcrit* standard_error_odds_ratio)
        
        # Method 2 - Woolf Corrected  (Wald adjusted) - Logarithmic asymptotyc CI's with + 0.5 correction
        upper_or_woolf_adjusted = np.exp(np.log(odds_ratio_wald) + zcrit* standard_error_odds_ratio_adjusted)
        lower_or_woolf_adjusted = np.exp(np.log(odds_ratio_wald) - zcrit* standard_error_odds_ratio_adjusted)

        # Method 3 - Cornfield (Fisher exact CI's)
        odds_ratio_Conditional_test = odds_ratio([[int(p1), int(q1)], [int(p2), int(q2)]])
        odds_ratio_fisher_ci = odds_ratio_Conditional_test.confidence_interval(confidence_level)
        formatted_ci_lower = format(odds_ratio_fisher_ci[0], ".6f")
        formatted_ci_upper = format(odds_ratio_fisher_ci[1], ".6f")

        # Method 4 - Mid-p Confidence Intervals
        robjects.r('''odds_ratio_mid_p_value = function(p1, q1, p2, q2, conf.level = 0.95, interval = c(0, 1000)) {
            mid_p_function = function(or_val = 1) {
                less_p_value = fisher.test(matrix(c(p1, q1, p2, q2), 2, 2), or = or_val, alternative = "less")$p.value
                greater_p_value = fisher.test(matrix(c(p1, q1, p2, q2), 2, 2), or = or_val, alternative = "greater")$p.value
                0.5 * (less_p_value - greater_p_value + 1)
                }

            alpha = 1 - conf.level
            x = matrix(c(p1, q1, p2, q2), 2, 2)
                
            oddsratio = uniroot(function(or_val) {fisher.test(x, or = or_val, alternative = "less")$p.value - fisher.test(x, or = or_val, alternative = "greater")$p.value},interval = interval)$root
            lower_ci_small = uniroot(function(or_val) {1 - mid_p_function(or_val) - alpha/2}, interval = interval)$root
            upper_ci_small = 1/uniroot(function(or_val) {mid_p_function(1/or_val) - alpha/2}, interval = interval)$root

            return(c(oddsratio, lower_ci_small, upper_ci_small))}''')

        lower_ci_mid_p  = robjects.r['odds_ratio_mid_p_value'](p1, p2, q1, q2)[1] # type: ignore
        upper_ci_mid_p  = robjects.r['odds_ratio_mid_p_value'](p1, p2, q1, q2)[2] # type: ignore
        
        odds_ratio_mid_p = robjects.r['odds_ratio_mid_p_value'](p1, p2, q1, q2)[0] # type: ignore

        # Method 5 - the score method of Miettinen and Nurminen (1985)
        CI_MN = confint_proportions_2indep(nobs1=sample_size_1,count1 = p1,nobs2=sample_size_2,count2 = p2,method = "score", compare = "odds-ratio", alpha = 1 - confidence_level)
        

        # Method 6 - Baptista and Pike 1977
        robjects.r('''Baptista_Pike = function(p1,q1,p2,q2, conf.level = 0.95, orRange = c(10^-10, 10^10)) {
            x = matrix(c(p1,q1,p2,q2),2,2)
            alpha <- 1 - conf.level; 
            n1 <- sum(x[1, ]); n2 <- sum(x[2, ])
            sum_of_ps <- sum(x[, 1]); x <- x[1, 1]
            support <- max(0, sum_of_ps - n2):min(n1, sum_of_ps)
            dnhyper <- function(OR) {
                d <- dhyper(support, n1, n2, sum_of_ps, log = TRUE) + log(OR) * support
                exp(d - max(d)) / sum(exp(d - max(d)))}
            pnhyper <- function(x, OR, lower.tail = TRUE) {
                f <- dnhyper(OR)
                X <- if (lower.tail) support <= x else support >= x
                sum(f[X])}
            intercept <- function(xlo, xhi, ORRange = orRange) {
                X <- support <= xlo; Y <- support >= xhi
                uniroot(function(beta) sum(dnhyper(beta)[X]) - sum(dnhyper(beta)[Y]), ORRange)$root}
            ints_greater <- intercept(x, x + 1)
            ints_less <- intercept(x - 1, x)
            CINT_upper <-uniroot(function(or) alpha - pnhyper(x, or, lower.tail = TRUE), c(ints_greater, orRange[2]))$root
            CINT_lower <-uniroot(function(or) alpha - pnhyper(x, or, lower.tail = FALSE), c(orRange[1], ints_less))$root
            c(CINT_lower, CINT_upper)}''')

        lower_bp  = robjects.r['Baptista_Pike'](p1,q1,p2,q2)[0] # type: ignore
        upper_ci_bp  = robjects.r['Baptista_Pike'](p1,q1,p2,q2)[1] # type: ignore

        # Method 7 - inverse hyperbolic sine method (Newcomb, 2001)
        standart_error_sinh = 2 * math.asinh(zcrit / 2 * math.sqrt(1/p1 + 1/(q1) + 1/p2 + 1/(q2)))
        lower_limit_sinh = math.exp(math.log(odds_ratio_wald) - standart_error_sinh)
        upper_limit_sinh = math.exp(math.log(odds_ratio_wald) + standart_error_sinh) 

        # Method 8 - Independent Smooth logit (Agresti, 1999) (It is also called logit adjusted CI in statsmodels)
        p1new = p1 + 2 * sample_size_1 * (p1 + p2) / (sample_size_1 + sample_size_2)**2
        q1new = sample_size_1 - p1 + 2 * sample_size_1 * (sample_size_1 - p1 + sample_size_2 - p2) / (sample_size_1 + sample_size_2)**2
        p2new = p2 + 2 * sample_size_2 * (p1 + p2) / (sample_size_1 + sample_size_2)**2
        q2new = sample_size_2 - p2 + 2 * sample_size_2 * (sample_size_1 - p1 + sample_size_2 - p2) / (sample_size_1 + sample_size_2)**2
        log_theta = np.log(p1new * q2new / (p2new * q1new))
        ci_half_len = norm.ppf(1 - (1 - confidence_level) / 2) * np.sqrt(1 / p1new + 1 / q1new + 1 / p2new + 1 / q2new)
        ci_lower_agresti_ind = np.exp(log_theta - ci_half_len)
        ci_upper_agresti_ind = np.exp(log_theta + ci_half_len)

        # Method 9 - Farrington-Manning
        robjects.r('''score_test_statistic.Uncorrected <- function(theta0, n11, n21, n1p, n2p) {
            p2hat <- (-(n1p * theta0 + n2p - (n11 + n21) * (theta0 - 1)) + sqrt((n1p * theta0 +
            n2p - (n11 + n21) * (theta0 - 1))^2 - 4 * n2p * (theta0 - 1) * -(n11 + n21))) / (2 * n2p * (theta0 - 1))
            p1hat <- p2hat * theta0 / (1 + p2hat * (theta0 - 1))
            T0 <- ((n1p * (n11 / n1p - p1hat)) * sqrt(1 / (n1p * p1hat * (1 - p1hat)) + 1 / (n2p * p2hat * (1 - p2hat))))}

            lower_limit <- function(theta0, n11, n21, n1p, n2p, alpha) {
            T0 <- score_test_statistic.Uncorrected(theta0, n11, n21, n1p, n2p)
            f <- T0 - qnorm(1 - alpha / 2, 0, 1)}

            upper_limit <- function(theta0, n11, n21, n1p, n2p, alpha) {
            T0 <- score_test_statistic.Uncorrected(theta0, n11, n21, n1p, n2p)
            f <- T0 + qnorm(1 - alpha / 2, 0, 1)}

            FM_CI <- function(p1,p2,q1,q2, alpha = 0.05) {
            L <- uniroot(lower_limit, c(0.000001, 100000),p1, q1, (p1+p2), (q1+q2), alpha)$root
            U <- uniroot(upper_limit, c(0.000001, 100000),p1, q1, (p1+p2), (q1+q2), alpha )$root
            c(L, U)}''')
        
        lower_FM  = robjects.r['FM_CI'](p1,q1,p2,q2)[0]  #type: ignore
        upper_FM  = robjects.r['FM_CI'](p1,q1,p2,q2)[1]  #type: ignore
        
        

        # 3. Confidence intervals for the Relative Risk
        ###############################################
        
        #Standard Error of the relative risk
        standard_error_Relative_Risk_katz = np.sqrt((((1 - proportion_sample_1)) / p1) + ((1-proportion_sample_2)/p2))
        standard_error_Relative_Risk_Walters = np.sqrt((1/(p1+0.5)) - (1/(sample_size_1+0.5)) + (1/(p2+0.5)) - (1/(sample_size_2+0.5)))

        # 1. Walters
        upper_rr_walters = np.exp(np.log(Realtive_risk_adjusted_Walters) + zcrit* standard_error_Relative_Risk_Walters)
        lower_rr_walters = np.exp(np.log(Realtive_risk_adjusted_Walters) - zcrit* standard_error_Relative_Risk_Walters)

        # 2. Katz (The Delta Method - See Altman's Book "Statistics with Confidecne: p.58-59)
        upper_rr_katz = np.exp(np.log(Realtive_risk) + zcrit* standard_error_Relative_Risk_katz)
        lower_rr_katz = np.exp(np.log(Realtive_risk) - zcrit* standard_error_Relative_Risk_katz)
        
        # 3. Calculate Standard Error for Jewell's Relative Risk (RR)
        #standard_error_jewell_rr = np.sqrt((1 / p1) + (1 / p2) + (1 / q1) + (1 / q2))
        variance_RRss = (Realtive_risk_unbiased**2) * (q1 / (p1*(p1+q1))) + (q2 / ((p2+1)*(p2+q2))) - (Realtive_risk_unbiased**2) * ((q1 / (p1*(p1+q1))) * (q2 / ((p2+1)*(p2+q2+1))))
        standard_error_jewell_rr = np.sqrt(variance_RRss)

        # Confidence Interval for Jewell's RR
        lower_ci_jewell = np.exp(np.log(Realtive_risk_adjusted_Walters) - zcrit* standard_error_jewell_rr)
        upper_ci_jewell = np.exp(np.log(Realtive_risk_adjusted_Walters) + zcrit* standard_error_jewell_rr)


        # 4. Miettinen and Nurminen (Koopman)
        def RRci(x1, n1, x2, n2, conf_level):
            z = abs(norm.ppf((1 - conf_level) / 2))
            a1 = n2 * (n2 * (n2 + n1) * x1 + n1 * (n2 + x1) * (z ** 2))
            a2 = -n2 * (n2 * n1 * (x2 + x1) + 2 * (n2 + n1) * x2 * x1 + n1 * (n2 + x2 + 2 * x1) * (z ** 2))
            a3 = 2 * n2 * n1 * x2 * (x2 + x1) + (n2 + n1) * (x2 ** 2) * x1 + n2 * n1 * (x2 + x1) * (z ** 2)
            a4 = -n1 * (x2 ** 2) * (x2 + x1)
            b1 = a2 / a1
            b2 = a3 / a1
            b3 = a4 / a1
            c1 = b2 - (b1 ** 2) / 3
            c2 = b3 - b1 * b2 / 3 + 2 * (b1 ** 3) / 27
            ceta = math.acos(math.sqrt(27) * c2 / (2 * c1 * math.sqrt(-c1)))
            t1 = -2 * math.sqrt(-c1 / 3) * math.cos(math.pi / 3 - ceta / 3)
            t2 = -2 * math.sqrt(-c1 / 3) * math.cos(math.pi / 3 + ceta / 3)
            t3 = 2 * math.sqrt(-c1 / 3) * math.cos(ceta / 3)
            p01 = t1 - b1 / 3
            p02 = t2 - b1 / 3
            p03 = t3 - b1 / 3
            p0sum = p01 + p02 + p03
            p0up = min(p01, p02, p03)
            p0low = p0sum - p0up - max(p01, p02, p03)
            ul = (1 - (n1 - x1) * (1 - p0up) / (x2 + n1 - (n2 + n1) * p0up)) / p0up
            ll = (1 - (n1 - x1) * (1 - p0low) / (x2 + n1 - (n2 + n1) * p0low)) / p0low
            cint = [ll, ul]
            return cint

        CI_Koopman = RRci(p1,sample_size_1, p2, sample_size_2, confidence_level)

        # 5+6 Inverse Sine and adjusted Inverse Sine
        Adjusted_Epsilon = math.asinh(0.5 * zcrit * np.sqrt(1 / p1 + 1 / p2 - 1 / (sample_size_1+ 1) - 1 / (sample_size_2 + 1)))
        Epsilon = math.asinh(0.5 * zcrit * np.sqrt(1/p1 + 1/p2 - 1/sample_size_1 - 1/sample_size_2))
        Lower_Sinh_adjusted = np.exp(np.log(Realtive_risk) - 2 * Adjusted_Epsilon)
        Upper_Sinh_adjusted = np.exp(np.log(Realtive_risk) + 2 * Adjusted_Epsilon)
        Lower_Sinh_ = np.exp(np.log(Realtive_risk) - 2 * Epsilon)
        Upper_Sinh_ = np.exp(np.log(Realtive_risk) + 2 * Epsilon)

        # 7. Zou and Donner (2008) - Wilson Based CI's
        wilson_CI_1 = proportion_confint(p1, sample_size_1, (1-confidence_level), method = "wilson")
        wilson_CI_2 = proportion_confint(p2, sample_size_2, (1-confidence_level), method = "wilson")
        
        Lower_Zou_Donner = np.exp( (np.log(p1/sample_size_1)) - (np.log(p2/sample_size_2)) - np.sqrt( (((np.log(p1/sample_size_1)) - (np.log(wilson_CI_1[0])))**2) + (((np.log(wilson_CI_2[1])) - (np.log(p2/sample_size_2)))**2)))
        Upper_Zou_Donner = np.exp( (np.log(p1/sample_size_1)) - (np.log(p2/sample_size_2)) + np.sqrt( (((np.log(wilson_CI_1[1])) - (np.log(p1/sample_size_1)))**2) + (((np.log(p2/sample_size_2)) - (np.log(wilson_CI_2[0])))**2)))
        
        # Confidence intervals for the Yules Q family
        #############################################

        # Standard Errors by Bishop, Fienberg, Holland - 2007 (Discrete Mulyivariete analysis). For the Y star its from Bonett and Star
        Standard_Error_Yules_Q = 0.5*(1-Yules_Q**2) * standard_error_odds_ratio
        Standard_Error_Yules_Y = 0.25*(1-Yules_Y**2) * standard_error_odds_ratio
        Standard_Error_Digbys_H = 0.5*0.75*(1-Digbys_H**2) * standard_error_odds_ratio
        Standard_Error_Yules_Y_Star = np.sqrt( (c**2/4) * ((1 /(p1+0.1)) + (1/(q2+0.1)) + (1/(p2+0.1)) +( 1/(q1+0.1))))
        
        Yules_Q_CI_lower = (Yules_Q - zcrit*Standard_Error_Yules_Q)
        Yules_Q_CI_upper = (Yules_Q + zcrit*Standard_Error_Yules_Q)

        Digbys_H_Lower = (Yules_Q - zcrit*Standard_Error_Digbys_H)
        Digbys_H_Upper = (Digbys_H + zcrit*Standard_Error_Digbys_H)

        Yules_Y_CI_lower  = (Yules_Y - zcrit*Standard_Error_Yules_Y)
        Yules_Y_CI_upper  = (Yules_Y + zcrit*Standard_Error_Yules_Y)

        Yules_Y_Star_CI_lower = np.tanh(np.arctanh(Yules_Y_Star) - zcrit * Standard_Error_Yules_Y_Star)
        Yules_Y_Star_CI_upper = np.tanh(np.arctanh(Yules_Y_Star) + zcrit * Standard_Error_Yules_Y_Star)
        

        results = {}
        
        # Descriptive Statistics
        results["Table 1 - Descriptive Statistics"] = ""
        results["--------------------------------"] = ""
        results["Sample 2's Proportion"] = round(proportion_sample_2, 4)
        results["Difference Between Proportions"] = round(proportion_sample_1 - proportion_sample_2, 4)
        results["Number Of Successes 1"] = p1
        results["Number Of Successes 2"] = p2
        results["Sample 1 Size"] = sample_size_1
        results["Sample 2 Size"] = sample_size_2
        results["Difference in Population"] = difference_in_population

        results["                                                                                                                                                       "] = ""
        
        # Inferential Statistics
        results["Table 2 - Inferntial Statistics"] = ""
        results["------------------------------"] = ""
        results["Standard Error (Wald/Wald Corrected)"] = round(Standard_Error_Wald, 4)
        results["Z-score (Wald)"] = round(Z_wald, 4)
        results["P-value (Wald)"] = np.around(p_value_wald, 4)
        results["Z-score (Wald Corrected)"] = round(Z_Wald_corrected, 4)
        results["P-value (Wald Corrected)"] = np.around(p_value_Wald_corrected, 4)
        results["Standard Error (Wald_H0/Wald Corrected_H0)"] = round(Standard_error_Wald_H0, 4)
        results["Z-score (Wald_H0)"] = round(Z_Wald_H0, 4)
        results["P-value (Wald_H0)"] = np.around(p_value_Wald_H0, 4)
        results["Z-score (Wald_H0 Corrected)"] = round(Z_Wald_H0_corrected, 4)
        results["P-value (Wald_H0 Corrected)"] = np.around(p_value_Wald_H0_corrected, 4)
        results["Standard Error Hauck & Anderson"] = round(Standard_error_HA, 4)
        results["Z-score Hauck & Anderson"] = round(z_score_HA, 4)
        results["p-value Hauck & Anderson"] = np.around(p_value_HA, 4)
        results["Standard Error Mantel & Haenszel"] = round(Standart_Error_MH, 4)
        results["Z-score Mantel & Haenszel"] = round(Z_MH, 4)
        results["p-value Mantel & Haenszel"] = np.around(pval_MH, 4)
        results["D statistic Barnard Exact"] = round(Barnard_D_Statistic, 4)
        results["p-value Barnard Exact"] = round(pvalue_barnard, 4)
        results["p-value Fisher Exact"] = round(Fisher_Exact_p_value, 4)
        results["                                                                                                                                                         "] = ""

        # More Effect Sizes
        results["Table 3 - Effect Sizes"] = ""
        results["----------------------"] = ""
        results["Effect Size - Cohen's h"] = round(cohens_h, 4)
        results["Effect Size - Cohen's W"] = round(Cohens_w, 4)
        results["Effect Size - Probit d"] = np.around(Probit_d, 4)
        results["Effect Size - Logit d"] = round(logit_d, 4)
        results["                                                                                                                                                        "] = ""

        # Difference Between Proportions CI's
        results["Table 4 - Difference Between Proportions CI's"] = ""
        results["---------------------------------------------"] = ""
        results["Confidence Intervals Wald"] = f"({round(lower_ci_wald, 4)}, {round(upper_ci_wald, 4)})"
        results["Confidence Intervals Wald Corrected"] = f"({round(lower_ci_wald_corrected, 4)}, {round(upper_ci_wald_corrected, 4)})"
        results["Confidence Intervals Haldane"] = f"({round(lower_ci_haldane, 4)}, {round(upper_ci_haldane, 4)})"
        results["Confidence Intervals Jefferey-Perks)"] = f"({round(lower_ci_JP, 4)}, {round(upper_ci_JP, 4)})"
        results["Confidence Intervals Mee"] = f"({round(CI_mee[0], 4)}, {round(CI_mee[1], 4)})"
        results["Confidence Intervals Miettinen-Nurminen)"] = f"({round(CI_mn[0], 4)}, {round(CI_mn[1], 4)})"
        results["Confidence Intervals Agresti-Caffo"] = f"({round(lower_ci_AC, 4)}, {round(upper_ci_AC, 4)})"
        results["Confidence Intervals Wilson"] = f"({round(CI_wilson_lower, 4)}, {round(CI_wilson_upper, 4)})"
        results["Confidence Intervals Wilson Corrected"] = f"({round(CI_wilsonc_lower, 4)}, {round(CI_wilsonc_upper, 4)})"
        results["Confidence Intervals Hauck-Anderson"] = f"({round(CI_lower_HA, 4)}, {round(CI_upper_HA, 4)})"
        results["Confidence Intervals Brown Lee Jeffreyes"] = f"({round(CI_BLJ_lower, 4)}, {round(CI_BLJ_upper, 4)})"
        results["Confidence Intervals Gart-Nam"] = f"({round(Gart_Nam_CI[0], 4)}, {round(Gart_Nam_CI[1], 4)})"
        results["Confidence Intervals Newcomb"] = f"({round(Newcomb_CI[0], 4)}, {round(Newcomb_CI[1], 4)})"
        results["                                                                                                                                                                      "] = ""


        # Odds Ratio
        results["Table 5 - Odds Ratios and their CI's"] = ""
        results["-----------------------------------"] = ""

        # Odds Ratio Types
        results["Odds Ratio Conditional (Corenfield)"] = round(odds_ratio_Conditional, 4)
        results["Odds Ratio Unconditional (Wald)"] = round(odds_ratio_wald, 4)
        results["Odds Ratio Unconditional (Wald) Corrected"] = round(odds_ratio_wald_adjusted, 4)
        results["Odds Ratio Wald with Small Sample Correction"] = round(odds_ratio_small_samples_adjusted, 4)
        results["Mid-p (Median Unbiased) Odds Ratio"] = round(odds_ratio_mid_p, 4)
        results["Yules Q"] = round(Yules_Q, 4)
        results["Yules Y"] = round(Yules_Y, 4)
        results["Digbys H"] = round(Digbys_H, 4)
        results["Yules Q Star"] = round(Yules_Y_Star, 4)
        results["Chi2 odds ratio"] = round(chi2_odds_ratio, 4) # We dont necessarily need this type of odds ratio

        # Odds Ratio Types Standard Errors
        results["Standard Error of the Odds Ratio"] = round(standard_error_odds_ratio, 4)
        results["Standard Error of the Odds Ratio (adjusted)"] = round(standard_error_odds_ratio_adjusted, 4)
        results["Standard Error of Yules Q"] = np.around(Standard_Error_Yules_Q, 4)
        results["Standard Error of Yules Y"] = np.around(Standard_Error_Yules_Y, 4)
        results["Standard Error of Digby's H"] = np.around(Standard_Error_Digbys_H, 4)
        results["Standard Error of Yule's Y*"] = np.around(Standard_Error_Yules_Y_Star, 4)

        # Odds Ratio CI's
        results["Confidence Intervals Odds Ratio Woolf"] = f"({round(lower_ci_woolf, 4)}, {round(upper_ci_woolf, 4)})"
        results["Confidence Intervals Odds Ratio Woolf Adjusted"] = f"({round(lower_or_woolf_adjusted, 4)}, {round(upper_or_woolf_adjusted, 4)})"
        results["Confidence Intervals Odds Ratio Fisher"] = f"({formatted_ci_lower}, {formatted_ci_upper})"
        results["Confidence Intervals Odds Ratio Mid P"] = f"({round(lower_ci_mid_p, 4)}, {round(upper_ci_mid_p, 4)})"
        results["Confidence Intervals Odds Ratio MN"] = f"({round(CI_MN[0], 4)}, {round(CI_MN[1], 4)})"
        results["Confidence Intervals Odds Ratio BP"] = f"({round(lower_bp, 4)}, {round(upper_ci_bp, 4)})"
        results["Confidence Intervals Odds Ratio Sinh"] = f"({round(lower_limit_sinh, 4)}, {round(upper_limit_sinh, 4)})"
        results["Confidence Intervals Odds Ratio Agresti"] = f"({round(ci_lower_agresti_ind, 4)}, {round(ci_upper_agresti_ind, 4)})"
        results["Confidence Intervals Odds Ratio FM"] = f"({round(lower_FM, 4)}, {round(upper_FM, 4)})"                
               
        results["Confidence Intervals Yules Q"] = f"({round(Yules_Q_CI_lower, 4)}, {round(Yules_Q_CI_upper, 4)})"
        results["Confidence Intervals Yules Y"] = f"({round(Yules_Y_CI_lower, 4)}, {round(Yules_Y_CI_upper, 4)})"
        results["Confidence Intervals Digby's H "] = f"({round(Digbys_H_Lower, 4)}, {round(Digbys_H_Upper, 4)})"
        results["Confidence Intervals Yules Y*"] = f"({round(Yules_Y_Star_CI_lower, 4)}, {round(Yules_Y_Star_CI_upper, 4)})"
        results["                                                                                                                                                                         "] = ""
                                                                                                                                                                              
        # Relative Risk / Risk Ratio
        results["Table 7 - Relative Risk and CI's"] = ""
        results["--------------------------------"] = ""
        
        # Relative Risk / Risk Ratio Measures
        results["Relaive Risk (Risk Ratio) (Katz)"] = round(Realtive_risk, 4)
        results["Relaive Risk (Risk Ratio) Adjusted (Walters)"] = round((Realtive_risk_adjusted_Walters), 4)
        results["Relaive Risk (Risk Ratio) for Small Samples (Jewell)"] = round((Realtive_risk_unbiased), 4)
        
        # Standard Errors        
        results["Standard Error Relaive Risk (Risk Ratio) (Katz)"] = round((standard_error_Relative_Risk_katz), 4)
        results["Standard Error Relaive Risk (Risk Ratio) Adjusted (Walters)"] = round((standard_error_Relative_Risk_Walters), 4)
        results["Standard Error Relaive Risk (Risk Ratio) for Small Samples (Jewell)"] = round((standard_error_jewell_rr), 4)

        # Confidence Intervals        
        results["Confidence Intervals Relative Risk - Katz"] = f"({round(lower_rr_katz, 4)}, {round(upper_rr_katz, 4)})"   
        results["Confidence Intervals Relative Risk - Walters"] = f"({round(lower_rr_walters, 4)}, {round(upper_rr_walters, 4)})"
        results["Confidence Intervals Relative Risk - Jewell"] = f"({round(lower_ci_jewell, 4)}, {round(upper_ci_jewell, 4)})"
        results["Confidence Intervals Relative Risk - Koopman"] = CI_Koopman
        results["Confidence Intervals Relative Risk - Inverse sine"] = f"({round(Lower_Sinh_, 4)}, {round(Upper_Sinh_, 4)})"
        results["Confidence Intervals Relative Risk - Inverse sine adjusted"] = f"({round(Lower_Sinh_adjusted, 4)}, {round(Upper_Sinh_adjusted, 4)})"
        results["Confidence Intervals Relative Risk - Zou & Donner"] = f"({round(Lower_Zou_Donner, 4)}, {round(Upper_Zou_Donner, 4)})"

        results["                                                                                                                                                                             "] = ""
  
        results["Table 8 - Other Risk Measures"] = ""
        results["-----------------------------"] = ""
        results["Incidental Rate Exposed"] = round((Incidental_Rate_Exposed), 4)
        results["Incidental Rate UnExposed"] = round((Incidental_Rate_UnExposed), 4)
        results["Incidental Rate Population"] = round((Incidental_Rate_Population), 4)
        results["Exposed Attributable Fraction"] = round((Exposed_Attributable_Fraction), 4)
        results["Exposed_Attributable_Fraction (%)"] = round((abs(Exposed_Attributable_Fraction_percentages)), 4)
        results["Population Atributable Risk"] = round((Population_Attributional_Risk), 4)
        results["Population Atributable Risk (%) "] = round((abs(Population_attributable_risk_percentages)), 4)
        results["Population Atributable Fraction (Relative Risk Reduction) "] = round((Population_Attributable_Fraction), 4)
        results["Population Atributable Fraction (%) "] = round((abs(Population_Attributable_Fraction_percentages*100)), 4)
        results["Number Needed to Treat (NNT)"] = round(NNT, 7)
        # results["Risk Difference (Absolute Risk Reduction)"] = round((proportion_sample_1 - proportion_sample_2), 4) # This is another name to the same thing

        return results



class two_ind_samples_proportions():
    @staticmethod
    def Two_Ind_Proportions_From_Parameters (params: dict) -> dict:

        # Set params
        confidence_level_percentages = params["Confidence Level"]
        proportion_sample_1 = params["Proportion Sample 1"]
        proportion_sample_2 = params["Proportion Sample 2"]
        sample_size_1 = params["Sample Size 1"]
        sample_size_2 = params["Sample Size 2"]
        difference_in_population = params["Difference in the Population"]
        Two_Independent_proportions_output = main_Two_sample_proportions(proportion_sample_1, proportion_sample_2, sample_size_1, sample_size_2, confidence_level_percentages, difference_in_population)

        results = {"Two Independent Proportion From Parameters": Two_Independent_proportions_output}
    
        # Displaying results in separate rows
        for key, value in results.items():
            print(f"{key}:")
        
        for inner_key, inner_value in value.items():
            print(f"  {inner_key}: {inner_value}")

        return results

    @staticmethod
    def Two_Indpendent_Proportions_From_data (params: dict) -> dict:
        
        # Set params
        confidence_level_percentages = params["Confidence Level"]
        column_1 = params["Column 1"]
        column_2 = params["Column 2"]
        population_proportion = params["Populations Difference"]
        defined_sucess = params["Defined Success Value"]

        # Calculate
        sample_size_1 = len(column_1)
        sample_size_2 = len(column_2)

        number_of_successes_1 = np.count_nonzero(column_1 == defined_sucess)
        proportion_sample_1 = number_of_successes_1/ sample_size_1
        number_of_successes_2 = np.count_nonzero(column_2 == defined_sucess)
        proportion_sample_2 = number_of_successes_2/ sample_size_2
        Two_independent_proportions_output = main_Two_sample_proportions(proportion_sample_1, proportion_sample_2, sample_size_1, sample_size_2, confidence_level_percentages, population_proportion)
        
        results = {}
        results = {"One Sample Proportion From Data": Two_independent_proportions_output}
    
        # Displaying results in separate rows
        for key, value in results.items():
            print(f"{key}:")
        
        for inner_key, inner_value in value.items():
            print(f"  {inner_key}: {inner_value}")

        return results

    @staticmethod
    def Two_Indpendent_Proportions_From_Frequencies (params: dict) -> dict:
        
        # Set params
        confidence_level_percentages = params["Confidence Level"]
        number_of_successes_1 = params["Number of Successes Sample 1"]
        number_of_successes_2 = params["Number of Successes Sample 2"]
        population_proportion = params["Populations Difference"]
        sample_size_1 = params["Sample Size 1"]
        sample_size_2 = params["Sample Size 2"]

        # Calculate
        proportion_sample_1 = number_of_successes_1/sample_size_1
        proportion_sample_2 = number_of_successes_2/sample_size_2

        Two_independent_proportions_output = main_Two_sample_proportions(proportion_sample_1, proportion_sample_2, sample_size_1, sample_size_2, confidence_level_percentages, population_proportion)
        results = {"One Sample Proportion From Frequencies": Two_independent_proportions_output}
    
        # Displaying results in separate rows
        for key, value in results.items():
            print(f"{key}:")
        
        for inner_key, inner_value in value.items():
            print(f"  {inner_key}: {inner_value}")

        return results
    

