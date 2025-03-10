
###########################################
#### Confidence Intervals for Proportions #
###########################################

from scipy.stats import norm, beta, binom
from statsmodels.stats.proportion import proportion_confint
from scipy.optimize import newton
import numpy as np
import math
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep
from scipy.optimize import newton, root_scalar
import rpy2.robjects as robjects


class CI_Constructor_Proportions():
    
    ##########################################
    ## 1.1 One Sample Proportion CI ##########
    ##########################################

    @staticmethod
    def One_Sample_Proportion_CI(params: dict) -> dict:
        
        proportion_sample = params["Proportion"]
        sample_size = params["Sample Size"]
        Confidnece_Level_Percentages = params["Confidence Level"]
        confidence_level = Confidnece_Level_Percentages / 100

        Number_of_Succeses_sample = proportion_sample * sample_size
        
        z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))

        # Confidence Intervals for One Sample Proportion
        ################################################

        # 1. Agresti-Coull
        AC_CI = proportion_confint(Number_of_Succeses_sample, sample_size, (1-confidence_level), method = "agresti_coull")
        
        # 2. Wald CI's
        Wald_CI = proportion_confint(Number_of_Succeses_sample, sample_size, (1-confidence_level), method = "normal")

        # 3. Wald_Corrected 
        correction = 0.05/sample_size
        Wald_Corrected = np.array(Wald_CI) + np.array([-correction, correction])
        
        # 4. Wilson
        wilson_CI = proportion_confint(Number_of_Succeses_sample, sample_size, (1-confidence_level), method = "wilson")
        
        # 5. Wilson Corrected
        LowerCi_Wilson_Corrected=(2*Number_of_Succeses_sample + z_critical_value**2-1-z_critical_value*np.sqrt(z_critical_value**2-2-1/sample_size + 4*(Number_of_Succeses_sample/sample_size)*(sample_size*(1-Number_of_Succeses_sample/sample_size)+1)))/(2*(sample_size + z_critical_value**2)) 
        UpperCi_Wilson_Corrected=min((2*Number_of_Succeses_sample + z_critical_value**2+1+z_critical_value*np.sqrt(z_critical_value**2+2-1/sample_size + 4*(Number_of_Succeses_sample/sample_size)*(sample_size*(1-Number_of_Succeses_sample/sample_size)-1)))/(2*(sample_size + z_critical_value**2)),1)

        # 6. Logit
        lambdahat = math.log(Number_of_Succeses_sample/(sample_size-Number_of_Succeses_sample))
        term1 = sample_size/(Number_of_Succeses_sample*(sample_size-Number_of_Succeses_sample))
        lambdalow = lambdahat - z_critical_value*np.sqrt(term1)
        lambdaupper = lambdahat + z_critical_value*np.sqrt(term1)
        logitlower = math.exp(lambdalow)/(1 + math.exp(lambdalow))
        logitupper =min( math.exp(lambdaupper)/(1 + math.exp(lambdaupper)),1)

        # 7. Jeffereys
        lowerjeffreys = beta.ppf((1-confidence_level)/2, Number_of_Succeses_sample+0.5, sample_size-Number_of_Succeses_sample+0.5)
        upperjeffreys = min(beta.ppf(1-(1-confidence_level)/2, Number_of_Succeses_sample+0.5, sample_size-Number_of_Succeses_sample+0.5),1) # type: ignore

        # 8. Clopper-Pearson CI's
        lowerCP = beta.ppf((1-confidence_level)/2, Number_of_Succeses_sample, sample_size-Number_of_Succeses_sample+1)
        upperCP= max(beta.ppf(1-(1-confidence_level)/2, Number_of_Succeses_sample+1, sample_size-Number_of_Succeses_sample),1) # type: ignore

        # 9. arcsine CI's 1 (Kulynskaya)
        ptilde = (Number_of_Succeses_sample + 0.375)/(sample_size + 0.75)
        arcsinelower = math.sin(math.asin(np.sqrt(ptilde)) - 0.5*z_critical_value/np.sqrt(sample_size))**2
        arcsineupper = min(math.sin(math.asin(np.sqrt(ptilde)) + 0.5*z_critical_value/np.sqrt(sample_size))**2,1)

        # 10. Pratt
        A = ((Number_of_Succeses_sample + 1)/(sample_size-Number_of_Succeses_sample))**2
        B = 81 * (Number_of_Succeses_sample+1) * (sample_size-Number_of_Succeses_sample) - 9*sample_size - 8
        C = -3 * z_critical_value * np.sqrt(9*(Number_of_Succeses_sample+1)*(sample_size - Number_of_Succeses_sample) * (9*sample_size + 5 - z_critical_value**2) + sample_size + 1)
        D = 81 * (Number_of_Succeses_sample + 1)**2 - 9 *(Number_of_Succeses_sample+1)* (2+z_critical_value**2) + 1
        E = 1 + A * ((B+C)/D)**3
        A2 = (Number_of_Succeses_sample/ (sample_size-Number_of_Succeses_sample-1)) **2
        B2 = 81 * (Number_of_Succeses_sample) * (sample_size-Number_of_Succeses_sample-1) - 9*sample_size - 8
        C2 = 3 * z_critical_value * np.sqrt(9*Number_of_Succeses_sample*(sample_size-Number_of_Succeses_sample-1) * (9*sample_size + 5 - z_critical_value**2) + sample_size + 1)
        D2 = 81 * Number_of_Succeses_sample**2 - (9 *Number_of_Succeses_sample) * (2+z_critical_value**2) + 1
        E2 = 1 + A2 * ((B2+C2)/D2)**3

        upperPratt = min(1/E, 1)    
        lowerPratt = max(1/E2, 0)

        # 11. Blaker
        def blakersCI(x, n, conf_level=0.95, tol=0.00001):
        
            def acceptance_probability(x, n, p):
                probabilty1 = 1 - binom.cdf(x - 1, n, p)
                probabilty2 = binom.cdf(x, n, p)
                a1 = probabilty1 + binom.cdf(binom.ppf(probabilty1, n, p) - 1, n, p)
                a2 = probabilty2 + 1 - binom.cdf(binom.ppf(1 - probabilty2, n, p), n, p)
                return min(a1, a2) # type: ignore

            CI_lower_blaker = beta.ppf((1 - conf_level) / 2, x, n - x + 1)
            CI_upper_blaker = beta.ppf(1 - (1 - conf_level) / 2, x + 1, n - x)

            while x != 0 and acceptance_probability(x, n, CI_lower_blaker + tol) < (1 - conf_level):
                CI_lower_blaker += tol
            while x != n and acceptance_probability(x, n, CI_upper_blaker - tol) < (1 - conf_level):
                CI_upper_blaker -= tol

            ci = [max(CI_lower_blaker,0), min(CI_upper_blaker,1)] # type: ignore
            
            return ci
        
        CI_blakers = blakersCI(Number_of_Succeses_sample, sample_size, confidence_level)

        
        # 12. Mid-p
        def calculate_midp(x, n, conf_level):
            def f_low(pi):return 0.5 * binom.pmf(x, n, pi) + binom.cdf(x - 1, n, pi) - (1 + conf_level) / 2
            def f_up(pi):return 0.5 * binom.pmf(x, n, pi) + binom.cdf(x - 1, n, pi) - (1 - conf_level) / 2
            CI_lower_midp = newton(f_low, x/n)
            CI_upper_midp = newton(f_up, x/n)
            return CI_lower_midp, CI_upper_midp

        midp_cis = calculate_midp(Number_of_Succeses_sample, sample_size, confidence_level)


        results = {}

        # Confidence Intervals for One Sample Proportion
        results["Agresti Coull CI's"] = np.round(np.array(AC_CI),4)
        results["Wald CI"] = np.round(np.array(Wald_CI),4)
        results["Wald CI Corrected"] = np.around(Wald_Corrected,4)
        results["Wilson"] = np.around(np.array(wilson_CI),4)
        results["Wilson Corrected"] = np.around(np.array([LowerCi_Wilson_Corrected,UpperCi_Wilson_Corrected]),4)
        results["logit"] = np.around(np.array([logitlower,logitupper]),4)
        results["Jeffereys"] = np.around(np.array([lowerjeffreys, upperjeffreys]),4)
        results["Clopper-Pearson"] = np.around(np.array([lowerCP, upperCP]),4)
        results["Arcsine"] = np.around(np.array([arcsinelower, arcsineupper]),4)
        results["Pratt"] = np.around(np.array([lowerPratt, upperPratt]),4)
        results["Blaker"] = np.around(np.array(CI_blakers),4)
        results["Mid-p"] = np.around(np.array(midp_cis),4)

        return results
    

    ##########################################
    ## 1.2 Paired Samples Proportion #########
    ##########################################

    @staticmethod
    def Paired_Samples_Proportion_CI(params: dict) -> dict:
        
        # Set params
        confidence_level_percentages = params["Confidence Level"]
        proportion_sample_1 = params["Proportion Sample 1"]
        proportion_sample_2 = params["Proportion Sample 2"]
        proportion_of_yeses = params["Proportion of Sucess in Both Samples"]
        sample_size = params["Number of Pairs"]

        confidence_level = confidence_level_percentages / 100

        # Preperations
        yes_yes = proportion_of_yeses * sample_size
        yes_no = (proportion_sample_1 * sample_size) - yes_yes
        no_yes = (proportion_sample_2 * sample_size) - yes_yes
        no_no = sample_size - (yes_yes + yes_no + no_yes)


        # TO DO 
        # Consider adding the log odd ratios
        #adding +0.5 to deal with zeros
        #Check Panello 2010 for multiple Proportions

        sample_size = yes_yes + yes_no + no_yes + no_no

        # p1total = The number of peopole that said yes in Variable 1 (including yes1 and no1 and yes1 and no2)
        # p2total = The number of peopole that said yes in Variable 2 (including yes2 and no1 and yes2 and no2)
        p1total = yes_yes + yes_no
        p2total = yes_yes + no_yes
        Proportion_Sample_1 = p1total / sample_size
        Proportion_Sample_2 = p2total / sample_size


        #Difference Between Proportions and confidence intervals
        difference_between_proportions = Proportion_Sample_1 - Proportion_Sample_2
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
        
        A1 = (2 * sample_size * ((p1total)/sample_size) + z_critical_value**2) / (2 * sample_size + 2 * z_critical_value**2)
        B1 = (z_critical_value * np.sqrt(z_critical_value**2 + 4 * sample_size * (p1total/sample_size) * (1 - (p1total/sample_size)))) / (2 * sample_size + 2 * z_critical_value**2)  
        A2 = (2 * sample_size * ((p2total)/sample_size) + z_critical_value**2) / (2 * sample_size + 2 * z_critical_value**2)
        B2 = (z_critical_value * np.sqrt(z_critical_value**2 + 4 * sample_size * (p2total/sample_size) * (1 - (p2total/sample_size)))) / (2 * sample_size + 2 * z_critical_value**2)
        lower_p1 = A1 - B1
        upper_p1 = A1 + B1
        lower_p2 = A2 - B2
        upper_p2 = A2 + B2

        if p1total == 0 or p2total == 0 or (sample_size-p1total) == 0 or (sample_size - p2total) == 0:
                products_correction= 0
        else:
            marginals_product = p1total*p2total*(sample_size-p1total)*(sample_size-p2total)
            cells_product = yes_yes*no_no - no_yes*no_yes
            if cells_product > sample_size / 2:
                products_correction = (cells_product - sample_size / 2) / np.sqrt(marginals_product)
            elif cells_product >= 0 and cells_product <= sample_size / 2:
                products_correction = 0
            else:
                products_correction = cells_product / np.sqrt(marginals_product)

        LowerCi_newcomb = difference_between_proportions - np.sqrt((Proportion_Sample_1 - lower_p1)**2 + (upper_p2 - Proportion_Sample_2)**2 - 2 * products_correction * (Proportion_Sample_1 - lower_p1) * (upper_p2 - Proportion_Sample_2))
        UpperCi_newcomb = difference_between_proportions + np.sqrt((Proportion_Sample_2 - lower_p2)**2 + (upper_p1 - Proportion_Sample_1)**2 - 2 * products_correction * (Proportion_Sample_2 - lower_p2) * (upper_p1 - Proportion_Sample_1))
        
        results = {}

        results["Difference Between Proportions"] = round(difference_between_proportions, 7)
        results["Confidence Intervals Wald"] = f"({round(LowerCi_WALD, 4)}, {round(UpperCi_WALD, 4)})"
        results["Confidence Intervals Wald Corrected (Edwards)"] = f"({round(LowerCi_WALD_Corrected, 4)}, {round(UpperCi_WALD_Corrected, 4)})"
        results["Confidence Intervals Wald Corrected (Yates)"] = f"({round(LowerCi_WALD_Corrected_Yates, 4)}, {round(UpperCi_WALD_Corrected_Yates, 4)})"
        results["Confidence Intervals adjusted (Agresti & Min, 2005)"] = f"({round(LowerCi_AM, 4)}, {round(UpperCi_AM, 4)})"
        results["Confidence Intervals adjusted (Bonett & Price, 2012)"] = f"({round(LowerCi_BP, 4)}, {round(UpperCi_BP, 4)})"
        results["Confidence Intervals (NewComb)"] = f"({round(LowerCi_newcomb, 4)}, {round(UpperCi_newcomb, 4)})"

        return results
    
    ##########################################
    ## 1.3 Indpendent Proportions ############
    ##########################################

    @staticmethod
    def Independent_Samples_Proportion_CI(params: dict) -> dict:

        # Set params
        confidence_level_percentages = params["Confidence Level"]
        proportion_sample_1 = params["Proportion Sample 1"]
        proportion_sample_2 = params["Proportion Sample 2"]
        sample_size_1 = params["Sample Size 1"]
        sample_size_2 = params["Sample Size 2"]
        difference_in_population = params["Difference in the Population"]


        # Calculation
        #############
        confidence_level = confidence_level_percentages / 100

        p1 = proportion_sample_1 * sample_size_1
        p2 = proportion_sample_2 * sample_size_2
        sample_size = sample_size_1 + sample_size_2 
        samples_difference = proportion_sample_1-proportion_sample_2



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



    
        results = {}

        results["Confidence Intervals Wald"] = f"({round(lower_ci_wald, 4)}, {round(upper_ci_wald, 4)})"
        results["Confidence Intervals Wald Corrected"] = f"({round(lower_ci_wald_corrected, 4)}, {round(upper_ci_wald_corrected, 4)})"
        results["Confidence Intervals Haldane"] = f"({round(lower_ci_haldane, 4)}, {round(upper_ci_haldane, 4)})"
        results["Confidence Intervals Miettinen-Nurminen)"] = f"({round(CI_mn[0], 4)}, {round(CI_mn[1], 4)})"
        results["Confidence Intervals Mee"] = f"({round(CI_mee[0], 4)}, {round(CI_mee[1], 4)})"
        results["Confidence Intervals Agresti-Caffo"] = f"({round(lower_ci_AC, 4)}, {round(upper_ci_AC, 4)})"
        results["Confidence Intervals Wilson"] = f"({round(CI_wilson_lower, 4)}, {round(CI_wilson_upper, 4)})"
        results["Confidence Intervals Wilson Corrected"] = f"({round(CI_wilsonc_lower, 4)}, {round(CI_wilsonc_upper, 4)})"
        results["Confidence Intervals Hauck-Anderson"] = f"({round(CI_lower_HA, 4)}, {round(CI_upper_HA, 4)})"
        results["Confidence Intervals Brown Lee Jeffreyes"] = f"({round(CI_BLJ_lower, 4)}, {round(CI_BLJ_upper, 4)})"
        results["Confidence Intervals Gart-Nam"] = Gart_Nam_CI
        results["Confidence Intervals Newcomb"] = Newcomb_CI
   
        return results