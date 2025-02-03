    
###############################################
# Effect Size for One Sample Proportion #######
###############################################

import numpy as np
import math
from scipy.stats import norm, beta, binom, t
from statsmodels.stats.proportion import proportion_confint
from scipy.optimize import newton


# Main Function for One sample Proportion (It is identical for all input options)

def main_one_sample_proportion (proportion_sample, sample_size, population_proportion, confidence_level_percentages): 
        confidence_level = confidence_level_percentages / 100
        cohens_g = abs(proportion_sample - population_proportion)
        phi_sample = 2 *(np.arcsin(np.sqrt(proportion_sample)))
        phi_population = 2 *(np.arcsin(np.sqrt(population_proportion)))
        cohens_h = phi_sample - phi_population
        Standard_Error_Wald = np.sqrt((proportion_sample * (1 - proportion_sample)) / sample_size)
        Standard_Error_score = np.sqrt((population_proportion * (1 - population_proportion)) / sample_size)
        z_score_wald = (proportion_sample - population_proportion) / Standard_Error_Wald

        Number_of_Succeses_sample = proportion_sample*sample_size
        Number_Of_Failures_sample = sample_size - Number_of_Succeses_sample
        Number_of_Succeses_Population = population_proportion*sample_size
        Number_Of_Failures_Population = sample_size - Number_of_Succeses_Population

        z_score = (proportion_sample - population_proportion) / Standard_Error_score
        correction = sample_size * population_proportion + 0.5
        Z_score_wald_corrected = ((proportion_sample * sample_size) - correction) / np.sqrt((proportion_sample * (1 - proportion_sample) * sample_size))
        Z_score_corrected = ((proportion_sample * sample_size) - correction) / np.sqrt((population_proportion * (1 - population_proportion) * sample_size))

        p_value_wald = norm.sf(abs(z_score_wald))
        p_value_wald_corrected =  norm.sf(abs(Z_score_wald_corrected))
        p_value_score =  norm.sf(abs(z_score))
        p_value_score_corrected =  norm.sf(abs(Z_score_corrected))

        # Risk Measures (No CI's for these measures)
        Relative_Risk = proportion_sample/population_proportion
        Odds_Ratio = (proportion_sample/(1-proportion_sample)) / (population_proportion/(1-population_proportion))
        Risk_Difference = proportion_sample - population_proportion


        # Confidence Intervals for Cohen's h (* I need to verify these CI's)
        z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
        SE_arcsine = 2* np.sqrt(0.25 * (1/sample_size))
        lower_ci_Cohens_h = cohens_h - z_critical_value * SE_arcsine
        upper_ci_Cohens_h = cohens_h + z_critical_value * SE_arcsine


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
        Lower_CI_Wilson_Corrected=(2*Number_of_Succeses_sample + z_critical_value**2-1-z_critical_value*np.sqrt(z_critical_value**2-2-1/sample_size + 4*(Number_of_Succeses_sample/sample_size)*(sample_size*(1-Number_of_Succeses_sample/sample_size)+1)))/(2*(sample_size + z_critical_value**2)) 
        Upper_CI_Wilson_Corrected=min((2*Number_of_Succeses_sample + z_critical_value**2+1+z_critical_value*np.sqrt(z_critical_value**2+2-1/sample_size + 4*(Number_of_Succeses_sample/sample_size)*(sample_size*(1-Number_of_Succeses_sample/sample_size)-1)))/(2*(sample_size + z_critical_value**2)),1)
        

        # 6. Logit
        lambdahat = math.log(Number_of_Succeses_sample/(sample_size-Number_of_Succeses_sample))
        term1 = sample_size/(Number_of_Succeses_sample*(sample_size-Number_of_Succeses_sample))
        lambdalow = lambdahat - z_critical_value*np.sqrt(term1)
        lambdaupper = lambdahat + z_critical_value*np.sqrt(term1)
        logitlower = math.exp(lambdalow)/(1 + math.exp(lambdalow))
        logitupper =min( math.exp(lambdaupper)/(1 + math.exp(lambdaupper)),1)

        # 7. Jeffereys
        lowerjeffreys = beta.ppf((1-confidence_level)/2, Number_of_Succeses_sample+0.5, sample_size-Number_of_Succeses_sample+0.5)
        upperjeffreys = min(beta.ppf(1-(1-confidence_level)/2, Number_of_Succeses_sample+0.5, sample_size-Number_of_Succeses_sample+0.5),1)

        # 8. Clopper-Pearson CI's
        lowerCP = beta.ppf((1-confidence_level)/2, Number_of_Succeses_sample, sample_size-Number_of_Succeses_sample+1)
        upperCP= max(beta.ppf(1-(1-confidence_level)/2, Number_of_Succeses_sample+1, sample_size-Number_of_Succeses_sample),1)

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
                return min(a1, a2)

            CI_lower_blaker = beta.ppf((1 - conf_level) / 2, x, n - x + 1)
            CI_upper_blaker = beta.ppf(1 - (1 - conf_level) / 2, x + 1, n - x)

            while x != 0 and acceptance_probability(x, n, CI_lower_blaker + tol) < (1 - conf_level):
                CI_lower_blaker += tol
            while x != n and acceptance_probability(x, n, CI_upper_blaker - tol) < (1 - conf_level):
                CI_upper_blaker -= tol

            ci = [max(CI_lower_blaker,0), min(CI_upper_blaker,1)]
            
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
        results["Sample's Proportion"] = proportion_sample
        results["Population's Proportion"] = population_proportion
        results["Sample Size"] = sample_size
        results["Confidence Level"] = confidence_level
        results["Number of Successes (Sample)"] = Number_of_Succeses_sample
        results["Number of Failures (Sample)"] = Number_Of_Failures_sample
        results["Expected Number of Successes (Population)"] = Number_of_Succeses_Population
        results["Expected Number of Failures (Population)"] = Number_Of_Failures_Population
        results["Standard Error (Wald)"] = round(Standard_Error_Wald, 4)
        results["Standard Error (Score)"] = round(Standard_Error_score, 4)
        results["Z-score (Wald)"] = round(z_score_wald, 4)
        results["Z-score (Score)"] = round(z_score, 4)
        results["Z-score (Wald Corrected)"] = round(Z_score_wald_corrected, 4)
        results["Z-score (Score Corrected)"] = round(Z_score_corrected, 4)
        results["P-value (Wald)"] = np.round(p_value_wald, 4)
        results["P-value (Wald) Corrected"] = np.around(p_value_wald_corrected, 4)
        results["P-value (Score)"] = np.around(p_value_score, 4)
        results["P-value (Score) Corrected"] = np.around(p_value_score_corrected, 4)
        results["Cohen's g"] = round(cohens_g, 4)
        results["cohen's h"] = round(cohens_h, 40)
        results["Relative Risk"] = round(Relative_Risk, 4)
        results["Odds Ratio"] = round(Odds_Ratio, 4)
        results["Cohen's h CI's"] = [lower_ci_Cohens_h, upper_ci_Cohens_h]
        results["Cohen's h Standard Error"] = [SE_arcsine]
        formatted_p_value = "{:.3f}".format(p_value_score).lstrip('0') if p_value_score >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Cohens'h"] = " \033[3mZ\033[0m = {:.3f}, {}{}, Cohen's h = {:.3f},  {}% CI [{:.3f}, {:.3f}]".format(z_score, '\033[3mp = \033[0m' if p_value_score >= 0.001 else '', formatted_p_value, cohens_h, confidence_level_percentages, lower_ci_Cohens_h, upper_ci_Cohens_h)

        # Confidence Intervals for One Sample Proportion
        results["Agresti Coull CI's"] = np.round(np.array(AC_CI),4)
        results["Wald CI"] = np.round(np.array(Wald_CI),4)
        results["Wald CI Corrected"] = np.around(Wald_Corrected,4)
        results["Wilson"] = np.around(np.array(wilson_CI),4)
        results["Wilson Corrected"] = np.around(np.array([Lower_CI_Wilson_Corrected,Upper_CI_Wilson_Corrected]),4)
        results["logit"] = np.around(np.array([logitlower,logitupper]),4)
        results["Jeffereys"] = np.around(np.array([lowerjeffreys, upperjeffreys]),4)
        results["Clopper-Pearson"] = np.around(np.array([lowerCP, upperCP]),4)
        results["Arcsine"] = np.around(np.array([arcsinelower, arcsineupper]),4)
        results["Pratt"] = np.around(np.array([lowerPratt, upperPratt]),4)
        results["Blaker"] = np.around(np.array(CI_blakers),4)
        results["Mid-p"] = np.around(np.array(midp_cis),4)
        
        return results



class one_sample_proportion():
    @staticmethod
    def One_Sample_Proportion_From_Parameters (params: dict) -> dict:
    
        # Set params
        confidence_level_percentages = params["Confidence Level"]
        proportion_sample = params["Proportion in the Sample"]
        population_proportion = params["Proportion in the Population"] # Default is 0.5
        sample_size = params["Sample Size"]
        one_sample_proportion_output = main_one_sample_proportion(proportion_sample, sample_size, population_proportion, confidence_level_percentages)
        
        results = {"One Sample Proportion From Parameters": one_sample_proportion_output}
    
        # Displaying results in separate rows
        for key, value in results.items():
            print(f"{key}:")
        
        for inner_key, inner_value in value.items():
            print(f"  {inner_key}: {inner_value}")

        return results

    @staticmethod
    def One_Sample_Proportion_From_data (params: dict) -> dict:
        
        # Set params
        confidence_level_percentages = params["Confidence Level"]
        column_1 = params["Column 1"]
        population_proportion = params["Proportion in the Population"]
        defined_sucess = params["Defined Success Value"]

        # Calculate
        sample_size = len(column_1)
        number_of_successes = np.count_nonzero(column_1 == defined_sucess)
        proportion_sample = number_of_successes/ sample_size
        one_sample_proportion_output = main_one_sample_proportion(proportion_sample, sample_size, population_proportion, confidence_level_percentages)
        results = {}
        results = {"One Sample Proportion From Data": one_sample_proportion_output}
    
        # Displaying results in separate rows
        for key, value in results.items():
            print(f"{key}:")
        
        for inner_key, inner_value in value.items():
            print(f"  {inner_key}: {inner_value}")

        return results

    @staticmethod
    def One_Sample_Proportion_From_frequency (params: dict) -> dict:
        
        # Set params
        confidence_level_percentages = params["Confidence Level"]
        number_of_successes = params["Number of Successes"]
        population_proportion = params["Proportion in the Population"]
        sample_size = params["Sample Size"]

        # Calculate
        proportion_sample = number_of_successes/sample_size
        one_sample_proportion_output = main_one_sample_proportion(proportion_sample, sample_size, population_proportion, confidence_level_percentages)
        results = {"One Sample Proportion From Frequencies": one_sample_proportion_output}
    
        # Displaying results in separate rows
        for key, value in results.items():
            print(f"{key}:")
        
        for inner_key, inner_value in value.items():
            print(f"  {inner_key}: {inner_value}")

        return results
    

    @staticmethod
    def proportion_of_hits (params: dict) -> dict:

        # Set params
        number_correct_Answers = params["Number Of Correct Answers"]
        number_of_trials = params["Number of Trials"]
        Number_of_Choices = params["Number of Choices"]
        confidence_level_percentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentages / 100

        Proportion_Correct = number_correct_Answers/number_of_trials
        pi = (Proportion_Correct*(Number_of_Choices-1)) / (1+Proportion_Correct*(Number_of_Choices-2))
        Standard_Error_pi = (1/np.sqrt(number_of_trials)) * (  (pi*(1-pi)) / (np.sqrt(Proportion_Correct*(1-Proportion_Correct))) ) 
        Z_score = (pi-0.5)/Standard_Error_pi
        p_value = norm.sf(abs(Z_score))
        zcrit = t.ppf(1 - (1 - confidence_level) / 2, 100000)
        lower_CI = pi - zcrit * Standard_Error_pi
        upper_CI = pi + zcrit * Standard_Error_pi
        
        results = {}
        
        results["Proportion of Hits [π]"] = round(pi, 7)
        results["Standard Error of π"] = round(Standard_Error_pi,7)
        results["Z-score"] = round(Z_score, 7)
        results["p vlaue of π"] = np.around(np.array(p_value),4)
        results["Confidence Intervals for π"] = f"({round(lower_CI, 4)}, {round(upper_CI, 4)})"

        return results

