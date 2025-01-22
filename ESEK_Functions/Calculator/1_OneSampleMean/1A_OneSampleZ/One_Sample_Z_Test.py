###############################################
##### Effect Size for One Sample Z-Test #######
###############################################
import numpy as np
from scipy.stats import norm

# Relevant Functions for One Sample Z-Score
##########################################
def calculate_central_ci_from_cohens_d_one_sample(cohens_d, sample_size, confidence_level): # This is a function that calculates the Confidence Intervals of the Effect size in One Sample Z_score test (or two dependent samples)
    standard_error_es = np.sqrt((1 / sample_size) + ((cohens_d**2 / (2 * sample_size)))) #Note that since the effect size in the population and its standart deviation are unknown we can estimate it based on the sample. For the one sample case we will use the Hedges and Olkin 1985 Formula to estimate the standart deviation of the effect size
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = cohens_d - standard_error_es * z_critical_value,  cohens_d + standard_error_es * z_critical_value
    return ci_lower, ci_upper, standard_error_es

class One_Sample_ZTests():
    @staticmethod
    def one_sample_from_z_score(params: dict) -> dict:

        # Set params
        z_score = params["Z-score"]
        sample_size = params["Sample Size"]
        confidence_level_percentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentages/100
        p_value = min(float(norm.sf((abs(z_score))) * 2), 0.99999)
        cohens_d = z_score / np.sqrt(sample_size)
        ci_lower, ci_upper, standard_error_es = calculate_central_ci_from_cohens_d_one_sample(cohens_d, sample_size,confidence_level)

        # Set results
        results = {}
        results["Cohen's d"] = round(cohens_d,4)
        results["Z-score"] = round(z_score, 4)
        results["p-value"] = round(p_value,4)
        results["Cohen's d CI Lower"] = round(ci_lower, 4)
        results["Cohen's d CI Upper"] = round(ci_upper, 4)
        results["Standard Error of the Effect Size"] = round(standard_error_es,4)
        results["Statistical Line"] = f" \033[3mz\033[0m = {z_score:.3f}, \033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}, Cohen's d = {cohens_d:.3f}, {confidence_level_percentages}% CI [{ci_lower:.3f},{ci_upper:.3f}]"
        results["Statistical Line"] = results["Statistical Line"].replace(f"\033[3mp\033[0m {'= {:.3f}'.format(p_value)}", f"\033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}").replace(".000", ".").replace("= 0.", "= .").replace("< 0.", "< .")
        return results

    @staticmethod
    def one_sample_from_parameters(params: dict) -> dict:
        
        # Set params
        population_mean = params["Population Mean"]
        population_sd = params["Popoulation Standard Deviation"]
        sample_mean = params["Sample's Mean"]
        sample_size = params["Sample Size"]
        confidence_level_percentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentages/100
        mean_standard_error = population_sd / np.sqrt(sample_size)
        confidence_level = confidence_level_percentages/100
        z_score = (population_mean - sample_mean) / mean_standard_error
        cohens_d = ((population_mean - sample_mean) / population_sd)
        p_value = min(float(norm.sf((abs(z_score))) * 2), 0.99999)
        ci_lower, ci_upper, standard_error_es = calculate_central_ci_from_cohens_d_one_sample(cohens_d, sample_size, confidence_level)

        # Set Results
        results = {}
        results["Cohen's d"] = round(cohens_d, 4)
        results["Z-score"] = round(z_score, 4)
        results["p-value"] = round(p_value, 4)
        results["Standart Error of the Mean"] = round(mean_standard_error, 4)
        results["Cohen's d CI Lower"] = round(ci_lower, 4)
        results["Cohen's d CI Upper"] = round(ci_upper, 4)
        results["Standrd Error of the Effect Size"] = round(standard_error_es, 4)
        results["Statistical Line"] = f" \033[3mz\033[0m = {z_score:.3f}, \033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}, Cohen's d = {cohens_d:.3f}, {confidence_level_percentages}% CI [{ci_lower:.3f},{ci_upper:.3f}]"
        results["Statistical Line"] = results["Statistical Line"].replace(f"\033[3mp\033[0m {'= {:.3f}'.format(p_value)}", f"\033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}").replace(".000", ".").replace("= 0.", "= .").replace("< 0.", "< .")
        return results

    @staticmethod
    def one_sample_from_data(params: dict) -> dict:
        
        # Set params
        column_1 = params["column_1"]
        population_mean = params["Population Mean"]
        population_sd = params["Popoulation Standard Deviation"]
        confidence_level_percentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentages/100
        sample_mean = np.mean(column_1)
        sample_sd = np.std(column_1, ddof = 1)
        diff_mean = population_mean - sample_mean         
        sample_size = len(column_1)
        standard_error = population_sd/(np.sqrt(sample_size))        
        z_score = diff_mean / standard_error
        cohens_d = ((diff_mean)/population_sd) #This is the effect size for one sample z-test Cohen's d
        p_value = min(float(norm.sf((abs(z_score))) * 2), 0.99999)
        ci_lower, ci_upper, standard_error_es = calculate_central_ci_from_cohens_d_one_sample(cohens_d, sample_size, confidence_level)

        # Set results
        results = {}
        results["Cohen's d"] = round(cohens_d, 4)
        results["Z-score"] = round(z_score, 4)
        results["p-value"] = round(p_value, 20)
        results["Standart Error of the Mean"] = round(standard_error, 4)
        results["Cohen's d CI Lower"] = round(ci_lower, 4)
        results["Cohen's d CI Upper"] = round(ci_upper, 4)
        results["Standrd Error of the Effect Size"] = round(standard_error_es, 4)
        results["Sample's Mean"] = round(sample_mean, 4)
        results["Sample's Standard Deviation"] = round(sample_sd, 4)
        results["Difference Between Means"] = round(diff_mean, 4)
        results["Sample Size"] = round(sample_size, 4)
        results["Statistical Line"] = f" \033[3mz\033[0m = {z_score:.3f}, \033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}, Cohen's d = {cohens_d:.3f}, {confidence_level_percentages}% CI [{ci_lower:.3f},{ci_upper:.3f}]"
        results["Statistical Line"] = results["Statistical Line"].replace(f"\033[3mp\033[0m {'= {:.3f}'.format(p_value)}", f"\033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}").replace(".000", ".").replace("= 0.", "= .").replace("< 0.", "< .")
        return results

