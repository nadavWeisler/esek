
###############################################
# Effect Size for Indepednent Samples Z-Test ##
###############################################
import numpy as np
from scipy.stats import norm

# Relevant Functions for Two Independent Samples Z-Score
##########################################
def calculate_central_ci_from_cohens_d_two_samples(cohens_d, sample_size_1, sample_size_2, confidence_level):
    standard_error_es = np.sqrt(((sample_size_1 + sample_size_2)/(sample_size_1 * sample_size_2)) + ((cohens_d ** 2 / (2 * (sample_size_1 + sample_size_2)))))
    z_critical_value = norm.ppf(confidence_level + ((1-confidence_level)/2))
    ci_lower = cohens_d - standard_error_es * z_critical_value
    ci_upper = cohens_d + standard_error_es * z_critical_value
    return ci_lower, ci_upper, standard_error_es

class Two_independent_Samples_Z_Score():
    @staticmethod 
    def two_independent_sample_from_z_score(params: dict) -> dict:
        
        # Get parameters
        z_score = params["Z score"]
        sample_size_1 = params["Sample Size 1"]
        sample_size_2 = params["Sample Size 2"]
        confidence_level_prcentages = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_prcentages / 100
        total_sample_size = sample_size_1 + sample_size_2
        mean_sample_size = (sample_size_1 + sample_size_2) / 2
        cohens_d = ((2 * z_score) / np.sqrt(total_sample_size)) * np.sqrt(mean_sample_size / ((2 * sample_size_1 * sample_size_2) / total_sample_size))
        p_value = min(float(norm.sf((abs(z_score))) * 2), 0.99999)
        ci_lower, ci_upper, standard_error_es = calculate_central_ci_from_cohens_d_two_samples(cohens_d, sample_size_1, sample_size_2, confidence_level)

        # Set results
        results = {}
        results["Cohen's d"] = round(cohens_d, 4)
        results["Z score"] = round(z_score, 4)
        results["p-value"] = round(p_value, 4)
        results["Cohen's d CI Lower"] = round(ci_lower, 4)
        results["Cohen's d CI Upper"] = round(ci_upper, 4)
        results["Standard Error of Cohen's d"] = round(standard_error_es, 4)
        results["Statistical Line"] = f" \033[3mz\033[0m = {z_score:.3f}, \033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}, Cohen's d = {cohens_d:.3f}, {confidence_level_prcentages}% CI [{ci_lower:.3f}, {ci_upper:.3f}]"
        results["Statistical Line"] = results["Statistical Line"].replace(f"\033[3mp\033[0m {'= {:.3f}'.format(p_value)}", f"\033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}").replace(".000", ".").replace("= 0.", "= .").replace("< 0.", "< .")
        return results

    
    @staticmethod
    def two_independent_samples_from_params(params: dict) -> dict:
        
        # Get parameters
        sample_mean_1 = params["Sample Mean 1"]
        sample_mean_2 = params["Sample Mean 2"]
        Population_sd_1 = params["Standard Deviation Population 1"]
        Population_sd_2 = params["Standard Deviation Population 2"]
        sample_size_1 = params["Sample Size 1"]
        sample_size_2 = params["Sample Size 2"]
        population_diff = params["Difference in the Population"]
        confidence_level_prcentages = params["Confidence Level"]

        # Calculate
        confidence_level = confidence_level_prcentages / 100
        var1 = Population_sd_1 ** 2
        var2 = Population_sd_2 ** 2
        Standard_Error_Mean_Difference_Population = np.sqrt(((var1 / sample_size_1) + (var2/sample_size_2)))
        SD_difference_pop = np.sqrt((var1 + var2)/2)
        z_score = (population_diff - (sample_mean_1 - sample_mean_2)) / Standard_Error_Mean_Difference_Population
        cohens_d = abs((population_diff - (sample_mean_1 - sample_mean_2)) / SD_difference_pop)
        p_value = min(float(norm.sf((abs(z_score))) * 2), 0.99999)
        ci_lower, ci_upper, standard_error_es = calculate_central_ci_from_cohens_d_two_samples(cohens_d, sample_size_1, sample_size_2, confidence_level)

        # Set results
        results = {}
        results["Cohen's d"] = round(cohens_d, 4)
        results["Z score"] = round(z_score, 4)
        results["p-value"] = round(p_value, 4)
        results["Standard Error of Mean Difference"] = round(Standard_Error_Mean_Difference_Population, 4)
        results["Cohen's d CI Lower"] = round(ci_lower, 4)
        results["Cohen's d CI Upper"] = round(ci_upper, 4)
        results["Standard Error of Cohen's d"] = round(standard_error_es, 4)
        results["Statistical Line"] = f" \033[3mz\033[0m = {z_score:.3f}, \033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}, Cohen's d = {cohens_d:.3f}, {confidence_level_prcentages}% CI [{ci_lower:.3f}, {ci_upper:.3f}]"
        results["Statistical Line"] = results["Statistical Line"].replace(f"\033[3mp\033[0m {'= {:.3f}'.format(p_value)}", f"\033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}").replace(".000", ".").replace("= 0.", "= .").replace("< 0.", "< .")
        return results

    @staticmethod
    def two_independent_samples_from_data(params: dict) -> dict:
        
        # Get params
        column_1 = params["column 1"]
        column_2 = params["column 2"]
        population_diff = params["Difference in the Population"]
        Population_sd_1 = params["Standard Deviation Population 1"]
        Population_sd_2 = params["Standard Deviation Population 2"]
        confidence_level_prcentages = params["Confidence Level"]

        # Calculate
        confidence_level = confidence_level_prcentages / 100
        sample_mean_1 = np.mean(column_1)
        sample_mean_2 = np.mean(column_2)
        sample_std_1 = np.std(column_1, ddof=1)
        sample_std_2 = np.std(column_2, ddof=1)
        var1 = Population_sd_1**2 #calculating the variances from the standard deviation
        var2 = Population_sd_2**2
        sample_size_1 = len(column_1)
        sample_size_2 = len(column_2)
        Standard_Error_Mean_Difference_Population = np.sqrt(((var1/sample_size_1) + (var2/sample_size_2)))  #This is the standrt error of mean's estimate in Z test
        SD_difference_pop = np.sqrt((var1+var2)/2) #this is the denominator for the effect size
        z_score = (population_diff - (sample_mean_1 - sample_mean_2)) / Standard_Error_Mean_Difference_Population
        cohens_d = abs((population_diff - (sample_mean_1 - sample_mean_2))/SD_difference_pop) #This is the effect size for one sample z-test Cohen's d
        p_value = min(float(norm.sf((abs(z_score))) * 2), 0.99999)
        ci_lower, ci_upper, standard_error_es = calculate_central_ci_from_cohens_d_two_samples(cohens_d, sample_size_1, sample_size_2, confidence_level)

        results = {}
        results["Cohen's d"] = round(cohens_d, 4)
        results["Z-score"] = round(z_score, 4)
        results["p-value"] = round(p_value, 4)
        results["Standard Error of Mean Difference"] = round(Standard_Error_Mean_Difference_Population, 4)
        results["Cohen's d CI Lower"] = round(ci_lower, 4)
        results["Cohen's d CI Upper"] = round(ci_upper, 4)
        results["Standard Error of Cohen's d"] = round(standard_error_es, 4)
        results["Mean Sample 1"] = round(sample_mean_1, 4)
        results["Mean Sample 2"] = round(sample_mean_2, 4)
        results["Standard Deviation Sample 1"] = round(sample_std_1, 4)
        results["Standard Deviation Sample 2"] = round(sample_std_2, 4)
        results["Standard Deviation of the Popoulations Difference"] = round(SD_difference_pop, 4)
        results["Sample Size 1"] = round(sample_size_1, 4)
        results["Sample Size 2"] = round(sample_size_2, 4)
        results["Standard Deviation Population 1"] = round(Population_sd_1, 4)
        results["Standard Deviation Population 2"] = round(Population_sd_2, 4)
        results["Statistical Line"] = f" \033[3mz\033[0m = {z_score:.3f}, \033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}, Cohen's d = {round(cohens_d,3)}, {confidence_level_prcentages}% CI [{ci_lower:.3f}, {ci_upper:.3f}]"
        results["Statistical Line"] = results["Statistical Line"].replace(f"\033[3mp\033[0m {'= {:.3f}'.format(p_value)}", f"\033[3mp\033[0m {'= {:.3f}'.format(p_value) if p_value >= 0.001 else '< .001'}").replace(".000", ".").replace("= 0.", "= .").replace("< 0.", "< .")
        return results
    
    # Things to consider
    # 1. Consider call the efffect size here dpop and if ones assume equal variances there should maybe only report on one SD of the population
    # 2. Since we know the mean and sd in the population we should use the varience of the effect size in the population
