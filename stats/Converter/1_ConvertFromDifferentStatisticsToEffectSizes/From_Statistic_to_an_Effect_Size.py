

##################### Effect Size Converter From Statistic to an Effect Size ###############################

import numpy as np
import math

################################
# 1. From z-Score to Cohen's d #
################################

class Convert_Statistic_to_Effect_Size():
    
    # 1. From Z Scores to Cohen's d
    @staticmethod
    def from_z_to_cohens_d_one_sample(params: dict) -> dict:
        
        z_score = params["Z-score"]
        sample_size = params["Sample Size"]

        cohens_d = z_score / np.sqrt(sample_size)
        
        results = {}
        results["Cohen's d"] = round(cohens_d,4)
        return results

    @staticmethod
    def from_z_to_cohens_d_paired_samples(params: dict) -> dict:
        
        z_score = params["Z-score"]
        sample_size = params["Sample Size"]

        cohens_d = z_score / np.sqrt(sample_size)
        
        results = {}
        results["Cohen's d"] = round(cohens_d,4)
        return results

    @staticmethod
    def from_z_to_cohens_d_indepednent_samples(params: dict) -> dict:
        
        z_score = params["Z-score"]
        sample_size_1 = params["Sample Size 1"]
        sample_size_2 = params["Sample Size 2"]

        total_sample_size = sample_size_1 + sample_size_2
        mean_sample_size = (sample_size_1 + sample_size_2) / 2
        cohens_d = ((2 * z_score) / np.sqrt(total_sample_size)) * np.sqrt(mean_sample_size / ((2 * sample_size_1 * sample_size_2) / total_sample_size))

        results = {}
        results["Cohen's d"] = round(cohens_d,4)
        return results
    
    
    
    #################################
    # 2. From t Scores to Cohen's d #
    #################################

    @staticmethod
    def from_t_to_cohens_d_one_sample(params: dict) -> dict:
        
        t_score = params["t-score"]
        sample_size = params["Sample Size"]
        degrees_of_freedom = params["Degrees of Freedom"]

        cohens_d = (t_score/np.sqrt(degrees_of_freedom))
        correction = math.exp(math.lgamma(degrees_of_freedom/2) - math.log(math.sqrt(degrees_of_freedom/2)) - math.lgamma((degrees_of_freedom-1)/2))
        hedges_g = correction*cohens_d
            
        results = {}
        results["Cohen's d"] = round(cohens_d,4)
        results["hedges g"] = round(cohens_d,4)

        return results

    @staticmethod
    def from_t_to_cohens_d_paired_samples(params: dict) -> dict:
        
        t_score = params["t-score"]
        sample_size = params["Sample Size"]

        cohens_d = t_score / np.sqrt(sample_size)
        
        results = {}
        results["Cohen's d"] = round(cohens_d,4)
        return results

    @staticmethod
    def from_t_to_cohens_d_indpendent_samples(params: dict) -> dict:
        
        t_score = params["t-score"]
        
        sample_size_1 = params["Sample Size 1"]
        sample_size_2 = params["Sample Size 2"]

        cohens_d = t_score / np.sqrt(sample_size)
        
        results = {}
        results["Cohen's d"] = round(cohens_d,4)
        return results

        

