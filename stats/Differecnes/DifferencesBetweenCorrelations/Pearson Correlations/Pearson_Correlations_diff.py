
###################################################
########### Differences Between Correlations ######
###################################################

import numpy as np
import math
from scipy.stats import norm, t

class Differences_Between_Pearson_Correlations():
    @staticmethod
    def independent_correlations(params: dict) -> dict:
    
        r12 = params["Correlation Between Variable 1 and Variable 2"]
        r34 = params["Correlation Between Variable 3 and Variable 4"]
        sample_size_1 = params["Sample Size Correlation r12"]
        sample_size_2 = params["Sample Size Correlation r34"]
        confidence_level_percentages = params["Confidence Level"]

        # Preperation
        confidence_level = confidence_level_percentages / 100
        # Convertion Method - Convert the spearmn into pearosn and then use its statistical signifnicance and confidence intervals - Myers Sirois, 2004
        #converted_pearson_1 = 2 * np.sin(r12 * (math.pi/6))     
        #converted_pearson_2 = 2 * np.sin(r34 * (math.pi/6))

        # Preperations     
        Standard_Error_r12 = np.sqrt((1/(sample_size_1-3)))
        Standard_Error_r34 = np.sqrt((1/(sample_size_2-3)))
        Zr12 = np.arctan(r12)
        Zr34 = np.arctan(r34)
        Cohens_q = Zr12 - Zr34
        difference_between_correlations = r12-r34

        # Significcance for single correlations
        t_score_r12 = (r12 * np.sqrt(sample_size_1 - 2)) / (np.sqrt(1-r12**2))
        t_score_r34 = (r34 * np.sqrt(sample_size_2 - 2)) / (np.sqrt(1-r34**2))

        p_value_r12 = t.sf(t_score_r12, (sample_size_1-2))
        p_value_r34 = t.sf(t_score_r34, (sample_size_2-2))


        # Significance of the Difference
        Standard_Error_Of_Difference = np.sqrt((1/(sample_size_1-3)) + (1/(sample_size_2-3)))
        Z_Statistic = (Zr12-Zr34)/Standard_Error_Of_Difference
        p_value_difference = norm.sf(Z_Statistic)

        # Confidence Intervals
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)

        # Confidence Intervals For single correlations
        Standard_Error_Correlation12 = np.sqrt(1 / (sample_size_1 - 3))
        Standard_Error_Correlation34 = np.sqrt(1 / (sample_size_2 - 3))

        Lower_r12_CI_Fisher = np.tanh(Zr12 - Standard_Error_Correlation12 * zcrit)
        Upper_r12_CI_Fisher = np.tanh(Zr12 + Standard_Error_Correlation12 * zcrit)
        Lower_r34_CI_Fisher = np.tanh(Zr34 - Standard_Error_Correlation34 * zcrit)
        Upper_r34_CI_Fisher = np.tanh(Zr34 + Standard_Error_Correlation34 * zcrit)

        # Confidence Intervals for the Difference between indpednent samples
        
        # 1. Fisher
        lower_ci_difference_fisher = np.tanh((Zr12 - Zr34) - zcrit * Standard_Error_Of_Difference)
        upper_ci_difference_fisher = np.tanh((Zr12 - Zr34) + zcrit * Standard_Error_Of_Difference)
 
        # 2. Zou (2007)
        lower_ci_Zou = difference_between_correlations - ((r12 - Lower_r12_CI_Fisher)**2+(Upper_r34_CI_Fisher-r34)**2)**0.5
        upper_ci_Zou = difference_between_correlations + ((Upper_r12_CI_Fisher - r12)**2+(r34-Lower_r34_CI_Fisher)**2)**0.5

        results= {}

        results["Fisher Zr_1"]= Zr12
        results["Fisher Zr_2"] = Zr34
        results["Difference_Between Correlations"]= difference_between_correlations
        results["Cohens q"]= Cohens_q
        results["Statistic"] = Z_Statistic
        results["p-value for the difference"] = p_value_difference
        results["Standrd Error Fisher"] = Standard_Error_Of_Difference
        results["Confidence Intervals Fisher"] = f"({round(lower_ci_difference_fisher, 4)}, {round(upper_ci_difference_fisher, 4)})"
        results["Confidence Intervals Zou"] = f"({round(lower_ci_Zou, 4)}, {round(upper_ci_Zou, 4)})"
        formatted_p_value_r12 = "{:.3f}".format(p_value_r12).lstrip('0') if p_value_r12 >= 0.001 else "\033[3mp\033[0m < .001"
        formatted_p_value_r34 = "{:.3f}".format(p_value_r34).lstrip('0') if p_value_r34 >= 0.001 else "\033[3mp\033[0m < .001"
        formatted_p_value_difference = "{:.3f}".format(p_value_difference).lstrip('0') if p_value_difference >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line r12"] = "\033[3mr12\033[0m({}) = {}{}, {}{}, {}{}% CI [{}{}, {}{}]".format((sample_size_1 - 2), ('-' if np.round(r12,3) < 0 else ''), str(np.abs(np.round(r12,3))).lstrip('0').rstrip(''), '\033[3mp = \033[0m' if p_value_r12 >= 0.001 else '', formatted_p_value_r12, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(Lower_r12_CI_Fisher,3) < 0 else ''), str(np.abs(np.round(Lower_r12_CI_Fisher,3))).lstrip('0').rstrip(''), ('-' if np.round(Upper_r12_CI_Fisher,3) < 0 else ''), str(np.abs(np.round(Upper_r12_CI_Fisher,3))).lstrip('0').rstrip(''))
        results["Statistical Line r34"] = "\033[3mr34\033[0m({}) = {}{}, {}{}, {}{}% CI [{}{}, {}{}]".format((sample_size_2 - 2), ('-' if np.round(r34,3) < 0 else ''), str(np.abs(np.round(r34,3))).lstrip('0').rstrip(''), '\033[3mp = \033[0m' if p_value_r34 >= 0.001 else '', formatted_p_value_r34, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(Lower_r34_CI_Fisher,3) < 0 else ''), str(np.abs(np.round(Lower_r34_CI_Fisher,3))).lstrip('0').rstrip(''), ('-' if np.round(Upper_r34_CI_Fisher,3) < 0 else ''), str(np.abs(np.round(Upper_r34_CI_Fisher,3))).lstrip('0').rstrip(''))
        results["Statistical Line difference"] = "\033[3mZ\033[0m = {}, {}{}, {}{}% CI [{}{}, {}{}]".format( round(Z_Statistic,3), '\033[3mp = \033[0m' if p_value_difference >= 0.001 else '', formatted_p_value_difference, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(lower_ci_Zou,3) < 0 else ''), str(np.abs(np.round(lower_ci_Zou,3))).lstrip('0').rstrip(''), ('-' if np.round(upper_ci_Zou,3) < 0 else ''), str(np.abs(np.round(upper_ci_Zou,3))).lstrip('0').rstrip(''))

        return results # type: ignore
    
    @staticmethod
    def dependent_correlations_Non_Overlapped(params: dict) -> dict:
        
            r12 = params["Correlation Between Variable 1 and Variable 2"]
            r34 = params["Correlation Between Variable 3 and Variable 4"]
            r13 = params["Correlation Between Variable 1 and Variable 3"]
            r14 = params["Correlation Between Variable 1 and Variable 4"]
            r23 = params["Correlation Between Variable 2 and Variable 3"]
            r24 = params["Correlation Between Variable 2 and Variable 4"]
            sample_size = params["Sample Size"]
            confidence_level_percentages = params["Confidence Level"]

            # Preperation
            sample_size = np.array(sample_size)
            confidence_level = confidence_level_percentages / 100
            Zr12 = math.atanh(r12)
            Zr34 = math.atanh(r34)
            mean_r = (r12 + r34) / 2
            mean_Zr = math.atan((np.arctanh(r12) + np.arctanh(r34)) / 2)
            Term1 = ((r13 - r12 * r23) * (r24 - r23 * r34) + (r14 - r13 * r34) * (r23 - r12 * r13) + (r13 - r14 * r34) * (r24 - r12 * r14) + (r14 - r12 * r24) * (r23 - r24 * r34))
            Term2 = ((r13 - mean_r * r23) * (r24 - r23 * mean_r) + (r14 - r13 * mean_r) * (r23 - mean_r * r13) + (r13 - r14 * mean_r) * (r24 - mean_r * r14) + (r14 - mean_r * r24) * (r23 - r24 * mean_r))
            Term3 = ((r13 - mean_Zr * r23) * (r24 - r23 * mean_Zr) + (r14 - r13 * mean_Zr) * (r23 - mean_Zr * r13) + (r13 - r14 * mean_Zr) * (r24 - mean_Zr * r14) + (r14 - mean_Zr * r24) * (r23 - r24 * mean_Zr))



            Standard_Error_r = np.sqrt(1/(sample_size-3))
            Cohens_q = Zr12 - Zr34
            difference_between_correlations = r12-r34

            # Significcance of each correlation
            t_score_r12 = (r12 * np.sqrt(sample_size - 2)) / (np.sqrt(1-r12**2))
            t_score_r34 = (r34 * np.sqrt(sample_size - 2)) / (np.sqrt(1-r34**2))
            p_value_r12 = t.sf(t_score_r12, (sample_size-2))
            p_value_r34 = t.sf(t_score_r34, (sample_size-2))


            # Significcance of the Difference between correlations
            # Method 1 - Pearson & Filon 1898
            statistic_pearson = (np.sqrt(sample_size) * (r12 - r34)) / np.sqrt(((1 - r12**2)**2 + (1 - r34**2)**2) - Term1)
            p_value_pearson = norm.sf(statistic_pearson)*2

            # Method 2 - Dunn and Clark, 1969
            statistic_Dunn =    (np.sqrt(sample_size - 3) * (Zr12 - Zr34)) / (np.sqrt(2 - 2 * (Term1 / (2 * (1 - r12**2) * (1 - r34**2)))))
            p_value_Dunn = norm.sf(statistic_Dunn)*2

            # Method 3 - Steiger, 1980
            statistic_Steiger =  (np.sqrt(sample_size - 3) * (Zr12 - Zr34)) / (np.sqrt(2 - 2 * (Term2 / (2 * (1 - mean_r**2)**2))))
            p_value_Steiger = norm.sf(statistic_Steiger)*2

            # Method 4 - Raghunathan Rsoenthal & Rubin 1996
            statistic_Raghunathan = np.sqrt((sample_size - 3)/2) * (Zr12 - Zr34) / np.sqrt(1 - (Term1 / (2 * (1 - r12**2) * (1 - r34**2))))
            p_value_Raghunathan = norm.sf(statistic_Raghunathan)*2

            # Method 5 - Silver, Hittner & May 2004
            statistic_Silver = (math.sqrt(sample_size - 3) * (math.atanh(r12) - math.atanh(r34))) / (math.sqrt(2 - 2 * (Term3 / (2 * (1 - mean_Zr**2)**2))))
            p_value_Silver = norm.sf(statistic_Silver)*2

            
            # Confidence Intervals
            c = (0.5 * r12 * r34 * (r13**2 + r14**2 + r23**2 + r24**2) + r13 * r24 + r14 * r23 - (r12 * r13 * r14 + r12 * r23 * r24 + r13 * r23 * r34 + r14 * r24 * r34)) / ((1 - r12**2) * (1 - r34**2))

            # Confidence Intervals Single Correlation
            zcrit = abs(norm.ppf((1 - confidence_level) / 2)) 
            Standard_Error_Correlation = math.sqrt(1 / (sample_size - 3))
            lower_ci_r12 = math.tanh(Zr12 - zcrit * Standard_Error_Correlation)
            upper_ci_r12 = math.tanh(Zr12 + zcrit * Standard_Error_Correlation)
            lower_ci_r34 = math.tanh(Zr34 - zcrit * Standard_Error_Correlation)
            upper_ci_r34 = math.tanh(Zr34 + zcrit * Standard_Error_Correlation)

            # Confidence Interval for the difference (Zou)
            Lower_CI_Zou = r12 - r34 - math.sqrt((r12 - lower_ci_r12)**2 + (upper_ci_r34 - r34)**2 - 2 * c * (r12 - lower_ci_r12) * (upper_ci_r34 - r34))
            Upper_CI_Zou = r12 - r34 + math.sqrt((upper_ci_r12 - r12)**2 + (r34 - lower_ci_r34)**2 - 2 * c * (upper_ci_r12 - r12) * (r34 - lower_ci_r34))


            results= {}

            results["Fisher Zr_1"] = Zr12
            results["Fisher Zr_2"] = Zr34
            results["Difference_Between Correlations"] = difference_between_correlations
            results["Cohens q"] = Cohens_q
            results["Statistic_Pearson"] = statistic_pearson
            results["p_value_Pearson"] = p_value_pearson
            results["Statistic_Dunn"] = statistic_Dunn
            results["p_value_Dunn"] = p_value_Dunn
            results["Statistic_Steiger"] = statistic_Steiger
            results["p_value_Steiger"] = p_value_Steiger
            results["Statistic_Raghunathan"] = statistic_Raghunathan
            results["p_value_Raghunathan"] = p_value_Raghunathan
            results["Statistic_Silver"] = statistic_Silver
            results["p_value_Silver"] = p_value_Silver
            results["Standrd Error of the correlation"] = Standard_Error_r
            results["Confidence Intervals Zou"] = f"({round(Lower_CI_Zou, 4)}, {round(Upper_CI_Zou, 4)})"
            formatted_p_value_r12 = "{:.3f}".format(p_value_r12).lstrip('0') if p_value_r12 >= 0.001 else "\033[3mp\033[0m < .001"
            formatted_p_value_r34 = "{:.3f}".format(p_value_r34).lstrip('0') if p_value_r34 >= 0.001 else "\033[3mp\033[0m < .001"
            formatted_p_value_difference = "{:.3f}".format(p_value_Silver).lstrip('0') if p_value_Silver >= 0.001 else "\033[3mp\033[0m < .001"

            results["Statistical Line r12"] = "\033[3mr12\033[0m({}) = {}{}, {}{}, {}{}% CI [{}{}, {}{}]".format((sample_size - 2), ('-' if np.round(r12,3) < 0 else ''), str(np.abs(np.round(r12,3))).lstrip('0').rstrip(''), '\033[3mp = \033[0m' if p_value_r12 >= 0.001 else '', formatted_p_value_r12, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(lower_ci_r12,3) < 0 else ''), str(np.abs(np.round(lower_ci_r12,3))).lstrip('0').rstrip(''), ('-' if np.round(upper_ci_r12,3) < 0 else ''), str(np.abs(np.round(upper_ci_r12,3))).lstrip('0').rstrip(''))
            results["Statistical Line r34"] = "\033[3mr34\033[0m({}) = {}{}, {}{}, {}{}% CI [{}{}, {}{}]".format((sample_size - 2), ('-' if np.round(r34,3) < 0 else ''), str(np.abs(np.round(r34,3))).lstrip('0').rstrip(''), '\033[3mp = \033[0m' if p_value_r34 >= 0.001 else '', formatted_p_value_r34, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(lower_ci_r34,3) < 0 else ''), str(np.abs(np.round(lower_ci_r34,3))).lstrip('0').rstrip(''), ('-' if np.round(upper_ci_r34,3) < 0 else ''), str(np.abs(np.round(upper_ci_r34,3))).lstrip('0').rstrip(''))
            results["Statistical Line difference"] = "\033[3mZ\033[0m = {}, {}{}, {}{}% CI [{}{}, {}{}]".format( round(statistic_Silver,3), '\033[3mp = \033[0m' if p_value_Silver >= 0.001 else '', formatted_p_value_difference, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(Lower_CI_Zou,3) < 0 else ''), str(np.abs(np.round(Lower_CI_Zou,3))).lstrip('0').rstrip(''), ('-' if np.round(Upper_CI_Zou,3) < 0 else ''), str(np.abs(np.round(Upper_CI_Zou,3))).lstrip('0').rstrip(''))

            return results

    
    @staticmethod
    def dependent_correlations_Overlapped(params: dict) -> dict:
        
        r12 = params["Correlation Between Variable 1 and Variable 2"]
        r13 = params["Correlation Between Variable 1 and Variable 3"]
        r23 = params["Correlation Between Variable 2 and Variable 3"]
        sample_size = params["Sample Size"]
        confidence_level_percentages = params["Confidence Level"]

        # Preperation
        sample_size = np.array(sample_size)
        confidence_level = confidence_level_percentages / 100
        Zr12 = math.atanh(r12)
        Zr13 = math.atanh(r13)
        Correlation_Difference = r12 - r13
        df =  (sample_size - 3)
        Term1 = (r12**2 + r13**2 - 2 * r23 * r12 * r13) / (1 - r23**2)
        mean_r = (r12 + r13) / 2
        Mean_r_Squared = (r12**2 + r13**2) / 2
        mean_zr = np.tanh((np.arctanh(r12) + np.arctanh(r13)) / 2)
        Cohens_q = Zr12 - Zr13
        Standard_Error_Correlation = 1 / np.sqrt(sample_size-3)

        # Significcance of each correlation
        t_score_r12 = (r12 * np.sqrt(sample_size - 2)) / (np.sqrt(1-r12**2))
        t_score_r13 = (r13 * np.sqrt(sample_size - 2)) / (np.sqrt(1-r13**2))
        p_value_r12 = t.sf(t_score_r12, (sample_size-2))
        p_value_r13 = t.sf(t_score_r13, (sample_size-2))


        # Significcance of the Difference between correlations

        # Pearson 1898
        statistic_pearson =  np.sqrt(sample_size) * Correlation_Difference / (np.sqrt((1 - r12**2)**2 + (1 - r13**2)**2 - 2 * (r23 * (1 - r12**2 - r13**2) - 0.5 * r12 * r13 * (1 - r12**2 - r13**2 - r23**2))))
        p_value_pearson = norm.sf(statistic_pearson)*2

        # Hotelling, 1940 (t)
        statistic_hotelling = (r12 - r13) * np.sqrt((sample_size - 3) * (1 + r23)) / (np.sqrt(2 * (1 + 2 * r12 * r13 * r23 - r12**2 - r13**2 - r23**2)))
        p_value_hotelling = t.sf(statistic_hotelling, (sample_size-2))*2

        # Williams, 1959 (t)
        statistic_williams = (r12 - r13) * math.sqrt(((sample_size - 1) * (1 + r23)) / (2 * ((sample_size - 1) / (sample_size - 3)) * (1 + 2 * r12 * r13 * r23 - r12 ** 2 - r13 ** 2 - r23 ** 2) + ((r12 + r13) / 2) ** 2 * (1 - r23) ** 3))
        p_value_williams = t.sf(statistic_williams, (sample_size-3))*2

        # Olkin, 1967
        statistic_Olkin = Correlation_Difference * np.sqrt(sample_size) / (np.sqrt((1 - r12**2)**2 + (1 - r13**2)**2 - 2 * r23**3 - (2 * r23 - r12 * r13) * (1 - r12**2 - r13**2 - r23**2)))
        p_value_Olkin = norm.sf(statistic_Olkin) *2

        # Dunn, 1969
        statistic_Dunn = (np.sqrt(df) * (np.arctanh(r12) - np.arctanh(r13))) / (np.sqrt(2 - 2 * ((r23 * (1 - r12**2 - r13**2) - 0.5 * (r12 * r13) * (1 - r12**2 - r13**2 - r23**2)) / ((1 - r12**2) * (1 - r13**2)))))
        p_value_Dunn = norm.sf(statistic_Dunn)*2

        # Hendrickson, 1970 (t)
        statistic_Hendrickson = (r12 - r13) * math.sqrt((sample_size - 3) * (1 + r23)) / (math.sqrt(2 * (1 + 2 * r12 * r13 * r23 - r12 ** 2 - r13 ** 2 - r23 ** 2) + (((r12 - r13) ** 2 * (1 - r23) ** 3) / (4 * (sample_size - 1)))))

        p_value_Hendrickson = t.sf(statistic_Hendrickson, (sample_size-3))*2

        # Steiger, 1970
        statistic_steiger = (math.sqrt(sample_size - 3) * (Zr12 - Zr13)) / ( math.sqrt(2 - 2 * ((r23 * (1 - 2 * mean_r ** 2) - 0.5 * mean_r ** 2 * (1 - 2 * mean_r ** 2 - r23 ** 2)) / ((1 - mean_r ** 2) ** 2))))
        p_value_steiger = norm.sf(statistic_steiger)*2

        # Meng, 1992
        f = 1 if ((1 - r23) / (2 * (1 - Mean_r_Squared)) > 1) and not np.isnan((1 - r23) / (2 * (1 - Mean_r_Squared))) else (1 - r23) / (2 * (1 - Mean_r_Squared))
        statistic_Meng = ( np.arctanh(r12) - np.arctanh(r13)) * np.sqrt(df / (2 * (1 - r23) * ((1 - f * Mean_r_Squared) / (1 - Mean_r_Squared))))
        p_value_Meng = norm.sf(statistic_Meng)*2

        Lower_Ci_Meng = (np.arctanh(r12) - np.arctanh(r13)) - (norm.ppf((1 - confidence_level) / 2, loc=0, scale=1) * np.sqrt((2 * (1 - r23) * ((1 - f * Mean_r_Squared) / (1 - Mean_r_Squared))) / df))
        Upper_Ci_Meng = (np.arctanh(r12) - np.arctanh(r13)) + (norm.ppf((1 - confidence_level) / 2, loc=0, scale=1) * np.sqrt((2 * (1 - r23) * ((1 - f * Mean_r_Squared) / (1 - Mean_r_Squared))) / df))

        # Hittner, 2003
        statistic_Hittner = (np.sqrt(sample_size - 3) * (math.atanh(r12) - math.atanh(r13))) / (np.sqrt(2 - 2 * ((r23 * (1 - 2 * mean_zr ** 2) - 0.5 * mean_zr ** 2 * (1 - 2 * mean_zr ** 2 - r23 ** 2)) / (1 - mean_zr ** 2) ** 2)))
        p_value_Hittner = norm.sf(statistic_Hittner)

        # Confidence Intervals
        
        # For single r
        lower_ci_r12 =  np.tanh( np.arctanh(r12) - ( (norm.ppf( (1 - confidence_level) / 2)) * Standard_Error_Correlation))
        upper_ci_r12 =  np.tanh( np.arctanh(r12) + ( (norm.ppf( (1 - confidence_level) / 2)) * Standard_Error_Correlation))
        lower_ci_r13 =  np.tanh( np.arctanh(r13) - ( (norm.ppf( (1 - confidence_level) / 2)) * Standard_Error_Correlation))
        upper_ci_r13 =  np.tanh( np.arctanh(r13) + ( (norm.ppf( (1 - confidence_level) / 2)) * Standard_Error_Correlation))

        # Zou, 2007
        critical_value = abs(norm.ppf((1 - confidence_level) / 2)) * np.sqrt(1 / df)
        c = ((r23 - 0.5 * r12 * r13) * (1 - r12**2 - r13**2 - r23**2) + r23**3) / ((1 - r12**2) * (1 - r13**2))
        Lower_ci_zou = r12 - r13 - np.sqrt((r12 - (np.tanh(np.arctanh(r12) - critical_value)))**2 + ((np.tanh(np.arctanh(r13) + critical_value)) - r13)**2 - 2 * c * (r12 - (np.tanh(np.arctanh(r12) - critical_value))) * ((np.tanh(np.arctanh(r13) + critical_value)) - r13))
        Upper_ci_zou = r12 - r13 + np.sqrt((( np.tanh(np.arctanh(r12) + critical_value)) - r12)**2 + (r13 - (np.tanh(np.arctanh(r13) - critical_value)))**2 - 2 * c * (( np.tanh(np.arctanh(r12) + critical_value)) - r12) * (r13 - (np.tanh(np.arctanh(r13) - critical_value))))

        results= {}

        results["Fisher Zr_1"]= np.arctanh(r12)
        results["Fisher Zr_2"] = np.arctanh(r13)
        results["Difference_Between Correlations"]= Correlation_Difference
        results["Cohens q"]= Cohens_q
        results["Standrd Error Fisher"] = Standard_Error_Correlation

        results["Fisher Zr_1"]= np.arctanh(r12)
        results["Fisher Zr_2"] = np.arctanh(r13)
        results["Difference_Between Correlations"]= Correlation_Difference
        results["Cohens q"]= Cohens_q
        results["Standrd Error Fisher"] = Standard_Error_Correlation
        results["Confidence Intervals Zou"] = f"({round(Lower_ci_zou, 4)}, {round(Upper_ci_zou, 4)})"

        # Single r's and their confidence intervals
        results["r12"] = r12
        results["r12 Lower Confidence Interval"] = lower_ci_r12
        results["r12 Upper Confidence Interval"] = upper_ci_r12
        results["r12 t-score"] = t_score_r12
        results["r12 p-value"] = p_value_r12
        results["r13"] = r13
        results["r13 Lower Confidence Interval"] = lower_ci_r13
        results["r13 Upper Confidence Interval"] = upper_ci_r13
        results["r13 t-score"] = t_score_r13
        results["r13 p-value"] = p_value_r13

        # Additional statistics and their p-values
        results["Pearson Statistic"] = statistic_pearson
        results["Pearson p-value"] = p_value_pearson
        results["Hotelling Statistic"] = statistic_hotelling
        results["Hotelling p-value"] = p_value_hotelling
        results["Williams Statistic"] = statistic_williams
        results["Williams p-value"] = p_value_williams
        results["Olkin Statistic"] = statistic_Olkin
        results["Olkin p-value"] = p_value_Olkin
        results["Dunn Statistic"] = statistic_Dunn
        results["Dunn p-value"] = p_value_Dunn
        results["Hendrickson Statistic"] = statistic_Hendrickson
        results["Hendrickson p-value"] = p_value_Hendrickson
        results["Steiger Statistic"] = statistic_steiger
        results["Steiger p-value"] = p_value_steiger
        results["Meng Statistic"] = statistic_Meng
        results["Meng p-value"] = p_value_Meng
        results["Hittner Statistic"] = statistic_Hittner
        results["Hittner p-value"] = p_value_Hittner

        results["Confidence Intervals Zou"] = f"({round(Lower_ci_zou, 4)}, {round(Upper_ci_zou, 4)})"
        results["Confidence Intervals Meng"] = f"({round(Lower_Ci_Meng, 4)}, {round(Upper_Ci_Meng, 4)})"

        formatted_p_value_r12 = "{:.3f}".format(p_value_r12).lstrip('0') if p_value_r12 >= 0.001 else "\033[3mp\033[0m < .001"
        formatted_p_value_r13 = "{:.3f}".format(p_value_r13).lstrip('0') if p_value_r13 >= 0.001 else "\033[3mp\033[0m < .001"
        formatted_p_value_difference = "{:.3f}".format(p_value_Hittner).lstrip('0') if p_value_Hittner >= 0.001 else "\033[3mp\033[0m < .001"

        results["Statistical Line r12"] = "\033[3mr12\033[0m({}) = {}{}, {}{}, {}{}% CI [{}{}, {}{}]".format((sample_size - 2), ('-' if np.round(r12,3) < 0 else ''), str(np.abs(np.round(r12,3))).lstrip('0').rstrip(''), '\033[3mp = \033[0m' if p_value_r12 >= 0.001 else '', formatted_p_value_r12, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(lower_ci_r12,3) < 0 else ''), str(np.abs(np.round(lower_ci_r12,3))).lstrip('0').rstrip(''), ('-' if np.round(upper_ci_r12,3) < 0 else ''), str(np.abs(np.round(upper_ci_r12,3))).lstrip('0').rstrip(''))
        results["Statistical Line r34"] = "\033[3mr34\033[0m({}) = {}{}, {}{}, {}{}% CI [{}{}, {}{}]".format((sample_size - 2), ('-' if np.round(r13,3) < 0 else ''), str(np.abs(np.round(r13,3))).lstrip('0').rstrip(''), '\033[3mp = \033[0m' if p_value_r13 >= 0.001 else '', formatted_p_value_r13, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(lower_ci_r13,3) < 0 else ''), str(np.abs(np.round(lower_ci_r13,3))).lstrip('0').rstrip(''), ('-' if np.round(upper_ci_r13,3) < 0 else ''), str(np.abs(np.round(upper_ci_r13,3))).lstrip('0').rstrip(''))
        results["Statistical Line difference"] = "\033[3mZ\033[0m = {}, {}{}, {}{}% CI [{}{}, {}{}]".format( round(statistic_Hittner,3), '\033[3mp = \033[0m' if p_value_Hittner >= 0.001 else '', formatted_p_value_difference, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(Lower_ci_zou,3) < 0 else ''), str(np.abs(np.round(Lower_ci_zou,3))).lstrip('0').rstrip(''), ('-' if np.round(Upper_ci_zou,3) < 0 else ''), str(np.abs(np.round(Upper_ci_zou,3))).lstrip('0').rstrip(''))

        return results






# Things to Consider
# 1. Add Calculation from data and add confidence intervals by the bootstrapp
    # For non overlapped reapeted measures correlation baguley is different then zou
        



