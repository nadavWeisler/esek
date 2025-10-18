###############################################
# Effect Size for One Sample Proportion #######
###############################################

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional
from scipy.stats import norm, beta, binom, t
from statsmodels.stats.proportion import proportion_confint
from scipy.optimize import newton
from ...utils import interfaces, res, utils, es


def blakersCI(x, n, conf_level=0.95, tol=0.00001):
    def acceptance_probability(x, n, p):
        probabilty1 = 1 - binom.cdf(x - 1, n, p)
        probabilty2 = binom.cdf(x, n, p)
        a1 = probabilty1 + binom.cdf(binom.ppf(probabilty1, n, p) - 1, n, p)
        a2 = probabilty2 + 1 - binom.cdf(binom.ppf(1 - probabilty2, n, p), n, p)
        return min(a1, a2)

    CI_lower_blaker = beta.ppf((1 - conf_level) / 2, x, n - x + 1)
    CI_upper_blaker = beta.ppf(1 - (1 - conf_level) / 2, x + 1, n - x)

    while x != 0 and acceptance_probability(x, n, CI_lower_blaker + tol) < (
        1 - conf_level
    ):
        CI_lower_blaker += tol
    while x != n and acceptance_probability(x, n, CI_upper_blaker - tol) < (
        1 - conf_level
    ):
        CI_upper_blaker -= tol

    ci = [max(CI_lower_blaker, 0), min(CI_upper_blaker, 1)]

    return ci


def calculate_midp(x, n, conf_level):
    def f_low(pi):
        return (
            0.5 * binom.pmf(x, n, pi) + binom.cdf(x - 1, n, pi) - (1 + conf_level) / 2
        )

    def f_up(pi):
        return (
            0.5 * binom.pmf(x, n, pi) + binom.cdf(x - 1, n, pi) - (1 - conf_level) / 2
        )

    CI_lower_midp = newton(f_low, x / n)
    CI_upper_midp = newton(f_up, x / n)
    return CI_lower_midp, CI_upper_midp


@dataclass
class OneSampleProportionResults:
    descriptive_statistics : Optional[res.DescriptiveStatistics] = None
    z_test: Optional[res.InferentialStatistics] = None
    z_test_corrected: Optional[res.InferentialStatistics] = None
    z_test_wald: Optional[res.InferentialStatistics] = None
    z_test_wald_corrected: Optional[res.InferentialStatistics] = None
    cohens_g : Optional[es.CohenG] = None
    cohens_h : Optional[es.CohenH] = None
    wald_type_confidence_interval : Optional[res.ConfidenceInterval] = None
    wald_type_corrected_confidence_interval : Optional[res.ConfidenceInterval] = None
    wilson_confidence_interval : Optional[res.ConfidenceInterval] = None
    wilson_corrected_confidence_interval : Optional[res.ConfidenceInterval] = None
    logit_confidence_interval : Optional[res.ConfidenceInterval] = None
    jeffreys_confidence_interval : Optional[res.ConfidenceInterval] = None
    clopper_pearson_confidence_interval : Optional[res.ConfidenceInterval] = None
    arcsine_confidence_interval : Optional[res.ConfidenceInterval] = None
    pratt_confidence_interval : Optional[res.ConfidenceInterval] = None
    blaker_confidence_interval : Optional[res.ConfidenceInterval] = None
    midp_confidence_interval : Optional[res.ConfidenceInterval] = None
    agresti_coull_confidence_interval : Optional[res.ConfidenceInterval] = None

class OneSampleProportions(interfaces.AbstractTest):
    """
    A class to perform One Sample Proportion tests and calculate effect sizes.
    This class provides methods to calculate Cohen's g and Hedges' h
    based on the sample proportion, sample size, and confidence level.
    It also provides methods to calculate confidence intervals for the proportion.

    Attributes:
        None
    Methods:
        from_parameters -> OneSampleProportionResults:
            Calculates effect sizes and confidence intervals from sample parameters.
        from_data -> OneSampleProportionResults:
            Calculates effect sizes and confidence intervals from data columns.
        from_frequency -> OneSampleProportionResults:
            Calculates effect sizes and confidence intervals from frequency counts.
        proportion_of_hits -> dict:
            Calculates proportion of hits (pi) and its confidence interval.
    """

    @staticmethod
    def from_score() -> OneSampleProportionResults:
        """
        Calculates effect sizes and confidence intervals from a t-score.
        """
        raise NotImplementedError("This method is not implemented yet.")

    @staticmethod
    def from_parameters(
        proportion_sample: float,
        sample_size: int,
        population_proportion: float,
        confidence_level: float,
    ) -> OneSampleProportionResults:
        cohens_g = abs(proportion_sample - population_proportion)

        phi_sample = 2 * (np.arcsin(np.sqrt(proportion_sample)))
        phi_population = 2 * (np.arcsin(np.sqrt(population_proportion)))
        cohens_h = phi_sample - phi_population
        se_wald = np.sqrt((proportion_sample * (1 - proportion_sample)) / sample_size)
        se_score = np.sqrt(
            (population_proportion * (1 - population_proportion)) / sample_size
        )
        z_score_wald = (proportion_sample - population_proportion) / se_wald

        number_of_succeses_sample = proportion_sample * sample_size
        number_of_failures_sample = sample_size - number_of_succeses_sample
        number_of_succeses_population = population_proportion * sample_size
        number_of_failures_population = sample_size - number_of_succeses_population

        z_score = (proportion_sample - population_proportion) / se_score
        correction = sample_size * population_proportion + 0.5
        z_score_wald_corrected = (
            (proportion_sample * sample_size) - correction
        ) / np.sqrt((proportion_sample * (1 - proportion_sample) * sample_size))
        z_score_corrected = ((proportion_sample * sample_size) - correction) / np.sqrt(
            (population_proportion * (1 - population_proportion) * sample_size)
        )

        p_value_wald = norm.sf(abs(z_score_wald))
        p_value_wald_corrected = norm.sf(abs(z_score_wald_corrected))
        p_value_score = norm.sf(abs(z_score))
        p_value_score_corrected = norm.sf(abs(z_score_corrected))

        # Risk Measures (No CI's for these measures)
        relative_risk = proportion_sample / population_proportion
        odds_ratio = (proportion_sample / (1 - proportion_sample)) / (
            population_proportion / (1 - population_proportion)
        )
        risk_difference = proportion_sample - population_proportion

        # Confidence Intervals for Cohen's h (* I need to verify these CI's)
        z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
        se_arcsine = 2 * np.sqrt(0.25 * (1 / sample_size))
        lower_ci_cohens_h = cohens_h - z_critical_value * se_arcsine
        upper_ci_cohens_h = cohens_h + z_critical_value * se_arcsine

        # 1. Agresti-Coull
        ac_ci = proportion_confint(
            number_of_succeses_sample,
            sample_size,
            (1 - confidence_level),
            method="agresti_coull",
        )

        # 2. Wald CI's
        wald_ci = proportion_confint(
            number_of_succeses_sample,
            sample_size,
            (1 - confidence_level),
            method="normal",
        )

        # 3. Wald_Corrected
        correction = 0.05 / sample_size
        wald_corrected = np.array(wald_ci) + np.array([-correction, correction])

        # 4. Wilson
        wilson_ci = proportion_confint(
            number_of_succeses_sample,
            sample_size,
            (1 - confidence_level),
            method="wilson",
        )

        # 5. Wilson Corrected
        lower_ci_wilson_corrected = (
            2 * number_of_succeses_sample
            + z_critical_value**2
            - 1
            - z_critical_value
            * np.sqrt(
                z_critical_value**2
                - 2
                - 1 / sample_size
                + 4
                * (number_of_succeses_sample / sample_size)
                * (sample_size * (1 - number_of_succeses_sample / sample_size) + 1)
            )
        ) / (2 * (sample_size + z_critical_value**2))
        upper_ci_wilson_corrected = min(
            (
                2 * number_of_succeses_sample
                + z_critical_value**2
                + 1
                + z_critical_value
                * np.sqrt(
                    z_critical_value**2
                    + 2
                    - 1 / sample_size
                    + 4
                    * (number_of_succeses_sample / sample_size)
                    * (sample_size * (1 - number_of_succeses_sample / sample_size) - 1)
                )
            )
            / (2 * (sample_size + z_critical_value**2)),
            1,
        )

        # 6. Logit
        lambda_hat = math.log(
            number_of_succeses_sample / (sample_size - number_of_succeses_sample)
        )
        term1 = sample_size / (
            number_of_succeses_sample * (sample_size - number_of_succeses_sample)
        )
        lambda_low = lambda_hat - z_critical_value * np.sqrt(term1)
        lambda_upper = lambda_hat + z_critical_value * np.sqrt(term1)
        logit_lower = math.exp(lambda_low) / (1 + math.exp(lambda_low))
        logit_upper = min(math.exp(lambda_upper) / (1 + math.exp(lambda_upper)), 1)

        # 7. Jeffereys
        lower_jeffreys = beta.ppf(
            (1 - confidence_level) / 2,
            number_of_succeses_sample + 0.5,
            sample_size - number_of_succeses_sample + 0.5,
        )
        upper_jeffreys = min(
            beta.ppf(
                1 - (1 - confidence_level) / 2,
                number_of_succeses_sample + 0.5,
                sample_size - number_of_succeses_sample + 0.5,
            ),
            1,
        )

        # 8. Clopper-Pearson CI's
        lower_cp = beta.ppf(
            (1 - confidence_level) / 2,
            number_of_succeses_sample,
            sample_size - number_of_succeses_sample + 1,
        )
        upper_cp = max(
            beta.ppf(
                1 - (1 - confidence_level) / 2,
                number_of_succeses_sample + 1,
                sample_size - number_of_succeses_sample,
            ),
            1,
        )

        # 9. arcsine CI's 1 (Kulynskaya)
        p_tilde = (number_of_succeses_sample + 0.375) / (sample_size + 0.75)
        arcsine_lower = (
            math.sin(
                math.asin(np.sqrt(p_tilde))
                - 0.5 * z_critical_value / np.sqrt(sample_size)
            )
            ** 2
        )
        arcsine_upper = min(
            math.sin(
                math.asin(np.sqrt(p_tilde))
                + 0.5 * z_critical_value / np.sqrt(sample_size)
            )
            ** 2,
            1,
        )

        # 10. Pratt
        a = (
            (number_of_succeses_sample + 1) / (sample_size - number_of_succeses_sample)
        ) ** 2
        b = (
            81
            * (number_of_succeses_sample + 1)
            * (sample_size - number_of_succeses_sample)
            - 9 * sample_size
            - 8
        )
        c = (
            -3
            * z_critical_value
            * np.sqrt(
                9
                * (number_of_succeses_sample + 1)
                * (sample_size - number_of_succeses_sample)
                * (9 * sample_size + 5 - z_critical_value**2)
                + sample_size
                + 1
            )
        )
        d = (
            81 * (number_of_succeses_sample + 1) ** 2
            - 9 * (number_of_succeses_sample + 1) * (2 + z_critical_value**2)
            + 1
        )
        e = 1 + a * ((b + c) / d) ** 3
        a2 = (
            number_of_succeses_sample / (sample_size - number_of_succeses_sample - 1)
        ) ** 2
        b2 = (
            81
            * (number_of_succeses_sample)
            * (sample_size - number_of_succeses_sample - 1)
            - 9 * sample_size
            - 8
        )
        c2 = (
            3
            * z_critical_value
            * np.sqrt(
                9
                * number_of_succeses_sample
                * (sample_size - number_of_succeses_sample - 1)
                * (9 * sample_size + 5 - z_critical_value**2)
                + sample_size
                + 1
            )
        )
        d2 = (
            81 * number_of_succeses_sample**2
            - (9 * number_of_succeses_sample) * (2 + z_critical_value**2)
            + 1
        )
        e2 = 1 + a2 * ((b2 + c2) / d2) ** 3

        upper_pratt = min(1 / e, 1)
        lower_pratt = max(1 / e2, 0)

        # 11. Blaker

        ci_blakers = blakersCI(number_of_succeses_sample, sample_size, confidence_level)

        # 12. Mid-p

        midp_cis = calculate_midp(
            number_of_succeses_sample, sample_size, confidence_level
        )

        results = {}
        results["Sample's Proportion"] = proportion_sample
        results["Population's Proportion"] = population_proportion
        results["Sample Size"] = sample_size
        results["Confidence Level"] = confidence_level
        
        
        
        results["Number of Successes (Sample)"] = number_of_succeses_sample
        results["Number of Failures (Sample)"] = number_of_failures_sample
        results["Expected Number of Successes (Population)"] = (
            number_of_succeses_population
        )
        results["Expected Number of Failures (Population)"] = (
            number_of_failures_population
        )
    
        
        results["Standard Error (Wald)"] = round(se_wald, 4)
        results["Standard Error (Score)"] = round(se_score, 4)
        results["Z-score (Wald)"] = round(z_score_wald, 4)
        results["Z-score (Score)"] = round(z_score, 4)
        results["Z-score (Wald Corrected)"] = round(z_score_wald_corrected, 4)
        results["Z-score (Score Corrected)"] = round(z_score_corrected, 4)
        results["P-value (Wald)"] = np.round(p_value_wald, 4)
        results["P-value (Wald) Corrected"] = np.around(p_value_wald_corrected, 4)
        results["P-value (Score)"] = np.around(p_value_score, 4)
        results["P-value (Score) Corrected"] = np.around(p_value_score_corrected, 4)
        
        
        cohens_g_effect = es.CohenG(
            value=cohens_g,
            ci_lower=0.0,
            ci_upper=0.0,
            standard_error=0.0,
            name="Cohen's g",
        )
        
        results["Cohen's g"] = round(cohens_g, 4)
        results["cohen's h"] = round(cohens_h, 40)
        results["Relative Risk"] = round(relative_risk, 4)
        results["Odds Ratio"] = round(odds_ratio, 4)
        results["Cohen's h CI's"] = [lower_ci_Cohens_h, upper_ci_Cohens_h]
        results["Cohen's h Standard Error"] = [SE_arcsine]
        # Confidence Intervals for One Sample Proportion
        results["Agresti Coull CI's"] = np.round(np.array(AC_CI), 4)
        results["Wald CI"] = np.round(np.array(Wald_CI), 4)
        results["Wald CI Corrected"] = np.around(Wald_Corrected, 4)
        results["Wilson"] = np.around(np.array(wilson_CI), 4)
        results["Wilson Corrected"] = np.around(
            np.array([LowerCi_Wilson_Corrected, UpperCi_Wilson_Corrected]), 4
        )
        results["logit"] = np.around(np.array([logitlower, logitupper]), 4)
        results["Jeffereys"] = np.around(np.array([lowerjeffreys, upperjeffreys]), 4)
        results["Clopper-Pearson"] = np.around(np.array([lowerCP, upperCP]), 4)
        results["Arcsine"] = np.around(np.array([arcsinelower, arcsineupper]), 4)
        results["Pratt"] = np.around(np.array([lowerPratt, upperPratt]), 4)
        results["Blaker"] = np.around(np.array(ci_blakers), 4)
        results["Mid-p"] = np.around(np.array(midp_cis), 4)

        return results

    @staticmethod
    def from_data(
        columns: list,
        population_proportion: float,
        defined_sucess_value: float,
        confidence_level: float,
    ) -> OneSampleProportionResults:
        """
        Calculate One Sample Proportion results from data columns.
        Args:
            columns (list): List of data columns.
            population_proportion (float): Population proportion.
            defined_sucess_value (float): Value defined as success in the data.
            confidence_level (float): Confidence level for the intervals.

        Returns:
            OneSampleProportionResults: Results containing effect sizes and confidence intervals.
        """
        column_1 = columns[0]
        sample_size = len(column_1)
        number_of_successes = np.count_nonzero(column_1 == defined_sucess_value)
        proportion_sample = number_of_successes / sample_size
        return OneSampleProportions.from_parameters(
            proportion_sample,
            sample_size,
            population_proportion,
            confidence_level,
        )

    @staticmethod
    def from_frequency(
        population_proportion, number_of_successes, sample_size, confidence_level
    ) -> dict:
        proportion_sample = number_of_successes / sample_size
        return OneSampleProportions.from_parameters(
            proportion_sample,
            sample_size,
            population_proportion,
            confidence_level,
        )

    @staticmethod
    def proportion_of_hits(
        number_correct_answers: int,
        number_of_trials: int,
        number_of_choices: int,
        confidence_level: float,
    ) -> OneSampleProportionResults:

        proportion_correct = number_correct_answers / number_of_trials
        pi = (proportion_correct * (number_of_choices - 1)) / (
            1 + proportion_correct * (number_of_choices - 2)
        )
        se_pi = (1 / np.sqrt(number_of_trials)) * (
            (pi * (1 - pi)) / (np.sqrt(proportion_correct * (1 - proportion_correct)))
        )
        z_score = (pi - 0.5) / se_pi
        p_value = norm.sf(abs(z_score))
        zcrit = t.ppf(1 - (1 - confidence_level) / 2, 100000)
        lower_ci = pi - zcrit * se_pi
        upper_ci = pi + zcrit * se_pi

        results = {}

        results["Proportion of Hits [π]"] = round(pi, 7)
        results["Standard Error of π"] = round(se_pi, 7)
        results["Z-score"] = round(z_score, 7)
        results["p vlaue of π"] = np.around(np.array(p_value), 4)
        results["Confidence Intervals for π"] = (
            f"({round(lower_ci, 4)}, {round(upper_ci, 4)})"
        )

        return results
