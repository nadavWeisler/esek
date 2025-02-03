"""
This module contains functions and classes for performing one-sample t-tests and
calculating various statistics such as Cohen's d, Hedges' g, t-score, p-value,
confidence intervals, and standard errors.

The module includes the following functions:
- pivotal_ci_t: Calculate the Pivotal confidence intervals for a one-sample t-test.
- calculate_central_ci_one_sample_t_test: Calculate the central confidence intervals
for the effect size in a one-sample t-test.
- CI_NCP_one_Sample: Calculate the Non-Central Parameter (NCP) confidence intervals
for a one-sample t-test.

The module also includes the following class:
- One_Sample_ttest: A class containing static methods for performing one-sample
t-tests from t-score and parameters.
"""

import math
import numpy as np
from scipy.stats import norm, nct, t

def pivotal_ci_t(t_Score, df, sample_size, confidence_level):
    """
    Calculate the Pivotal confidence intervals for a one-sample t-test.

    Parameters
    ----------
    t_Score : float
        The t-score value.
    df : int
        Degrees of freedom.
    sample_size : int
        The size of the sample.
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%).

    Returns
    -------
    tuple
        A tuple containing:
        - lower_ci (float): Lower bound of the confidence interval.
        - upper_ci (float): Upper bound of the confidence interval.
    """
    is_negative = False
    if t_Score < 0:
        is_negative = True
        t_Score = abs(t_Score)
    upper_limit = 1 - (1 - confidence_level) / 2
    lower_limit = (1 - confidence_level) / 2

    lower_criterion = [-t_Score, t_Score / 2, t_Score]
    upper_criterion = [t_Score, 2 * t_Score, 3 * t_Score]

    while nct.cdf(t_Score, df, lower_criterion[0]) < upper_limit:
        lower_criterion = [
            lower_criterion[0] - t_Score,
            lower_criterion[0],
            lower_criterion[2],
        ]

    while nct.cdf(t_Score, df, upper_criterion[0]) < lower_limit:
        if nct.cdf(t_Score, df) < lower_limit:
            lower_ci = [0, nct.cdf(t_Score, df)]
            upper_criterion = [
                upper_criterion[0] / 4,
                upper_criterion[0],
                upper_criterion[2],
            ]

    while nct.cdf(t_Score, df, upper_criterion[2]) > lower_limit:
        upper_criterion = [
            upper_criterion[0],
            upper_criterion[2],
            upper_criterion[2] + t_Score,
        ]

    lower_ci = 0.0
    diff_lower = 1
    while diff_lower > 0.00001:
        if nct.cdf(t_Score, df, lower_criterion[1]) < upper_limit:
            lower_criterion = [
                lower_criterion[0],
                (lower_criterion[0] + lower_criterion[1]) / 2,
                lower_criterion[1],
            ]
        else:
            lower_criterion = [
                lower_criterion[1],
                (lower_criterion[1] + lower_criterion[2]) / 2,
                lower_criterion[2],
            ]
        diff_lower = abs(nct.cdf(t_Score, df, lower_criterion[1]) - upper_limit)
        lower_ci = lower_criterion[1] / (np.sqrt(sample_size))

    upper_ci = 0.0
    diff_upper = 1
    while diff_upper > 0.00001:
        if nct.cdf(t_Score, df, upper_criterion[1]) < lower_limit:
            upper_criterion = [
                upper_criterion[0],
                (upper_criterion[0] + upper_criterion[1]) / 2,
                upper_criterion[1],
            ]
        else:
            upper_criterion = [
                upper_criterion[1],
                (upper_criterion[1] + upper_criterion[2]) / 2,
                upper_criterion[2],
            ]
        diff_upper = abs(nct.cdf(t_Score, df, upper_criterion[1]) - lower_limit)
        upper_ci = upper_criterion[1] / (np.sqrt(sample_size))
    if is_negative:
        return -upper_ci, -lower_ci
    else:
        return lower_ci, upper_ci


def calculate_central_ci_one_sample_t_test(effect_size, sample_size, confidence_level):
    """
    Calculate the central confidence intervals for the effect size in a one-sample t-test.

    Parameters
    ----------
    effect_size : float
        The calculated effect size (Cohen's d).
    sample_size : int
        The size of the sample.
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%).

    Returns
    -------
    tuple
        A tuple containing:
        - ci_lower (float): Lower bound of the confidence interval.
        - ci_upper (float): Upper bound of the confidence interval.
        - Standard_error_effect_size_True (float): Standard error of the effect size (True).
        - Standard_error_effect_size_Morris (float): Standard error of the effect size (Morris).
        - Standard_error_effect_size_Hedges (float): Standard error of the effect size (Hedges).
        - Standard_error_effect_size_Hedges_Olkin (float): Standard error of the effect size (Hedges_Olkin).
        - Standard_error_effect_size_MLE (float): Standard error of the effect size (MLE).
        - Standard_error_effect_size_Large_N (float): Standard error of the effect size (Large N).
        - Standard_error_effect_size_Small_N (float): Standard error of the effect size (Small N).
    """
    df = sample_size - 1
    correction_factor = math.exp(
        math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2)
    )
    Standard_error_effect_size_True = np.sqrt(
        (
            (df / (df - 2)) * (1 / sample_size) * (1 + effect_size**2 * sample_size)
            - (effect_size**2 / correction_factor**2)
        )
    )
    Standard_error_effect_size_Morris = np.sqrt(
        (df / (df - 2)) * (1 / sample_size) * (1 + effect_size**2 * sample_size)
        - (effect_size**2 / (1 - (3 / (4 * (df - 1) - 1))) ** 2)
    )
    Standard_error_effect_size_Hedges = np.sqrt(
        (1 / sample_size) + effect_size**2 / (2 * df)
    )
    Standard_error_effect_size_Hedges_Olkin = np.sqrt(
        (1 / sample_size) + effect_size**2 / (2 * sample_size)
    )
    Standard_error_effect_size_MLE = np.sqrt(
        Standard_error_effect_size_Hedges * ((df + 2) / df)
    )
    Standard_error_effect_size_Large_N = np.sqrt(
        1 / sample_size * (1 + effect_size**2 / 8)
    )
    Standard_error_effect_size_Small_N = np.sqrt(
        Standard_error_effect_size_Large_N * ((df + 1) / (df - 1))
    )
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = (
        effect_size - Standard_error_effect_size_True * z_critical_value,
        effect_size + Standard_error_effect_size_True * z_critical_value,
    )
    return (
        ci_lower,
        ci_upper,
        Standard_error_effect_size_True,
        Standard_error_effect_size_Morris,
        Standard_error_effect_size_Hedges,
        Standard_error_effect_size_Hedges_Olkin,
        Standard_error_effect_size_MLE,
        Standard_error_effect_size_Large_N,
        Standard_error_effect_size_Small_N,
    )


def ci_ncp_one_sample(Effect_Size, sample_size, confidence_level):
    """
    Calculate the Non-Central Parameter (NCP) confidence intervals for a one-sample t-test.

    Parameters
    ----------
    Effect_Size : float
        The calculated effect size (Cohen's d).
    sample_size : int
        The size of the sample.
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%).

    Returns
    -------
    tuple
        A tuple containing:
        - CI_NCP_low (float): Lower bound of the NCP confidence interval.
        - CI_NCP_High (float): Upper bound of the NCP confidence interval.
    """
    NCP_value = Effect_Size * math.sqrt(sample_size)
    CI_NCP_low = (
        (
            nct.ppf(
                1 / 2 - confidence_level / 2,
                (sample_size - 1),
                loc=0,
                scale=1,
                nc=NCP_value,
            )
        )
        / NCP_value
        * Effect_Size
    )
    CI_NCP_High = (
        (
            nct.ppf(
                1 / 2 + confidence_level / 2,
                (sample_size - 1),
                loc=0,
                scale=1,
                nc=NCP_value,
            )
        )
        / NCP_value
        * Effect_Size
    )
    return CI_NCP_low, CI_NCP_High


class OneSampleT:
    """
    A class containing static methods for performing one-sample t-tests from t-score and parameters.

    This class includes the following static methods:
    - one_sample_from_t_score: Calculate the one-sample t-test results from a given t-score.
    - one_sample_from_params: Calculate the one-sample t-test results from given parameters.
    """

    @staticmethod
    def one_sample_from_t_score(params: dict) -> dict:
        """
        Calculate the one-sample t-test results from a given t-score.

        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - "t score" (float): The t-score value.
            - "Sample Size" (int): The size of the sample.
            - "Confidence Level" (float): The confidence level as a percentage (e.g., 95 for 95%).

        Returns
        -------
        dict
            A dictionary containing the calculated results:
            - "Cohen's d" (float): The calculated Cohen's d effect size.
            - "Hedges' g" (float): The calculated Hedges' g effect size.
            - "t score" (float): The t-score value.
            - "Degrees of Freedom" (float): The degrees of freedom.
            - "p-value" (float): The p-value.
            - "Standard Error of Cohen's d (True)" (float): Standard error of Cohen's d (True).
            - "Standard Error of Cohen's d (Morris)" (float): Standard error of Cohen's d (Morris).
            - "Standard Error of Cohen's d (Hedges)" (float): Standard error of Cohen's d (Hedges).
            - "Standard Error of Cohen's d (Hedges_Olkin)" (float): Standard error of Cohen's d (Hedges_Olkin).
            - "Standard Error of Cohen's d (MLE)" (float): Standard error of Cohen's d (MLE).
            - "Standard Error of Cohen's d (Large N)" (float): Standard error of Cohen's d (Large N).
            - "Standard Error of Cohen's d (Small N)" (float): Standard error of Cohen's d (Small N).
            - "Standard Error of Hedges' g (True)" (float): Standard error of Hedges' g (True).
            - "Standard Error of Hedges' g (Morris)" (float): Standard error of Hedges' g (Morris).
            - "Standard Error of Hedges' g (Hedges)" (float): Standard error of Hedges' g (Hedges).
            - "Standard Error of Hedges' g (Hedges_Olkin)" (float): Standard error of Hedges' g (Hedges_Olkin).
            - "Standard Error of Hedges' g (MLE)" (float): Standard error of Hedges' g (MLE).
            - "Standard Error of Hedges' g (Large N)" (float): Standard error of Hedges' g (Large N).
            - "Standard Error of Hedges' g (Small N)" (float): Standard error of Hedges' g (Small N).
            - "Lower Central CI's Cohen's d" (float): Lower bound of the central confidence interval for Cohen's d.
            - "Upper Central CI's Cohen's d" (float): Upper bound of the central confidence interval for Cohen's d.
            - "Lower Central CI's Hedges' g" (float): Lower bound of the central confidence interval for Hedges' g.
            - "Upper Central CI's Hedges' g" (float): Upper bound of the central confidence interval for Hedges' g.
            - "Lower Pivotal CI's Cohen's d" (float): Lower bound of the Pivotal confidence interval for Cohen's d.
            - "Upper Pivotal CI's Cohen's d" (float): Upper bound of the Pivotal confidence interval for Cohen's d.
            - "Lower Pivotal CI's Hedges' g" (float): Lower bound of the Pivotal confidence interval for Hedges' g.
            - "Upper Pivotal CI's Hedges' g" (float): Upper bound of the Pivotal confidence interval for Hedges' g.
            - "Lower NCP CI's Cohen's d" (float): Lower bound of the NCP confidence interval for Cohen's d.
            - "Upper NCP CI's Cohen's d" (float): Upper bound of the NCP confidence interval for Cohen's d.
            - "Lower NCP CI's Hedges' g" (float): Lower bound of the NCP confidence interval for Hedges' g.
            - "Upper NCP CI's Hedges' g" (float): Upper bound of the NCP confidence interval for Hedges' g.
            - "Correction Factor" (float): The correction factor.
            - "Statistical Line Cohen's d" (str): A formatted string with the statistical results for Cohen's d.
            - "Statistical Line Hedges' g" (str): A formatted string with the statistical results for Hedges' g.
        """
        # Get params
        t_score = params["t score"]
        sample_size = params["Sample Size"]
        confidence_level_percentage = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentage / 100
        df = sample_size - 1
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)
        cohens_d = t_score / np.sqrt(
            df
        )  # This is Cohen's d and it is calculated based on the sample's standard deviation
        correction = math.exp(
            math.lgamma(df / 2)
            - math.log(math.sqrt(df / 2))
            - math.lgamma((df - 1) / 2)
        )
        hedges_g = correction * cohens_d
        (
            ci_lower_cohens_d_central,
            ci_upper_cohens_d_central,
            Standard_error_cohens_d_true,
            Standard_error_cohens_d_morris,
            Standard_error_cohens_d_hedges,
            Standard_error_cohens_d_hedges_olkin,
            Standard_error_cohens_d_MLE,
            Standard_error_cohens_d_Largen,
            Standard_error_cohens_d_Small_n,
        ) = calculate_central_ci_one_sample_t_test(
            cohens_d, sample_size, confidence_level
        )
        (
            ci_lower_hedges_g_central,
            ci_upper_hedges_g_central,
            Standard_error_hedges_g_true,
            Standard_error_hedges_g_morris,
            Standard_error_hedges_g_hedges,
            Standard_error_hedges_g_hedges_olkin,
            Standard_error_hedges_g_MLE,
            Standard_error_hedges_g_Largen,
            Standard_error_hedges_g_Small_n,
        ) = calculate_central_ci_one_sample_t_test(
            hedges_g, sample_size, confidence_level
        )
        ci_lower_cohens_d_Pivotal, ci_upper_cohens_d_Pivotal = pivotal_ci_t(
            t_score, df, sample_size, confidence_level
        )
        ci_lower_hedges_g_Pivotal, ci_upper_hedges_g_Pivotal = pivotal_ci_t(
            t_score, df, sample_size, confidence_level
        )
        ci_lower_cohens_d_NCP, ci_upper_cohens_d_NCP = ci_ncp_one_sample(
            cohens_d, sample_size, confidence_level
        )
        ci_lower_hedges_g_NCP, ci_upper_hedges_g_NCP = ci_ncp_one_sample(
            hedges_g, sample_size, confidence_level
        )

        # Set results
        results = {}
        results["Cohen's d"] = round(cohens_d, 4)
        results["Hedges' g"] = round(hedges_g, 4)
        results["t score"] = round(t_score, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["p-value"] = round(p_value, 4)
        results["Standard Error of Cohen's d (True)"] = round(
            Standard_error_cohens_d_true, 4
        )
        results["Standard Error of Cohen's d (Morris)"] = round(
            Standard_error_cohens_d_morris, 4
        )
        results["Standard Error of Cohen's d (Hedges)"] = round(
            Standard_error_cohens_d_hedges, 4
        )
        results["Standard Error of Cohen's d (Hedges_Olkin)"] = round(
            Standard_error_cohens_d_hedges_olkin, 4
        )
        results["Standard Error of Cohen's d (MLE)"] = round(
            Standard_error_cohens_d_MLE, 4
        )
        results["Standard Error of Cohen's d (Large N)"] = round(
            Standard_error_cohens_d_Largen, 4
        )
        results["Standard Error of Cohen's d (Small N)"] = round(
            Standard_error_cohens_d_Small_n, 4
        )
        results["Standard Error of Hedges' g (True)"] = round(
            Standard_error_hedges_g_true, 4
        )
        results["Standard Error of Hedges' g (Morris)"] = round(
            Standard_error_hedges_g_morris, 4
        )
        results["Standard Error of Hedges' g (Hedges)"] = round(
            Standard_error_hedges_g_hedges, 4
        )
        results["Standard Error of Hedges' g (Hedges_Olkin)"] = round(
            Standard_error_hedges_g_hedges_olkin, 4
        )
        results["Standard Error of Hedges' g (MLE)"] = round(
            Standard_error_hedges_g_MLE, 4
        )
        results["Standard Error of Hedges' g (Large N)"] = round(
            Standard_error_hedges_g_Largen, 4
        )
        results["Standard Error of Hedges' g (Small N)"] = round(
            Standard_error_hedges_g_Small_n, 4
        )
        results["Lower Central CI's Cohen's d"] = round(ci_lower_cohens_d_central, 4)
        results["Upper Central CI's Cohen's d"] = round(ci_upper_cohens_d_central, 4)
        results["Lower Central CI's Hedges' g"] = round(ci_lower_hedges_g_central, 4)
        results["Upper Central CI's Hedges' g"] = round(ci_upper_hedges_g_central, 4)
        results["Lower Pivotal CI's Cohen's d"] = round(ci_lower_cohens_d_Pivotal, 4)
        results["Upper Pivotal CI's Cohen's d"] = round(ci_upper_cohens_d_Pivotal, 4)
        results["Lower Pivotal CI's Hedges' g"] = round(
            ci_lower_hedges_g_Pivotal * correction, 4
        )
        results["Upper Pivotal CI's Hedges' g"] = round(
            ci_upper_hedges_g_Pivotal * correction, 4
        )
        results["Lower NCP CI's Cohen's d"] = round(ci_lower_cohens_d_NCP, 4)
        results["Upper NCP CI's Cohen's d"] = round(ci_upper_cohens_d_NCP, 4)
        results["Lower NCP CI's Hedges' g"] = round(ci_lower_hedges_g_NCP, 4)
        results["Upper NCP CI's Hedges' g"] = round(ci_upper_hedges_g_NCP, 4)
        results["Correction Factor"] = round(correction, 4)
        formatted_p_value = (
            "{:.3f}".format(p_value).lstrip("0")
            if p_value >= 0.001
            else "\033[3mp\033[0m < .001"
        )
        results["Statistical Line Cohen's d"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's d = {:.3f}, {}% CI(Pivotal) [{:.3f},{:.3f}]".format(
                int(df),
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                cohens_d,
                confidence_level_percentage,
                round(ci_lower_cohens_d_Pivotal, 3),
                round(ci_upper_cohens_d_Pivotal, 3),
            )
        )
        results["Statistical Line Hedges' g"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Hedges' g = {:.3f}, {}% CI(Pivotal) [{:.3f},{:.3f}]".format(
                int(df),
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                hedges_g,
                confidence_level_percentage,
                round(ci_lower_hedges_g_Pivotal * correction, 3),
                round(ci_upper_hedges_g_Pivotal * correction, 3),
            )
        )

        return results

    @staticmethod
    def one_sample_from_params(params: dict) -> dict:
        """
        Calculate the one-sample t-test results from given parameters.

        Parameters
        ----------
        params : dict
            A dictionary containing the following keys:
            - "Population Mean" (float): The mean of the population.
            - "Mean Sample" (float): The mean of the sample.
            - "Standard Deviation Sample" (float): The standard deviation of the sample.
            - "Sample Size" (int): The size of the sample.
            - "Confidence Level" (float): The confidence level as a percentage (e.g., 95 for 95%).

        Returns
        -------
        dict
            A dictionary containing the calculated results:
            - "Cohen's d" (float): The calculated Cohen's d effect size.
            - "Hedges' g" (float): The calculated Hedges' g effect size.
            - "t score" (float): The t-score value.
            - "Degrees of Freedom" (float): The degrees of freedom.
            - "p-value" (float): The p-value.
            - "Standardizer Cohen's d (Sample's Standard Deviation)" (float): The sample's standard deviation.
            - "Standardizer Hedge's g" (float): The standardizer for Hedges' g.
            - "Standard Error of the Mean" (float): The standard error of the mean.
            - "Standard Error of Cohen's d (True)" (float): Standard error of Cohen's d (True).
            - "Standard Error of Cohen's d (Morris)" (float): Standard error of Cohen's d (Morris).
            - "Standard Error of Cohen's d (Hedges)" (float): Standard error of Cohen's d (Hedges).
            - "Standard Error of Cohen's d (Hedges_Olkin)" (float): Standard error of Cohen's d (Hedges_Olkin).
            - "Standard Error of Cohen's d (MLE)" (float): Standard error of Cohen's d (MLE).
            - "Standard Error of Cohen's d (Large N)" (float): Standard error of Cohen's d (Large N).
            - "Standard Error of Cohen's d (Small N)" (float): Standard error of Cohen's d (Small N).
            - "Standard Error of Hedges' g (True)" (float): Standard error of Hedges' g (True).
            - "Standard Error of Hedges' g (Morris)" (float): Standard error of Hedges' g (Morris).
            - "Standard Error of Hedges' g (Hedges)" (float): Standard error of Hedges' g (Hedges).
            - "Standard Error of Hedges' g (Hedges_Olkin)" (float): Standard error of Hedges' g (Hedges_Olkin).
            - "Standard Error of Hedges' g (MLE)" (float): Standard error of Hedges' g (MLE).
            - "Standard Error of Hedges' g (Large N)" (float): Standard error of Hedges' g (Large N).
            - "Standard Error of Hedges' g (Small N)" (float): Standard error of Hedges' g (Small N).
            - "Lower Central CI's Cohen's d" (float): Lower bound of the central confidence interval for Cohen's d.
            - "Upper Central CI's Cohen's d" (float): Upper bound of the central confidence interval for Cohen's d.
            - "Lower Central CI's Hedges' g" (float): Lower bound of the central confidence interval for Hedges' g.
            - "Upper Central CI's Hedges' g" (float): Upper bound of the central confidence interval for Hedges' g.
            - "Lower Pivotal CI's Cohen's d" (float): Lower bound of the Pivotal confidence interval for Cohen's d.
            - "Upper Pivotal CI's Cohen's d" (float): Upper bound of the Pivotal confidence interval for Cohen's d.
            - "Lower Pivotal CI's Hedges' g" (float): Lower bound of the Pivotal confidence interval for Hedges' g.
            - "Upper Pivotal CI's Hedges' g" (float): Upper bound of the Pivotal confidence interval for Hedges' g.
            - "Lower NCP CI's Cohen's d" (float): Lower bound of the NCP confidence interval for Cohen's d.
            - "Upper NCP CI's Cohen's d" (float): Upper bound of the NCP confidence interval for Cohen's d.
            - "Lower NCP CI's Hedges' g" (float): Lower bound of the NCP confidence interval for Hedges' g.
            - "Upper NCP CI's Hedges' g" (float): Upper bound of the NCP confidence interval for Hedges' g.
            - "Correction Factor" (float): The correction factor.
            - "Statistical Line Cohen's d" (str): A formatted string with the statistical results for Cohen's d.
            - "Statistical Line Hedges' g" (str): A formatted string with the statistical results for Hedges' g.
        """
        # Set params
        population_mean = params["Population Mean"]
        sample_mean = params["Mean Sample"]
        sample_sd = params["Standard Deviation Sample"]
        sample_size = params["Sample Size"]
        confidence_level_percentage = params["Confidence Level"]

        # Calculation
        confidence_level = confidence_level_percentage / 100
        df = sample_size - 1
        standard_error = sample_sd / np.sqrt(
            df
        )  # This is the standrt error of mean's estimate in o ne samaple t-test
        t_score = (
            sample_mean - population_mean
        ) / standard_error  # This is the t score in the test which is used to calculate the p-value
        cohens_d = (
            sample_mean - population_mean
        ) / sample_sd  # This is the effect size for one sample t-test Cohen's d
        correction = math.exp(
            math.lgamma(df / 2)
            - math.log(math.sqrt(df / 2))
            - math.lgamma((df - 1) / 2)
        )
        hedges_g = cohens_d * correction  # This is the actual corrected effect size
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)
        (
            ci_lower_cohens_d_central,
            ci_upper_cohens_d_central,
            Standard_error_cohens_d_true,
            Standard_error_cohens_d_morris,
            Standard_error_cohens_d_hedges,
            Standard_error_cohens_d_hedges_olkin,
            Standard_error_cohens_d_MLE,
            Standard_error_cohens_d_Largen,
            Standard_error_cohens_d_Small_n,
        ) = calculate_central_ci_one_sample_t_test(
            cohens_d, sample_size, confidence_level
        )
        (
            ci_lower_hedges_g_central,
            ci_upper_hedges_g_central,
            Standard_error_hedges_g_true,
            Standard_error_hedges_g_morris,
            Standard_error_hedges_g_hedges,
            Standard_error_hedges_g_hedges_olkin,
            Standard_error_hedges_g_MLE,
            Standard_error_hedges_g_Largen,
            Standard_error_hedges_g_Small_n,
        ) = calculate_central_ci_one_sample_t_test(
            hedges_g, sample_size, confidence_level
        )
        ci_lower_cohens_d_Pivotal, ci_upper_cohens_d_Pivotal = pivotal_ci_t(
            t_score, df, sample_size, confidence_level
        )
        ci_lower_hedges_g_Pivotal, ci_upper_hedges_g_Pivotal = pivotal_ci_t(
            t_score, df, sample_size, confidence_level
        )
        ci_lower_cohens_d_NCP, ci_upper_cohens_d_NCP = ci_ncp_one_sample(
            cohens_d, sample_size, confidence_level
        )
        ci_lower_hedges_g_NCP, ci_upper_hedges_g_NCP = ci_ncp_one_sample(
            hedges_g, sample_size, confidence_level
        )

        # Set results
        results = {}
        results["Cohen's d"] = round(cohens_d, 4)
        results["Hedges' g"] = round(hedges_g, 4)
        results["t score"] = round(t_score, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["p-value"] = round(p_value, 4)
        results["Standardizer Cohen's d (Sample's Standard Deviation)"] = round(
            sample_sd, 4
        )
        results["Standardizer Hedge's g"] = round(sample_sd / correction, 4)
        results["Standard Error of the Mean"] = round(standard_error, 4)
        results["Standard Error of Cohen's d (True)"] = round(
            Standard_error_cohens_d_true, 4
        )
        results["Standard Error of Cohen's d (Morris)"] = round(
            Standard_error_cohens_d_morris, 4
        )
        results["Standard Error of Cohen's d (Hedges)"] = round(
            Standard_error_cohens_d_hedges, 4
        )
        results["Standard Error of Cohen's d (Hedges_Olkin)"] = round(
            Standard_error_cohens_d_hedges_olkin, 4
        )
        results["Standard Error of Cohen's d (MLE)"] = round(
            Standard_error_cohens_d_MLE, 4
        )
        results["Standard Error of Cohen's d (Large N)"] = round(
            Standard_error_cohens_d_Largen, 4
        )
        results["Standard Error of Cohen's d (Small N)"] = round(
            Standard_error_cohens_d_Small_n, 4
        )
        results["Standard Error of Hedges' g (True)"] = round(
            Standard_error_hedges_g_true, 4
        )
        results["Standard Error of Hedges' g (Morris)"] = round(
            Standard_error_hedges_g_morris, 4
        )
        results["Standard Error of Hedges' g (Hedges)"] = round(
            Standard_error_hedges_g_hedges, 4
        )
        results["Standard Error of Hedges' g (Hedges_Olkin)"] = round(
            Standard_error_hedges_g_hedges_olkin, 4
        )
        results["Standard Error of Hedges' g (MLE)"] = round(
            Standard_error_hedges_g_MLE, 4
        )
        results["Standard Error of Hedges' g (Large N)"] = round(
            Standard_error_hedges_g_Largen, 4
        )
        results["Standard Error of Hedges' g (Small N)"] = round(
            Standard_error_hedges_g_Small_n, 4
        )
        results["Standard Error of the Mean"] = round(standard_error, 4)
        results["Standardizer Cohen's d (Sample's Standard Deviation)"] = round(
            sample_sd, 4
        )
        results["Standardizer Hedge's g"] = round(sample_sd / correction, 4)
        results["Sample's Mean"] = round(sample_mean, 4)
        results["Population's Mean"] = round(population_mean, 4)
        results["Means Difference"] = round(sample_mean - population_mean, 4)
        results["Sample Size"] = round(sample_size, 4)
        results["Sample's Standard Deviation"] = round(sample_sd, 4)
        results["Lower Central CI's Cohen's d"] = round(ci_lower_cohens_d_central, 4)
        results["Upper Central CI's Cohen's d"] = round(ci_upper_cohens_d_central, 4)
        results["Lower NCP CI's Cohen's d"] = round(ci_lower_cohens_d_NCP, 4)
        results["Upper NCP CI's Cohen's d"] = round(ci_upper_cohens_d_NCP, 4)

        results["Lower Pivotal CI's Cohen's d"] = round(ci_lower_cohens_d_Pivotal, 4)
        results["Upper Pivotal CI's Cohen's d"] = round(ci_upper_cohens_d_Pivotal, 4)
        results["Lower Pivotal CI's Hedges' g"] = round(
            ci_lower_hedges_g_Pivotal * correction, 4
        )
        results["Upper Pivotal CI's Hedges' g"] = round(
            ci_upper_hedges_g_Pivotal * correction, 4
        )
        results["Lower Central CI's Hedges' g"] = round(ci_lower_hedges_g_central, 4)
        results["Upper Central CI's Hedges' g"] = round(ci_upper_hedges_g_central, 4)
        results["Lower NCP CI's Hedges' g"] = round(ci_lower_hedges_g_NCP, 4)
        results["Upper NCP CI's Hedges' g"] = round(ci_upper_hedges_g_NCP, 4)
        results["Correction Factor"] = round(correction, 4)
        formatted_p_value = (
            "{:.3f}".format(p_value).lstrip("0")
            if p_value >= 0.001
            else "\033[3mp\033[0m < .001"
        )
        results["Statistical Line Cohen's d"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's d = {:.3f}, {}% CI(Pivotal) [{:.3f},{:.3f}]".format(
                int(df),
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                cohens_d,
                confidence_level_percentage,
                round(ci_lower_cohens_d_Pivotal, 3),
                round(ci_upper_cohens_d_Pivotal, 3),
            )
        )
        results["Statistical Line Hedges' g"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Hedges' g = {:.3f}, {}% CI(Pivotal) [{:.3f},{:.3f}]".format(
                int(df),
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                hedges_g,
                confidence_level_percentage,
                round(ci_lower_hedges_g_Pivotal * correction, 3),
                round(ci_upper_hedges_g_Pivotal * correction, 3),
            )
        )

        return results

    # Things to consider

    # 1. Using a different default for CI - maybe switch to the NCP's one
    # 2. imporve the Pivotal accuracy to match r functions...
    # 3. One Sample from Data
