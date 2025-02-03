"""
This module provides functions and classes to calculate effect sizes and statistics for paired samples t-tests.
It includes methods for calculating confidence intervals, standard errors, and various effect size measures.
"""

import numpy as np
import math
from scipy.stats import norm, nct, t
from scipy.stats import gmean
import rpy2.robjects as robjects

# robjects.r('install.packages("sadists")')
robjects.r("library(sadists)")
qlambdap = robjects.r["qlambdap"]


def pivotal_ci_t(t_Score, df, sample_size, confidence_level):
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


def nct_ci_t(t_Score, df, sample_size, confidence_level):
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
        lower_ci = lower_criterion[1]

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
        upper_ci = upper_criterion[1]
    if is_negative:
        return -upper_ci, -lower_ci
    else:
        return lower_ci, upper_ci


def calculate_central_ci_paired_samples_t_test(
    effect_size, sample_size, confidence_level
):
    """
    Calculate central confidence intervals for paired samples t-test.

    Parameters
    ----------
    effect_size : float
        Effect size value
    sample_size : int
        Sample size
    confidence_level : float
        Confidence level (between 0 and 1)

    Returns
    -------
    tuple
        Contains lower CI, upper CI, and various standard error estimates
    """
    df = sample_size - 1  # This is the Degrees of Freedom for one sample t-test
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


def calculate_se_pooled_paired_samples_t_test(
    effect_size, sample_size, correlation, confidence_level
):
    """
    Calculate standard errors for paired samples t-test with pooled design.

    Parameters
    ----------
    effect_size : float
        Effect size value
    sample_size : int
        Sample size
    correlation : float
        Correlation between paired measurements
    confidence_level : float
        Confidence level (between 0 and 1)

    Returns
    -------
    tuple
        Contains CI bounds and various standard error estimates
    """
    df = sample_size - 1  # This is the Degrees of Freedom for one sample t-test
    correction_factor = math.exp(
        math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2)
    )
    A = sample_size / (2 * (1 - correlation))
    Standard_error_effect_size_True = np.sqrt(
        (
            (df / (df - 2)) * (1 / A) * (1 + effect_size**2 * A)
            - (effect_size**2 / correction_factor**2)
        )
    )
    Standard_error_effect_size_Morris = np.sqrt(
        (df / (df - 2)) * (1 / A) * (1 + effect_size**2 * A)
        - (effect_size**2 / (1 - (3 / (4 * (df - 1) - 1))) ** 2)
    )
    Standard_error_effect_size_Hedges = np.sqrt((1 / A) + effect_size**2 / (2 * df))
    Standard_error_effect_size_Hedges_Olkin = np.sqrt(
        (1 / A) + effect_size**2 / (2 * sample_size)
    )
    Standard_error_effect_size_MLE = np.sqrt(
        Standard_error_effect_size_Hedges * ((df + 2) / df)
    )
    Standard_error_effect_size_Large_N = np.sqrt(1 / A * (1 + effect_size**2 / 8))
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


def ci_ncp_paired_samples_difference(Effect_Size, sample_size, confidence_level):
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


# 5. Adjusted Lambda Prime CI's
def ci_adjusted_lambda_prime_Paired_Samples(
    Effect_Size,
    Standard_Devisation_1,
    Standard_Deviation_2,
    sample_size,
    correlation,
    confidence_level,
):
    Corrected_Correlation = correlation * (
        gmean([Standard_Devisation_1**2, Standard_Deviation_2**2])
        / np.mean((Standard_Devisation_1**2, Standard_Deviation_2**2))
    )
    df = sample_size - 1
    df_corrected = 2 / (1 + correlation**2) * df
    correction1 = math.exp(
        math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2)
    )
    correction2 = math.exp(
        math.lgamma(df_corrected / 2)
        - math.log(math.sqrt(df_corrected / 2))
        - math.lgamma((df_corrected - 1) / 2)
    )
    Lambda = float(
        Effect_Size
        * correction1
        * np.sqrt(sample_size / (2 * (1 - Corrected_Correlation)))
    )
    Lower_CI_adjusted_lambda = qlambdap(1 / 2 - confidence_level / 2, df=(2 / (1 + correlation**2) * (sample_size - 1)), t=Lambda) / (2 * (1 - Corrected_Correlation)) / correction2  # type: ignore
    Upper_CI_adjusted_lambda = qlambdap(1 / 2 + confidence_level / 2, df=(2 / (1 + correlation**2) * (sample_size - 1)), t=Lambda) / (2 * (1 - Corrected_Correlation)) / correction2  # type: ignore
    return Lower_CI_adjusted_lambda, Upper_CI_adjusted_lambda


# 6. MAG CI's (Combination of Morris (2000), Algina and Kesselman 2003), and Goulet-Pelletier & Cousineau (2018))
def ci_mag_paired_samples(
    Effect_Size,
    Standard_Devisation_1,
    Standard_Deviation_2,
    sample_size,
    correlation,
    confidence_level,
):
    Corrected_Correlation = correlation * (
        gmean([Standard_Devisation_1**2, Standard_Deviation_2**2])
        / np.mean((Standard_Devisation_1**2, Standard_Deviation_2**2))
    )
    df = sample_size - 1
    correction = math.exp(
        math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2)
    )
    Lambda = float(
        Effect_Size
        * correction**2
        * np.sqrt(sample_size / (2 * (1 - Corrected_Correlation)))
    )
    Lower_CI_adjusted_MAG = nct.ppf(
        1 / 2 - confidence_level / 2, df=df, nc=Lambda
    ) / np.sqrt(sample_size / (2 * (1 - Corrected_Correlation)))
    Upper_CI_adjusted_MAG = nct.ppf(
        1 / 2 + confidence_level / 2, df=df, nc=Lambda
    ) / np.sqrt(sample_size / (2 * (1 - Corrected_Correlation)))
    return Lower_CI_adjusted_MAG, Upper_CI_adjusted_MAG


# 7. Morris (2000)
def ci_morris_paired_samples(Effect_Size, sample_size, correlation, confidence_level):
    df = sample_size - 1
    correction = math.exp(
        math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2)
    )
    Cohens_d_Variance_corrected = (
        (df / (df - 2))
        * 2
        * (1 - correlation)
        / sample_size
        * (1 + Effect_Size**2 * sample_size / (2 * (1 - correlation)))
        - Effect_Size**2 / correction**2
    ) * correction**2
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower_Morris, ci_upper_Morris = (
        Effect_Size - np.sqrt(Cohens_d_Variance_corrected) * z_critical_value,
        Effect_Size + np.sqrt(Cohens_d_Variance_corrected) * z_critical_value,
    )
    return ci_lower_Morris, ci_upper_Morris


# 8. t prime CI's - Goulet-Pelletier & Cousineau (2021)
def ci_t_prime_paired_samples(
    Effect_Size,
    Standard_Devisation_1,
    Standard_Deviation_2,
    sample_size,
    correlation,
    confidence_level,
):
    Corrected_Correlation = correlation * (
        gmean([Standard_Devisation_1**2, Standard_Deviation_2**2])
        / np.mean((Standard_Devisation_1**2, Standard_Deviation_2**2))
    )
    df = sample_size - 1
    df_corrected = 2 / (1 + correlation**2) * df
    correction = math.exp(
        math.lgamma(df_corrected / 2)
        - math.log(math.sqrt(df_corrected / 2))
        - math.lgamma((df_corrected - 1) / 2)
    )
    Lambda = float(
        Effect_Size
        * correction
        * np.sqrt(sample_size / (2 * (1 - Corrected_Correlation)))
    )
    Lower_CI_adjusted_lambda = qlambdap(1 / 2 - confidence_level / 2, df=(2 / (1 + correlation**2) * (sample_size - 1)), t=Lambda) / np.sqrt(sample_size / (2 * (1 - Corrected_Correlation)))  # type: ignore
    Upper_CI_adjusted_lambda = qlambdap(1 / 2 + confidence_level / 2, df=(2 / (1 + correlation**2) * (sample_size - 1)), t=Lambda) / np.sqrt(sample_size / (2 * (1 - Corrected_Correlation)))  # type: ignore
    return Lower_CI_adjusted_lambda, Upper_CI_adjusted_lambda


# 9. Non Central Parameter CI's (see Cousineau, 2018) ####Need to fix this
def ci_ncp_paired_samples_pooled(
    Effect_Size, sample_size, correlation, confidence_level
):
    NCP_value = Effect_Size * math.sqrt(sample_size / (2 * (1 - correlation)))
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


# 10. Algina Keselman CI's
def ci_t_algina_keselman(
    Effect_Size,
    Standard_Devisation_1,
    Standard_Deviation_2,
    sample_size,
    correlation,
    confidence_level,
):
    df = sample_size - 1
    Corrected_Correlation = correlation * (
        gmean([Standard_Devisation_1**2, Standard_Deviation_2**2])
        / np.mean((Standard_Devisation_1**2, Standard_Deviation_2**2))
    )
    Constant = np.sqrt(sample_size / (2 * (1 - Corrected_Correlation)))
    lower_CI_NCT, upper_CI_NCT = pivotal_ci_t(
        Effect_Size * Constant, df, sample_size, confidence_level
    )
    lower_CI_Algina_Keselman, upper_CI_Algina_Keselman = lower_CI_NCT / np.sqrt(
        sample_size / (2 * (1 - Corrected_Correlation))
    ), upper_CI_NCT / np.sqrt(sample_size / (2 * (1 - Corrected_Correlation)))
    return lower_CI_Algina_Keselman, upper_CI_Algina_Keselman


class PairedSamplesT:
    """
    A class to calculate effect sizes and statistics for paired samples t-test.

    Methods
    -------
    paired_samples_from_t_score(params: dict) -> dict
        Calculate effect sizes and statistics for paired samples t-test from t-score.

    paired_samples_from_parameters(params: dict) -> dict
        Calculate effect sizes and statistics for paired samples t-test from summary statistics.

    paired_samples_from_data(params: dict) -> dict
        Calculate effect sizes and statistics for paired samples t-test from raw data.
    """

    @staticmethod
    def paired_samples_from_t_score(params: dict) -> dict:
        """
        Calculate effect sizes and statistics for paired samples t-test from t-score.

        Parameters
        ----------
        params : dict
            Dictionary containing t-test parameters:
            - t-score
            - degrees of freedom
            - sample size
            - confidence level

        Returns
        -------
        dict
            Dictionary containing:
            - Cohen's d and Hedges' g effect sizes
            - Confidence intervals
            - Test statistics and p-values
            - Standard errors
        """
        # Set params
        t_score = params["t Score"]
        sample_size = params["Number of Pairs"]
        confidence_level_percentages = params["Confidence Level"]

        # Calculate
        confidence_level = confidence_level_percentages / 100
        df = int(sample_size - 1)
        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)
        cohens_dz = t_score / np.sqrt(
            sample_size
        )  # This is Cohens dz and it is calculated based on the sample's standard deviation of the difference
        correction = math.exp(
            math.lgamma(df / 2)
            - math.log(math.sqrt(df / 2))
            - math.lgamma((df - 1) / 2)
        )
        hedges_gz = correction * cohens_dz
        standardizer_cohens_dz = (2 * cohens_dz) / t_score
        standardizer_hedges_gz = standardizer_cohens_dz / correction

        (
            ci_lower_cohens_dz_central,
            ci_upper_cohens_dz_central,
            standard_error_cohens_dz_true,
            standard_error_cohens_dz_morris,
            standard_error_cohens_dz_hedges,
            standard_error_cohens_dz_hedges_olkin,
            standard_error_cohens_dz_MLE,
            standard_error_cohens_dz_Largen,
            standard_error_cohens_dz_Small_n,
        ) = calculate_central_ci_paired_samples_t_test(
            cohens_dz, sample_size, confidence_level
        )
        (
            ci_lower_hedges_gz_central,
            ci_upper_hedges_gz_central,
            standard_error_hedges_gz_true,
            standard_error_hedges_gz_morris,
            standard_error_hedges_gz_hedges,
            standard_error_hedges_gz_hedges_olkin,
            standard_error_hedges_gz_MLE,
            standard_error_hedges_gz_Largen,
            standard_error_hedges_gz_Small_n,
        ) = calculate_central_ci_paired_samples_t_test(
            hedges_gz, sample_size, confidence_level
        )

        ci_lower_cohens_dz_Pivotal, ci_upper_cohens_dz_Pivotal = pivotal_ci_t(
            t_score, df, sample_size, confidence_level
        )
        ci_lower_hedges_gz_Pivotal, ci_upper_hedges_gz_Pivotal = pivotal_ci_t(
            t_score, df, sample_size, confidence_level
        )

        # Set results
        results = {}
        results["Cohen's dz"] = round(cohens_dz, 4)
        results["Hedges gz"] = round(hedges_gz, 4)
        results["t-score"] = round(t_score, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["p-value"] = round(p_value, 4)
        results["Standardizer Cohen's dz"] = round(standardizer_cohens_dz, 4)
        results["Standardizer Hedges' gz"] = round(standardizer_hedges_gz, 4)
        results["Lower Central CI's Cohen's dz"] = round(ci_lower_cohens_dz_central, 4)
        results["Upper Central CI's Cohen's dz"] = round(ci_upper_cohens_dz_central, 4)
        results["Lower Central CI's Hedges' gz"] = round(ci_lower_hedges_gz_central, 4)
        results["Upper Central CI's Hedges' gz"] = round(ci_upper_hedges_gz_central, 4)
        results["Lower Pivotal CI's Cohen's dz"] = round(ci_lower_cohens_dz_Pivotal, 4)
        results["Upper Pivotal CI's Cohen's dz"] = round(ci_upper_cohens_dz_Pivotal, 4)
        results["Lower Pivotal CI's Hedges' gz"] = round(
            ci_lower_hedges_gz_Pivotal * correction, 4
        )
        results["Upper Pivotal CI's Hedges' gz"] = round(
            ci_upper_hedges_gz_Pivotal * correction, 4
        )

        # All type of Standard Errors
        results["Standard Error of Cohen's dz (True)"] = round(
            standard_error_cohens_dz_true, 4
        )
        results["Standard Error of Cohen's dz (Morris)"] = round(
            standard_error_cohens_dz_morris, 4
        )
        results["Standard Error of Cohen's dz (Hedges)"] = round(
            standard_error_cohens_dz_hedges, 4
        )
        results["Standard Error of Cohen's dz (Hedges_Olkin)"] = round(
            standard_error_cohens_dz_hedges_olkin, 4
        )
        results["Standard Error of Cohen's dz (MLE)"] = round(
            standard_error_cohens_dz_MLE, 4
        )
        results["Standard Error of Cohen's dz (Large N)"] = round(
            standard_error_cohens_dz_Largen, 4
        )
        results["Standard Error of Cohen's dz (Small N)"] = round(
            standard_error_cohens_dz_Small_n, 4
        )
        results["Standard Error of Hedges' gz (True)"] = round(
            standard_error_hedges_gz_true, 4
        )
        results["Standard Error of Hedges' gz (Morris)"] = round(
            standard_error_hedges_gz_morris, 4
        )
        results["Standard Error of Hedges' gz (Hedges)"] = round(
            standard_error_hedges_gz_hedges, 4
        )
        results["Standard Error of Hedges' gz (Hedges_Olkin)"] = round(
            standard_error_hedges_gz_hedges_olkin, 4
        )
        results["Standard Error of Hedges' gz (MLE)"] = round(
            standard_error_hedges_gz_MLE, 4
        )
        results["Standard Error of Hedges' gz (Large N)"] = round(
            standard_error_hedges_gz_Largen, 4
        )
        results["Standard Error of Hedges' gz (Small N)"] = round(
            standard_error_hedges_gz_Small_n, 4
        )

        results["Correction Factor"] = round(correction, 4)
        formatted_p_value = (
            "{:.3f}".format(p_value).lstrip("0")
            if p_value >= 0.001
            else "\033[3mp\033[0m < .001"
        )
        results["Statistical Line Cohen's dz"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's dz = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                df,
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                cohens_dz,
                confidence_level_percentages,
                ci_lower_cohens_dz_Pivotal,
                ci_upper_cohens_dz_Pivotal,
            )
        )
        results["Statistical Line Hedges' gz"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Hedges' gz = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                df,
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                hedges_gz,
                confidence_level_percentages,
                ci_lower_cohens_dz_Pivotal * correction,
                ci_upper_cohens_dz_Pivotal * correction,
            )
        )

        return results

    @staticmethod
    def paired_samples_from_parameters(params: dict) -> dict:
        """
        Calculate effect sizes and statistics for paired samples t-test from summary statistics.

        Parameters
        ----------
        params : dict
            Dictionary containing:
            - Sample Mean 1
            - Sample Mean 2
            - Standard Deviation Sample 1
            - Standard Deviation Sample 2
            - Number of Pairs
            - Difference in the Population
            - Pearson Correlation
            - Confidence Level

        Returns
        -------
        dict
            Dictionary containing:
            - Multiple effect size measures (Cohen's d variants, Hedges' g variants)
            - Confidence intervals using various methods
            - Test statistics and p-values
            - Standard errors and descriptive statistics
        """
        # Set params
        sample_mean_1 = params["Sample Mean 1"]
        sample_mean_2 = params["Sample Mean 2"]
        sample_sd_1 = params["Standard Deviation Sample 1"]
        sample_sd_2 = params["Standard Deviation Sample 2"]
        sample_size = params["Number of Pairs"]
        population_mean_diff = params[
            "Difference in the Population"
        ]  # The default value should be 0
        correlation = params[
            "Pearson Correlation"
        ]  # This one is crucial to calcualte the dz (otherwise return only dav)
        confidence_level_percentages = params["Confidence Level"]

        # Calculate
        confidence_level = confidence_level_percentages / 100
        df = int(sample_size - 1)
        standardizer_dz = np.sqrt(
            sample_sd_1**2
            + sample_sd_2**2
            - 2 * correlation * sample_sd_1 * sample_sd_2
        )
        standardizer_dav = np.sqrt((sample_sd_1**2 + sample_sd_2**2) / 2)
        standardizer_drm = np.sqrt(
            sample_sd_1**2
            + sample_sd_2**2
            - 2 * correlation * sample_sd_1 * sample_sd_2
        ) / np.sqrt(2 * (1 - correlation))
        standard_error = np.sqrt(
            ((sample_sd_1**2 / sample_size) + (sample_sd_2**2 / sample_size))
            - (
                2
                * correlation
                * (sample_sd_1 / np.sqrt(sample_size))
                * (sample_sd_2 / np.sqrt(sample_size))
            )
        )
        t_score = (
            (sample_mean_1 - sample_mean_2) - population_mean_diff
        ) / standard_error  # This is the t score in the test which is used to calculate the p-value
        t_score_av = ((sample_mean_1 - sample_mean_2) - population_mean_diff) / (
            (np.sqrt((sample_sd_1**2 + sample_sd_2**2) / 2)) / np.sqrt(sample_size)
        )
        t_score_rm = np.sqrt(
            ((sample_sd_1**2 / sample_size) + (sample_sd_2**2 / sample_size))
            - (
                2
                * correlation
                * (sample_sd_1 / np.sqrt(sample_size))
                * (sample_sd_2 / np.sqrt(sample_size))
            )
        ) * np.sqrt(2 * (1 - correlation))
        t_crit = t.ppf(1 - (1 - confidence_level) / 2, sample_size - 1, df)

        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)
        cohens_dz = (
            (sample_mean_1 - sample_mean_2) - population_mean_diff
        ) / standardizer_dz  # This is the effect size for one sample t-test Cohen's d
        correction = math.exp(
            math.lgamma(df / 2)
            - math.log(math.sqrt(df / 2))
            - math.lgamma((df - 1) / 2)
        )
        hedges_gz = cohens_dz * correction
        cohens_dav = (
            (sample_mean_1 - sample_mean_2) - population_mean_diff
        ) / standardizer_dav
        hedges_gav = cohens_dav * correction
        cohens_drm = (
            sample_mean_1 - sample_mean_2 - population_mean_diff
        ) / standardizer_drm
        hedges_grm = cohens_drm * correction
        standardizer_hedges_gz = standardizer_dz / correction
        standardizer_hedges_gav = standardizer_dav / correction
        standardizer_hedges_grm = standardizer_drm / correction

        # Central Confidence Intervals
        (
            ci_lower_cohens_dz_central,
            ci_upper_cohens_dz_central,
            standard_error_cohens_dz_true,
            standard_error_cohens_dz_morris,
            standard_error_cohens_dz_hedges,
            standard_error_cohens_dz_hedges_olkin,
            standard_error_cohens_dz_MLE,
            standard_error_cohens_dz_Largen,
            standard_error_cohens_dz_Small_n,
        ) = calculate_central_ci_paired_samples_t_test(
            cohens_dz, sample_size, confidence_level
        )
        (
            ci_lower_hedges_gz_central,
            ci_upper_hedges_gz_central,
            standard_error_hedges_gz_true,
            standard_error_hedges_gz_morris,
            standard_error_hedges_gz_hedges,
            standard_error_hedges_gz_hedges_olkin,
            standard_error_hedges_gz_MLE,
            standard_error_hedges_gz_Largen,
            standard_error_hedges_gz_Small_n,
        ) = calculate_central_ci_paired_samples_t_test(
            hedges_gz, sample_size, confidence_level
        )
        (
            ci_lower_cohens_dav_central,
            ci_upper_cohens_dav_central,
            standard_error_cohens_dav_true,
            standard_error_cohens_dav_morris,
            standard_error_cohens_dav_hedges,
            standard_error_cohens_dav_hedges_olkin,
            standard_error_cohens_dav_MLE,
            standard_error_cohens_dav_Largen,
            standard_error_cohens_dav_Small_n,
        ) = calculate_central_ci_paired_samples_t_test(
            cohens_dav, sample_size, confidence_level
        )
        (
            ci_lower_hedges_gav_central,
            ci_upper_hedges_gav_central,
            standard_error_hedges_gav_true,
            standard_error_hedges_gav_morris,
            standard_error_hedges_gav_hedges,
            standard_error_hedges_gav_hedges_olkin,
            standard_error_hedges_gav_MLE,
            standard_error_hedges_gav_Largen,
            standard_error_hedges_gav_Small_n,
        ) = calculate_central_ci_paired_samples_t_test(
            hedges_gav, sample_size, confidence_level
        )
        (
            ci_lower_cohens_drm_central,
            ci_upper_cohens_drm_central,
            standard_error_cohens_drm_true,
            standard_error_cohens_drm_morris,
            standard_error_cohens_drm_hedges,
            standard_error_cohens_drm_hedges_olkin,
            standard_error_cohens_drm_MLE,
            standard_error_cohens_drm_Largen,
            standard_error_cohens_drm_Small_n,
        ) = calculate_central_ci_paired_samples_t_test(
            cohens_drm, sample_size, confidence_level
        )
        (
            ci_lower_hedges_grm_central,
            ci_upper_hedges_grm_central,
            standard_error_hedges_grm_true,
            standard_error_hedges_grm_morris,
            standard_error_hedges_grm_hedges,
            standard_error_hedges_grm_hedges_olkin,
            standard_error_hedges_grm_MLE,
            standard_error_hedges_grm_Largen,
            standard_error_hedges_grm_Small_n,
        ) = calculate_central_ci_paired_samples_t_test(
            hedges_gz, sample_size, confidence_level
        )
        (
            ci_lower_cohens_dz_central_pooled,
            ci_upper_cohens_dz_central_pooled,
            standard_error_cohens_dz_true_pooled,
            standard_error_cohens_dz_morris_pooled,
            standard_error_cohens_dz_hedges_pooled,
            standard_error_cohens_dz_hedges_olkin_pooled,
            standard_error_cohens_dz_MLE_pooled,
            standard_error_cohens_dz_Largen_pooled,
            standard_error_cohens_dz_Small_n_pooled,
        ) = calculate_se_pooled_paired_samples_t_test(
            cohens_dz, sample_size, correlation, confidence_level
        )
        (
            ci_lower_hedges_gz_central_pooled,
            ci_upper_hedges_gz_central_pooled,
            standard_error_hedges_gz_true_pooled,
            standard_error_hedges_gz_morris_pooled,
            standard_error_hedges_gz_hedges_pooled,
            standard_error_hedges_gz_hedges_olkin_pooled,
            standard_error_hedges_gz_MLE_pooled,
            standard_error_hedges_gz_Largen_pooled,
            standard_error_hedges_gz_Small_n_pooled,
        ) = calculate_se_pooled_paired_samples_t_test(
            hedges_gz, sample_size, correlation, confidence_level
        )
        (
            ci_lower_cohens_dav_central_pooled,
            ci_upper_cohens_dav_central_pooled,
            standard_error_cohens_dav_true_pooled,
            standard_error_cohens_dav_morris_pooled,
            standard_error_cohens_dav_hedges_pooled,
            standard_error_cohens_dav_hedges_olkin_pooled,
            standard_error_cohens_dav_MLE_pooled,
            standard_error_cohens_dav_Largen_pooled,
            standard_error_cohens_dav_Small_n_pooled,
        ) = calculate_se_pooled_paired_samples_t_test(
            cohens_dav, sample_size, correlation, confidence_level
        )
        (
            ci_lower_hedges_gav_central_pooled,
            ci_upper_hedges_gav_central_pooled,
            standard_error_hedges_gav_true_pooled,
            standard_error_hedges_gav_morris_pooled,
            standard_error_hedges_gav_hedges_pooled,
            standard_error_hedges_gav_hedges_olkin_pooled,
            standard_error_hedges_gav_MLE_pooled,
            standard_error_hedges_gav_Largen_pooled,
            standard_error_hedges_gav_Small_n_pooled,
        ) = calculate_se_pooled_paired_samples_t_test(
            hedges_gav, sample_size, correlation, confidence_level
        )
        (
            ci_lower_cohens_drm_central_pooled,
            ci_upper_cohens_drm_central_pooled,
            standard_error_cohens_drm_true_pooled,
            standard_error_cohens_drm_morris_pooled,
            standard_error_cohens_drm_hedges_pooled,
            standard_error_cohens_drm_hedges_olkin_pooled,
            standard_error_cohens_drm_MLE_pooled,
            standard_error_cohens_drm_Largen_pooled,
            standard_error_cohens_drm_Small_n_pooled,
        ) = calculate_se_pooled_paired_samples_t_test(
            cohens_drm, sample_size, correlation, confidence_level
        )
        (
            ci_lower_hedges_grm_central_pooled,
            ci_upper_hedges_grm_central_pooled,
            standard_error_hedges_grm_true_pooled,
            standard_error_hedges_grm_morris_pooled,
            standard_error_hedges_grm_hedges_pooled,
            standard_error_hedges_grm_hedges_olkin_pooled,
            standard_error_hedges_grm_MLE_pooled,
            standard_error_hedges_grm_Largen_pooled,
            standard_error_hedges_grm_Small_n_pooled,
        ) = calculate_se_pooled_paired_samples_t_test(
            hedges_gz, sample_size, correlation, confidence_level
        )

        SE_becker = np.sqrt(
            ((2 * (1 - correlation)) / sample_size)
            + (cohens_drm**2 / (2 * (sample_size - 1)))
        )
        CI_lower_dunplap = cohens_dav - t_crit * SE_becker
        CI_upper_dunplap = cohens_dav + t_crit * SE_becker

        ci_lower_cohens_dz_Pivotal, ci_upper_cohens_dz_Pivotal = pivotal_ci_t(
            t_score, df, sample_size, confidence_level
        )
        ci_lower_cohens_dav_Pivotal, ci_upper_cohens_dav_Pivotal = pivotal_ci_t(
            t_score_av, df, sample_size, confidence_level
        )
        ci_lower_cohens_drm_Pivotal, ci_upper_cohens_drm_Pivotal = pivotal_ci_t(
            t_score_rm, df, sample_size, confidence_level
        )

        # Cohen's dav and Goulet-Pelletier and Cousineau CI's Alternatives # I can also add Algina and Kesselman 2003
        lower_ci_tprime_dav, upper_ci_tprime_dav = ci_t_prime_paired_samples(
            cohens_dav,
            sample_sd_1,
            sample_sd_2,
            sample_size,
            correlation,
            confidence_level,
        )
        lower_ci_tprime_gav, upper_ci_tprime_gav = (
            lower_ci_tprime_dav * correction,
            upper_ci_tprime_dav * correction,
        )
        lower_ci_lambda_prime_dav, upper_ci_lambda_prime_dav = (
            ci_adjusted_lambda_prime_Paired_Samples(
                cohens_dav,
                sample_sd_1,
                sample_sd_2,
                sample_size,
                correlation,
                confidence_level,
            )
        )
        lower_ci_lambda_prime_gav, upper_ci_lambda_prime_gav = (
            lower_ci_lambda_prime_dav * correction,
            upper_ci_lambda_prime_dav * correction,
        )
        lower_ci_MAG_dav, upper_ci_MAG_dav = ci_mag_paired_samples(
            cohens_dav,
            sample_sd_1,
            sample_sd_2,
            sample_size,
            correlation,
            confidence_level,
        )
        lower_ci_MAG_gav, upper_ci_MAG_gav = (
            lower_ci_MAG_dav * correction,
            upper_ci_MAG_dav * correction,
        )
        lower_ci_Morris_dav, upper_ci_Morris_dav = ci_morris_paired_samples(
            cohens_dav, sample_size, correlation, confidence_level
        )
        lower_ci_Morris_gav, upper_ci_Morris_gav = (
            lower_ci_Morris_dav * correction,
            upper_ci_Morris_dav * correction,
        )

        # Ratio of Means
        ratio_of_means = sample_mean_1 / sample_mean_2
        Varaince_of_means_ratio = (
            sample_sd_1**2 / (sample_mean_1**2)
            + sample_sd_2**2 / (sample_mean_2**2)
            - 2
            * correlation
            * np.sqrt(sample_sd_1**2 * sample_sd_2**2)
            / (sample_mean_1 * sample_mean_2)
        ) / sample_size
        Standard_Error_of_means_ratio = np.sqrt(Varaince_of_means_ratio)
        Degrees_of_freedom_means_ratio = sample_size - 1
        t_critical_value = t.ppf(
            confidence_level + ((1 - confidence_level) / 2),
            Degrees_of_freedom_means_ratio,
        )
        Lower_CI_Means_Ratio = math.exp(
            np.log(ratio_of_means) - t_critical_value * np.sqrt(Varaince_of_means_ratio)
        )
        Upper_CI_Means_Ratio = math.exp(
            np.log(ratio_of_means) + t_critical_value * np.sqrt(Varaince_of_means_ratio)
        )

        # Set results
        results = {}
        results["Cohens dz"] = round(cohens_dz, 4)
        results["Cohens drm"] = round(cohens_drm, 4)
        results["Cohens dav"] = round(cohens_dav, 4)
        results["Hedges gz"] = round(hedges_gz, 4)
        results["Hedges grm"] = round(hedges_grm, 4)
        results["Hedges gav"] = round(hedges_gav, 4)
        results["t-score"] = round(t_score, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["p-value"] = round(p_value, 4)
        results["Standard Error of the Mean Difference"] = round(standard_error, 4)
        results["Standardizer Cohen's dz"] = round(standardizer_dz, 4)
        results["Standardizer Hedges gz"] = round(standardizer_hedges_gz, 4)
        results["Standardizer Cohen's dav"] = round(standardizer_dav, 4)
        results["Standardizer Hedges gav"] = round(standardizer_hedges_gav, 4)
        results["Standardizer Cohen's drm"] = round(standardizer_drm, 4)
        results["Standardizer Hedges grm"] = round(standardizer_hedges_grm, 4)

        # All types of Standard Errors
        results["Standard Error of Cohen's d (True)"] = round(
            standard_error_cohens_dz_true, 4
        )
        results["Standard Error of Cohen's d (Morris)"] = round(
            standard_error_cohens_dz_morris, 4
        )
        results["Standard Error of Cohen's d (Hedges)"] = round(
            standard_error_cohens_dz_hedges, 4
        )
        results["Standard Error of Cohen's d (Hedges_Olkin)"] = round(
            standard_error_cohens_dz_hedges_olkin, 4
        )
        results["Standard Error of Cohen's d (MLE)"] = round(
            standard_error_cohens_dz_MLE, 4
        )
        results["Standard Error of Cohen's d (Large N)"] = round(
            standard_error_cohens_dz_Largen, 4
        )
        results["Standard Error of Cohen's d (Small N)"] = round(
            standard_error_cohens_dz_Small_n, 4
        )
        results["Standard Error of Hedges' g (True)"] = round(
            standard_error_hedges_gz_true, 4
        )
        results["Standard Error of Hedges' g (Morris)"] = round(
            standard_error_hedges_gz_morris, 4
        )
        results["Standard Error of Hedges' g (Hedges)"] = round(
            standard_error_hedges_gz_hedges, 4
        )
        results["Standard Error of Hedges' g (Hedges' Olkin)"] = round(
            standard_error_hedges_gz_hedges_olkin, 4
        )
        results["Standard Error of Hedges' g (MLE)"] = round(
            standard_error_hedges_gz_MLE, 4
        )
        results["Standard Error of Hedges' g (Large N)"] = round(
            standard_error_hedges_gz_Largen, 4
        )
        results["Standard Error of Hedges' g (Small N)"] = round(
            standard_error_hedges_gz_Small_n, 4
        )
        results["Standard Error of Cohen's dav (True)"] = round(
            standard_error_cohens_dav_true, 4
        )
        results["Standard Error of Cohen's dav (Morris)"] = round(
            standard_error_cohens_dav_morris, 4
        )
        results["Standard Error of Cohen's dav (Hedges)"] = round(
            standard_error_cohens_dav_hedges, 4
        )
        results["Standard Error of Cohen's dav (Hedges_Olkin)"] = round(
            standard_error_cohens_dav_hedges_olkin, 4
        )
        results["Standard Error of Cohen's dav (MLE)"] = round(
            standard_error_cohens_dav_MLE, 4
        )
        results["Standard Error of Cohen's dav (Large N)"] = round(
            standard_error_cohens_dav_Largen, 4
        )
        results["Standard Error of Cohen's dav (Small N)"] = round(
            standard_error_cohens_dav_Small_n, 4
        )
        results["Standard Error of Hedges' gav (True)"] = round(
            standard_error_hedges_gav_true, 4
        )
        results["Standard Error of Hedges' gav (Morris)"] = round(
            standard_error_hedges_gav_morris, 4
        )
        results["Standard Error of Hedges' gav (Hedges)"] = round(
            standard_error_hedges_gav_hedges, 4
        )
        results["Standard Error of Hedges' gav (Hedges_Olkin)"] = round(
            standard_error_hedges_gav_hedges_olkin, 4
        )
        results["Standard Error of Hedges' gav (MLE)"] = round(
            standard_error_hedges_grm_MLE, 4
        )
        results["Standard Error of Hedges' gav (Large N)"] = round(
            standard_error_hedges_grm_Largen, 4
        )
        results["Standard Error of Hedges' gav (Small N)"] = round(
            standard_error_hedges_grm_Small_n, 4
        )
        results["Standard Error of Cohen's drm (True)"] = round(
            standard_error_cohens_drm_true, 4
        )
        results["Standard Error of Cohen's drm (Morris)"] = round(
            standard_error_cohens_drm_morris, 4
        )
        results["Standard Error of Cohen's drm (Hedges)"] = round(
            standard_error_cohens_drm_hedges, 4
        )
        results["Standard Error of Cohen's drm (Hedges_Olkin)"] = round(
            standard_error_cohens_drm_hedges_olkin, 4
        )
        results["Standard Error of Cohen's drm (MLE)"] = round(
            standard_error_cohens_drm_MLE, 4
        )
        results["Standard Error of Cohen's drm (Large N)"] = round(
            standard_error_cohens_drm_Largen, 4
        )
        results["Standard Error of Cohen's drm (Small N)"] = round(
            standard_error_cohens_drm_Small_n, 4
        )
        results["Standard Error of Hedges' grm (True)"] = round(
            standard_error_hedges_grm_true, 4
        )
        results["Standard Error of Hedges' grm (Morris)"] = round(
            standard_error_hedges_grm_morris, 4
        )
        results["Standard Error of Hedges' grm (Hedges)"] = round(
            standard_error_hedges_grm_hedges, 4
        )
        results["Standard Error of Hedges' grm (Hedges_Olkin)"] = round(
            standard_error_hedges_grm_hedges_olkin, 4
        )
        results["Standard Error of Hedges' grm (MLE)"] = round(
            standard_error_hedges_grm_MLE, 4
        )
        results["Standard Error of Hedges' grm (Large N)"] = round(
            standard_error_hedges_grm_Largen, 4
        )
        results["Standard Error of Hedges' grm (Small N)"] = round(
            standard_error_hedges_grm_Small_n, 4
        )

        results["Pooled Standard Error of Cohen's ds (True)"] = round(
            standard_error_cohens_dz_true_pooled, 4
        )
        results["Pooled Standard Error of Cohen's ds (Morris)"] = round(
            standard_error_cohens_dz_morris_pooled, 4
        )
        results["Pooled Standard Error of Cohen's ds (Hedges)"] = round(
            standard_error_cohens_dz_hedges_pooled, 4
        )
        results["Pooled Standard Error of Cohen's ds (Hedges_Olkin)"] = round(
            standard_error_cohens_dz_hedges_olkin_pooled, 4
        )
        results["Pooled Standard Error of Cohen's ds (MLE)"] = round(
            standard_error_cohens_dz_MLE_pooled, 4
        )
        results["Pooled Standard Error of Cohen's ds (Large N)"] = round(
            standard_error_cohens_dz_Largen_pooled, 4
        )
        results["Pooled Standard Error of Cohen's ds (Small N)"] = round(
            standard_error_cohens_dz_Small_n_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gs (True)"] = round(
            standard_error_hedges_gz_true_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gs (Morris)"] = round(
            standard_error_hedges_gz_morris_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gs (Hedges)"] = round(
            standard_error_hedges_gz_hedges_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gs (Hedges' Olkin)"] = round(
            standard_error_hedges_gz_hedges_olkin_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gs (MLE)"] = round(
            standard_error_hedges_gz_MLE_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gs (Large N)"] = round(
            standard_error_hedges_gz_Largen_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gs (Small N)"] = round(
            standard_error_hedges_gz_Small_n_pooled, 4
        )
        results["Pooled Standard Error of Cohen's dav (True)"] = round(
            standard_error_cohens_dav_true_pooled, 4
        )
        results["Pooled Standard Error of Cohen's dav (Morris)"] = round(
            standard_error_cohens_dav_morris_pooled, 4
        )
        results["Pooled Standard Error of Cohen's dav (Hedges)"] = round(
            standard_error_cohens_dav_hedges_pooled, 4
        )
        results["Pooled Standard Error of Cohen's dav (Hedges_Olkin)"] = round(
            standard_error_cohens_dav_hedges_olkin_pooled, 4
        )
        results["Pooled Standard Error of Cohen's dav (MLE)"] = round(
            standard_error_cohens_dav_MLE_pooled, 4
        )
        results["Pooled Standard Error of Cohen's dav (Large N)"] = round(
            standard_error_cohens_dav_Largen_pooled, 4
        )
        results["Pooled Standard Error of Cohen's dav (Small N)"] = round(
            standard_error_cohens_dav_Small_n_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gav (True)"] = round(
            standard_error_hedges_gav_true_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gav (Morris)"] = round(
            standard_error_hedges_gav_morris_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gav (Hedges)"] = round(
            standard_error_hedges_gav_hedges_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gav (Hedges_Olkin)"] = round(
            standard_error_hedges_gav_hedges_olkin_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gav (MLE)"] = round(
            standard_error_hedges_grm_MLE_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gav (Large N)"] = round(
            standard_error_hedges_grm_Largen_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gav (Small N)"] = round(
            standard_error_hedges_grm_Small_n_pooled, 4
        )
        results["Pooled Standard Error of Cohen's drm (True)"] = round(
            standard_error_cohens_drm_true_pooled, 4
        )
        results["Pooled Standard Error of Cohen's drm (Morris)"] = round(
            standard_error_cohens_drm_morris_pooled, 4
        )
        results["Pooled Standard Error of Cohen's drm (Hedges)"] = round(
            standard_error_cohens_drm_hedges_pooled, 4
        )
        results["Pooled Standard Error of Cohen's drm (Hedges_Olkin)"] = round(
            standard_error_cohens_drm_hedges_olkin_pooled, 4
        )
        results["Pooled Standard Error of Cohen's drm (MLE)"] = round(
            standard_error_cohens_drm_MLE_pooled, 4
        )
        results["Pooled Standard Error of Cohen's drm (Large N)"] = round(
            standard_error_cohens_drm_Largen_pooled, 4
        )
        results["Pooled Standard Error of Cohen's drm (Small N)"] = round(
            standard_error_cohens_drm_Small_n_pooled, 4
        )
        results["Pooled Standard Error of Hedges' grm (True)"] = round(
            standard_error_hedges_grm_true_pooled, 4
        )
        results["Pooled Standard Error of Hedges' grm (Morris)"] = round(
            standard_error_hedges_grm_morris_pooled, 4
        )
        results["Pooled Standard Error of Hedges' grm (Hedges)"] = round(
            standard_error_hedges_grm_hedges_pooled, 4
        )
        results["Pooled Standard Error of Hedges' grm (Hedges_Olkin)"] = round(
            standard_error_hedges_grm_hedges_olkin_pooled, 4
        )
        results["Pooled Standard Error of Hedges' grm (MLE)"] = round(
            standard_error_hedges_grm_MLE_pooled, 4
        )
        results["Pooled Standard Error of Hedges' grm (Large N)"] = round(
            standard_error_hedges_grm_Largen_pooled, 4
        )
        results["Pooled Standard Error of Hedges' grm (Small N)"] = round(
            standard_error_hedges_grm_Small_n_pooled, 4
        )

        # Confidence Intervals
        results["Lower Central CI's Cohen's dz"] = round(ci_lower_cohens_dz_central, 4)
        results["Upper Central CI's Cohen's dz"] = round(ci_upper_cohens_dz_central, 4)
        results["Lower Central CI's Hedges' gz"] = round(ci_lower_hedges_gz_central, 4)
        results["Upper Central CI's Hedges' gz"] = round(ci_upper_hedges_gz_central, 4)
        results["Lower Central CI's Cohen's dav"] = round(
            ci_lower_cohens_dav_central, 4
        )
        results["Upper Central CI's Cohen's dav"] = round(
            ci_upper_cohens_dav_central, 4
        )
        results["Lower Central CI's Hedges' gav"] = round(
            ci_lower_hedges_gav_central, 4
        )
        results["Upper Central CI's Hedges' gav"] = round(
            ci_upper_hedges_gav_central, 4
        )
        results["Lower Central CI's Cohen's drm"] = round(
            ci_lower_cohens_drm_central * (np.sqrt(2 * (1 - correlation))) / correction,
            4,
        )
        results["Upper Central CI's Cohen's drm"] = round(
            ci_upper_cohens_drm_central * (np.sqrt(2 * (1 - correlation))) / correction,
            4,
        )
        results["Lower Central CI's Hedges' grm"] = round(
            ci_lower_hedges_grm_central * (np.sqrt(2 * (1 - correlation))), 4
        )
        results["Upper Central CI's Hedges' grm"] = round(
            ci_upper_hedges_grm_central * (np.sqrt(2 * (1 - correlation))), 4
        )
        results["Lower Pivotal CI's Cohen's dz"] = round(ci_lower_cohens_dz_Pivotal, 4)
        results["Upper Pivotal CI's Cohen's dz"] = round(ci_upper_cohens_dz_Pivotal, 4)
        results["Lower Pivotal CI's Hedges' gz"] = round(
            ci_lower_cohens_dz_Pivotal * correction, 4
        )
        results["Upper Pivotal CI's Hedges' gz"] = round(
            ci_upper_cohens_dz_Pivotal * correction, 4
        )
        results["Lower Pivotal CI's Cohen's dav"] = round(
            ci_lower_cohens_dav_Pivotal, 4
        )
        results["Upper Pivotal CI's Cohen's dav"] = round(
            ci_upper_cohens_dav_Pivotal, 4
        )
        results["Lower Pivotal CI's Hedges' gav"] = round(
            ci_lower_cohens_dav_Pivotal * correction, 4
        )
        results["Upper Pivotal CI's Hedges' gav"] = round(
            ci_upper_cohens_dav_Pivotal * correction, 4
        )
        results["Lower Pivotal CI's Cohen's drm"] = round(
            ci_lower_cohens_drm_Pivotal, 4
        )
        results["Upper Pivotal CI's Cohen's drm"] = round(
            ci_upper_cohens_drm_Pivotal, 4
        )
        results["Lower Pivotal CI's Hedges' grm"] = round(
            ci_lower_cohens_drm_Pivotal * correction, 4
        )
        results["Upper Pivotal CI's Hedges' grm"] = round(
            ci_upper_cohens_drm_Pivotal * correction, 4
        )
        results["Lower tprime CI's Cohen's dav"] = np.around(lower_ci_tprime_dav, 4)
        results["Upper tprime CI's Cohen's dav"] = np.around(upper_ci_tprime_dav, 4)
        results["Lower tprime CI's Hedges' gav"] = np.around(lower_ci_tprime_gav, 4)
        results["Upper tprime CI's Hedges' gav"] = np.around(upper_ci_tprime_gav, 4)
        results["Lower lambdaprime CI's Cohen's dav"] = np.around(
            lower_ci_lambda_prime_dav, 4
        )
        results["Upper lambdaprime CI's Cohen's dav"] = np.around(
            upper_ci_lambda_prime_dav, 4
        )
        results["Lower lambdaprime CI's Hedges' gav"] = np.around(
            lower_ci_lambda_prime_gav, 4
        )
        results["Upper lambdaprime CI's Hedges' gav"] = np.around(
            upper_ci_lambda_prime_gav, 4
        )
        results["Lower MAG CI's Cohen's dav"] = np.around(lower_ci_MAG_dav, 4)
        results["Upper MAG CI's Cohen's dav"] = np.around(upper_ci_MAG_dav, 4)
        results["Lower MAG CI's Hedges' gav"] = np.around(lower_ci_MAG_gav, 4)
        results["Upper MAG CI's Hedges' gav"] = np.around(upper_ci_MAG_gav, 4)
        results["Lower Morris CI's Cohen's dav"] = np.around(lower_ci_Morris_dav, 4)
        results["Upper Morris CI's Cohen's dav"] = np.around(upper_ci_Morris_dav, 4)
        results["Lower Morris CI's Hedges' gav"] = np.around(lower_ci_Morris_gav, 4)
        results["Upper Morris CI's Hedges' gav"] = np.around(upper_ci_Morris_gav, 4)
        results["Lower becker CI's Cohen's drm"] = np.around(CI_lower_dunplap, 4)
        results["Upper becker CI's Cohen's drm"] = np.around(CI_upper_dunplap, 4)
        results["Lower becker CI's Hedges' grm"] = np.around(
            CI_lower_dunplap * correction, 4
        )
        results["Upper becker CI's Hedges' grm"] = np.around(
            CI_upper_dunplap * correction, 4
        )

        # Ratio of dependent Means
        results["Ratio of Means"] = round(ratio_of_means, 4)
        results["Standard Error of Ratio of Means"] = round(
            Standard_Error_of_means_ratio, 4
        )
        results["Lower CI's Ratio of Means"] = round(Lower_CI_Means_Ratio, 4)
        results["Upper CI's Ratio of Means"] = round(Upper_CI_Means_Ratio, 4)

        # Mean Difference (Unstandardized)
        mean_difference = sample_mean_1 - sample_mean_2

        formatted_p_value = (
            "{:.3f}".format(p_value).lstrip("0")
            if p_value >= 0.001
            else "\033[3mp\033[0m < .001"
        )
        results["Statistical Line Cohen's dz"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's dz = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                df,
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                cohens_dz,
                confidence_level_percentages,
                ci_lower_cohens_dz_Pivotal,
                ci_upper_cohens_dz_Pivotal,
            )
        )
        results["Statistical Line Hedges' gz"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Hedges' gz = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                df,
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                hedges_gz,
                confidence_level_percentages,
                ci_lower_cohens_dz_Pivotal * correction,
                ci_upper_cohens_dz_Pivotal * correction,
            )
        )
        results["Statistical Line Cohen's drm"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's drm = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                df,
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                cohens_drm,
                confidence_level_percentages,
                ci_lower_cohens_drm_Pivotal,
                ci_upper_cohens_drm_Pivotal,
            )
        )
        results["Statistical Line Hedges' grm"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Hedges' grm = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                df,
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                hedges_grm,
                confidence_level_percentages,
                ci_lower_cohens_drm_Pivotal * correction,
                ci_upper_cohens_drm_Pivotal * correction,
            )
        )
        results["Statistical Line Cohen's dav"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's dav = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                df,
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                cohens_dav,
                confidence_level_percentages,
                ci_lower_cohens_dav_Pivotal,
                ci_upper_cohens_dav_Pivotal,
            )
        )
        results["Statistical Line Hedges' gav"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Hedges' gav = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                df,
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                hedges_gav,
                confidence_level_percentages,
                ci_lower_cohens_dav_Pivotal * correction,
                ci_upper_cohens_dav_Pivotal * correction,
            )
        )

        results["Correction Factor"] = round(correction, 4)

        return results

    @staticmethod
    def paired_samples_from_data(params: dict) -> dict:
        """
        Calculate effect sizes and statistics for paired samples t-test from raw data.

        Parameters
        ----------
        params : dict
            Dictionary containing:
            - Raw data for both samples
            - Confidence level
            - Expected population difference

        Returns
        -------
        dict
            Dictionary containing:
            - Multiple effect size measures (Cohen's d variants, Hedges' g variants)
            - Confidence intervals using various methods
            - Test statistics and p-values
            - Standard errors and descriptive statistics
        """
        # Get params
        column_1 = params["column_1"]
        column_2 = params["column_2"]
        population_mean_diff = params["Difference in the Population"]
        confidence_level_percentages = params["Confidence Level"]

        # Calculate
        confidence_level = confidence_level_percentages / 100
        sample_mean_1 = np.mean(column_1)
        sample_mean_2 = np.mean(column_2)
        sample_sd_1 = np.std(column_1, ddof=1)
        sample_sd_2 = np.std(column_2, ddof=1)
        sample_size = len(column_1)
        correlation_matrix = np.corrcoef(column_1, column_2)
        correlation = correlation_matrix[0, 1]
        p_value_correlation = correlation_matrix[1, 1]

        df = int(sample_size - 1)
        standardizer_dz = np.sqrt(
            sample_sd_1**2
            + sample_sd_2**2
            - 2 * correlation * sample_sd_1 * sample_sd_2
        )
        standardizer_dav = np.sqrt((sample_sd_1**2 + sample_sd_2**2) / 2)
        standardizer_drm = np.sqrt(
            sample_sd_1**2
            + sample_sd_2**2
            - 2 * correlation * sample_sd_1 * sample_sd_2
        ) / np.sqrt(2 * (1 - correlation))
        standard_error = np.sqrt(
            ((sample_sd_1**2 / sample_size) + (sample_sd_2**2 / sample_size))
            - (
                2
                * correlation
                * (sample_sd_1 / np.sqrt(sample_size))
                * (sample_sd_2 / np.sqrt(sample_size))
            )
        )
        t_score = (
            (sample_mean_1 - sample_mean_2) - population_mean_diff
        ) / standard_error  # This is the t score in the test which is used to calculate the p-value
        t_score_av = ((sample_mean_1 - sample_mean_2) - population_mean_diff) / (
            (np.sqrt((sample_sd_1**2 + sample_sd_2**2) / 2)) / np.sqrt(sample_size)
        )
        t_score_rm = ((sample_mean_1 - sample_mean_2) - population_mean_diff) / (
            standardizer_drm / np.sqrt(sample_size)
        )

        p_value = min(float(t.sf((abs(t_score)), df) * 2), 0.99999)
        cohens_dz = (
            (sample_mean_1 - sample_mean_2) - population_mean_diff
        ) / standardizer_dz  # This is the effect size for one sample t-test Cohen's d
        correction = math.exp(
            math.lgamma(df / 2)
            - math.log(math.sqrt(df / 2))
            - math.lgamma((df - 1) / 2)
        )
        hedges_gz = cohens_dz * correction
        cohens_dav = (
            (sample_mean_1 - sample_mean_2) - population_mean_diff
        ) / standardizer_dav
        hedges_gav = cohens_dav * correction
        cohens_drm = (
            sample_mean_1 - sample_mean_2 - population_mean_diff
        ) / standardizer_drm
        hedges_grm = cohens_drm * correction
        standardizer_hedges_gz = standardizer_dz / correction
        standardizer_hedges_gav = standardizer_dav / correction
        standardizer_hedges_grm = standardizer_drm / correction
        t_crit = t.ppf(1 - (1 - confidence_level) / 2, sample_size - 1, df)
        z_crit = norm.ppf(1 - (1 - confidence_level) / 2, sample_size - 1)

        SE_becker = np.sqrt(
            ((2 * (1 - correlation)) / sample_size)
            + (cohens_drm**2 / (2 * (sample_size - 1)))
        )
        CI_lower_dunplap = cohens_dav - t_crit * SE_becker
        CI_upper_dunplap = cohens_dav + t_crit * SE_becker

        (
            ci_lower_cohens_dz_central,
            ci_upper_cohens_dz_central,
            standard_error_cohens_dz_true,
            standard_error_cohens_dz_morris,
            standard_error_cohens_dz_hedges,
            standard_error_cohens_dz_hedges_olkin,
            standard_error_cohens_dz_MLE,
            standard_error_cohens_dz_Largen,
            standard_error_cohens_dz_Small_n,
        ) = calculate_central_ci_paired_samples_t_test(
            cohens_dz, sample_size, confidence_level
        )
        (
            ci_lower_hedges_gz_central,
            ci_upper_hedges_gz_central,
            standard_error_hedges_gz_true,
            standard_error_hedges_gz_morris,
            standard_error_hedges_gz_hedges,
            standard_error_hedges_gz_hedges_olkin,
            standard_error_hedges_gz_MLE,
            standard_error_hedges_gz_Largen,
            standard_error_hedges_gz_Small_n,
        ) = calculate_central_ci_paired_samples_t_test(
            hedges_gz, sample_size, confidence_level
        )
        (
            ci_lower_cohens_dav_central,
            ci_upper_cohens_dav_central,
            standard_error_cohens_dav_true,
            standard_error_cohens_dav_morris,
            standard_error_cohens_dav_hedges,
            standard_error_cohens_dav_hedges_olkin,
            standard_error_cohens_dav_MLE,
            standard_error_cohens_dav_Largen,
            standard_error_cohens_dav_Small_n,
        ) = calculate_central_ci_paired_samples_t_test(
            cohens_dav, sample_size, confidence_level
        )
        (
            ci_lower_hedges_gav_central,
            ci_upper_hedges_gav_central,
            standard_error_hedges_gav_true,
            standard_error_hedges_gav_morris,
            standard_error_hedges_gav_hedges,
            standard_error_hedges_gav_hedges_olkin,
            standard_error_hedges_gav_MLE,
            standard_error_hedges_gav_Largen,
            standard_error_hedges_gav_Small_n,
        ) = calculate_central_ci_paired_samples_t_test(
            hedges_gav, sample_size, confidence_level
        )
        (
            ci_lower_cohens_drm_central,
            ci_upper_cohens_drm_central,
            standard_error_cohens_drm_true,
            standard_error_cohens_drm_morris,
            standard_error_cohens_drm_hedges,
            standard_error_cohens_drm_hedges_olkin,
            standard_error_cohens_drm_MLE,
            standard_error_cohens_drm_Largen,
            standard_error_cohens_drm_Small_n,
        ) = calculate_central_ci_paired_samples_t_test(
            cohens_drm, sample_size, confidence_level
        )
        (
            ci_lower_hedges_grm_central,
            ci_upper_hedges_grm_central,
            standard_error_hedges_grm_true,
            standard_error_hedges_grm_morris,
            standard_error_hedges_grm_hedges,
            standard_error_hedges_grm_hedges_olkin,
            standard_error_hedges_grm_MLE,
            standard_error_hedges_grm_Largen,
            standard_error_hedges_grm_Small_n,
        ) = calculate_central_ci_paired_samples_t_test(
            hedges_gz, sample_size, confidence_level
        )
        ci_lower_cohens_dz_Pivotal, ci_upper_cohens_dz_Pivotal = pivotal_ci_t(
            t_score, df, sample_size, confidence_level
        )
        ci_lower_cohens_dav_Pivotal, ci_upper_cohens_dav_Pivotal = pivotal_ci_t(
            t_score_av, df, sample_size, confidence_level
        )
        ci_lower_cohens_drm_Pivotal, ci_upper_cohens_drm_Pivotal = pivotal_ci_t(
            t_score_rm, df, sample_size, confidence_level
        )
        (
            ci_lower_cohens_dz_central_pooled,
            ci_upper_cohens_dz_central_pooled,
            standard_error_cohens_dz_true_pooled,
            standard_error_cohens_dz_morris_pooled,
            standard_error_cohens_dz_hedges_pooled,
            standard_error_cohens_dz_hedges_olkin_pooled,
            standard_error_cohens_dz_MLE_pooled,
            standard_error_cohens_dz_Largen_pooled,
            standard_error_cohens_dz_Small_n_pooled,
        ) = calculate_se_pooled_paired_samples_t_test(
            cohens_dz, sample_size, correlation, confidence_level
        )
        (
            ci_lower_hedges_gz_central_pooled,
            ci_upper_hedges_gz_central_pooled,
            standard_error_hedges_gz_true_pooled,
            standard_error_hedges_gz_morris_pooled,
            standard_error_hedges_gz_hedges_pooled,
            standard_error_hedges_gz_hedges_olkin_pooled,
            standard_error_hedges_gz_MLE_pooled,
            standard_error_hedges_gz_Largen_pooled,
            standard_error_hedges_gz_Small_n_pooled,
        ) = calculate_se_pooled_paired_samples_t_test(
            hedges_gz, sample_size, correlation, confidence_level
        )
        (
            ci_lower_cohens_dav_central_pooled,
            ci_upper_cohens_dav_central_pooled,
            standard_error_cohens_dav_true_pooled,
            standard_error_cohens_dav_morris_pooled,
            standard_error_cohens_dav_hedges_pooled,
            standard_error_cohens_dav_hedges_olkin_pooled,
            standard_error_cohens_dav_MLE_pooled,
            standard_error_cohens_dav_Largen_pooled,
            standard_error_cohens_dav_Small_n_pooled,
        ) = calculate_se_pooled_paired_samples_t_test(
            cohens_dav, sample_size, correlation, confidence_level
        )
        (
            ci_lower_hedges_gav_central_pooled,
            ci_upper_hedges_gav_central_pooled,
            standard_error_hedges_gav_true_pooled,
            standard_error_hedges_gav_morris_pooled,
            standard_error_hedges_gav_hedges_pooled,
            standard_error_hedges_gav_hedges_olkin_pooled,
            standard_error_hedges_gav_MLE_pooled,
            standard_error_hedges_gav_Largen_pooled,
            standard_error_hedges_gav_Small_n_pooled,
        ) = calculate_se_pooled_paired_samples_t_test(
            hedges_gav, sample_size, correlation, confidence_level
        )
        (
            ci_lower_cohens_drm_central_pooled,
            ci_upper_cohens_drm_central_pooled,
            standard_error_cohens_drm_true_pooled,
            standard_error_cohens_drm_morris_pooled,
            standard_error_cohens_drm_hedges_pooled,
            standard_error_cohens_drm_hedges_olkin_pooled,
            standard_error_cohens_drm_MLE_pooled,
            standard_error_cohens_drm_Largen_pooled,
            standard_error_cohens_drm_Small_n_pooled,
        ) = calculate_se_pooled_paired_samples_t_test(
            cohens_drm, sample_size, correlation, confidence_level
        )
        (
            ci_lower_hedges_grm_central_pooled,
            ci_upper_hedges_grm_central_pooled,
            standard_error_hedges_grm_true_pooled,
            standard_error_hedges_grm_morris_pooled,
            standard_error_hedges_grm_hedges_pooled,
            standard_error_hedges_grm_hedges_olkin_pooled,
            standard_error_hedges_grm_MLE_pooled,
            standard_error_hedges_grm_Largen_pooled,
            standard_error_hedges_grm_Small_n_pooled,
        ) = calculate_se_pooled_paired_samples_t_test(
            hedges_gz, sample_size, correlation, confidence_level
        )
        birds_se = np.sqrt(
            (2 * (sample_sd_1**2 + sample_sd_2**2 - 2 * correlation))
            / (sample_size * (sample_sd_1**2 + sample_sd_2**2))
        )
        birds_ci = np.array(
            [cohens_drm - z_crit * birds_se, cohens_drm - z_crit * birds_se]
        )

        lower_ci_tprime_dav, upper_ci_tprime_dav = ci_t_prime_paired_samples(
            cohens_dav,
            sample_sd_1,
            sample_sd_2,
            sample_size,
            float(correlation),
            confidence_level,
        )
        lower_ci_tprime_gav, upper_ci_tprime_gav = (
            lower_ci_tprime_dav * correction,
            upper_ci_tprime_dav * correction,
        )
        lower_ci_lambda_prime_dav, upper_ci_lambda_prime_dav = (
            ci_adjusted_lambda_prime_Paired_Samples(
                cohens_dav,
                sample_sd_1,
                sample_sd_2,
                sample_size,
                float(correlation),
                confidence_level,
            )
        )
        lower_ci_lambda_prime_gav, upper_ci_lambda_prime_gav = (
            lower_ci_lambda_prime_dav * correction,
            upper_ci_lambda_prime_dav * correction,
        )
        lower_ci_MAG_dav, upper_ci_MAG_dav = ci_mag_paired_samples(
            cohens_dav,
            sample_sd_1,
            sample_sd_2,
            sample_size,
            correlation,
            confidence_level,
        )
        lower_ci_MAG_gav, upper_ci_MAG_gav = (
            lower_ci_MAG_dav * correction,
            upper_ci_MAG_dav * correction,
        )
        lower_ci_Morris_dav, upper_ci_Morris_dav = ci_morris_paired_samples(
            cohens_dav, sample_size, correlation, confidence_level
        )
        lower_ci_Morris_gav, upper_ci_Morris_gav = (
            lower_ci_Morris_dav * correction,
            upper_ci_Morris_dav * correction,
        )
        lower_ci_algina_keselman, upper_ci_algina_kesselman = ci_t_algina_keselman(
            cohens_dav,
            sample_sd_1,
            sample_sd_2,
            sample_size,
            correlation,
            confidence_level,
        )

        # Ratio of Means
        ratio_of_means = sample_mean_1 / sample_mean_2
        Varaince_of_means_ratio = (
            sample_sd_1**2 / (sample_mean_1**2)
            + sample_sd_2**2 / (sample_mean_2**2)
            - 2
            * correlation
            * np.sqrt(sample_sd_1**2 * sample_sd_2**2)
            / (sample_mean_1 * sample_mean_2)
        ) / sample_size
        Standard_Error_of_means_ratio = np.sqrt(Varaince_of_means_ratio)
        Degrees_of_freedom_means_ratio = sample_size - 1
        t_critical_value = t.ppf(
            confidence_level + ((1 - confidence_level) / 2),
            Degrees_of_freedom_means_ratio,
        )
        Lower_CI_Means_Ratio = math.exp(
            np.log(ratio_of_means) - t_critical_value * np.sqrt(Varaince_of_means_ratio)
        )
        Upper_CI_Means_Ratio = math.exp(
            np.log(ratio_of_means) + t_critical_value * np.sqrt(Varaince_of_means_ratio)
        )

        # Set results
        results = {}
        results["Cohens dz"] = round(
            cohens_dz, 4
        )  # a.k.a Dd (Cousineau, 2018) - The standardizer here is the SD of the Difference
        results["Cohens drm"] = round(
            cohens_drm, 4
        )  # a.k.a Ddc (Cousineau, 2018) - The Standardizer here is the SD of the difference corrected by the Correlation between measures
        results["Cohens dav"] = round(
            cohens_dav, 4
        )  # a.k.a as dp or ds since the samples are equal the classic cohens d can be used here as well- The Standardizer here is based on the Avarge between the two SD's
        results["Hedges' gz"] = round(hedges_gz, 4)
        results["Hedges' grm"] = round(hedges_grm, 4)
        results["Hedges' gav"] = round(hedges_gav, 4)
        results["t-score"] = round(t_score, 4)
        results["Degrees of Freedom"] = round(df, 4)
        results["p-value"] = round(p_value, 4)
        results["Standard Error of the Mean Difference"] = round(standard_error, 4)
        results["Standardizer Cohen's dz"] = round(standardizer_dz, 4)
        results["Standardizer Hedges' gz"] = round(standardizer_hedges_gz, 4)
        results["Standardizer Cohen's dav"] = round(standardizer_dav, 4)
        results["Standardizer Hedges' gav"] = round(standardizer_hedges_gav, 4)
        results["Standardizer Cohen's drm"] = round(standardizer_drm, 4)
        results["Standardizer Hedges' grm"] = round(standardizer_hedges_grm, 4)

        # All types of Standard Errors
        results["Standard Error of Cohen's ds (True)"] = round(
            standard_error_cohens_dz_true, 4
        )
        results["Standard Error of Cohen's ds (Morris)"] = round(
            standard_error_cohens_dz_morris, 4
        )
        results["Standard Error of Cohen's ds (Hedges)"] = round(
            standard_error_cohens_dz_hedges, 4
        )
        results["Standard Error of Cohen's ds (Hedges_Olkin)"] = round(
            standard_error_cohens_dz_hedges_olkin, 4
        )
        results["Standard Error of Cohen's ds (MLE)"] = round(
            standard_error_cohens_dz_MLE, 4
        )
        results["Standard Error of Cohen's ds (Large N)"] = round(
            standard_error_cohens_dz_Largen, 4
        )
        results["Standard Error of Cohen's ds (Small N)"] = round(
            standard_error_cohens_dz_Small_n, 4
        )
        results["Standard Error of Hedges' gs (True)"] = round(
            standard_error_hedges_gz_true, 4
        )
        results["Standard Error of Hedges' gs (Morris)"] = round(
            standard_error_hedges_gz_morris, 4
        )
        results["Standard Error of Hedges' gs (Hedges)"] = round(
            standard_error_hedges_gz_hedges, 4
        )
        results["Standard Error of Hedges' gs (Hedges' Olkin)"] = round(
            standard_error_hedges_gz_hedges_olkin, 4
        )
        results["Standard Error of Hedges' gs (MLE)"] = round(
            standard_error_hedges_gz_MLE, 4
        )
        results["Standard Error of Hedges' gs (Large N)"] = round(
            standard_error_hedges_gz_Largen, 4
        )
        results["Standard Error of Hedges' gs (Small N)"] = round(
            standard_error_hedges_gz_Small_n, 4
        )
        results["Standard Error of Cohen's dav (True)"] = round(
            standard_error_cohens_dav_true, 4
        )
        results["Standard Error of Cohen's dav (Morris)"] = round(
            standard_error_cohens_dav_morris, 4
        )
        results["Standard Error of Cohen's dav (Hedges)"] = round(
            standard_error_cohens_dav_hedges, 4
        )
        results["Standard Error of Cohen's dav (Hedges_Olkin)"] = round(
            standard_error_cohens_dav_hedges_olkin, 4
        )
        results["Standard Error of Cohen's dav (MLE)"] = round(
            standard_error_cohens_dav_MLE, 4
        )
        results["Standard Error of Cohen's dav (Large N)"] = round(
            standard_error_cohens_dav_Largen, 4
        )
        results["Standard Error of Cohen's dav (Small N)"] = round(
            standard_error_cohens_dav_Small_n, 4
        )
        results["Standard Error of Hedges' gav (True)"] = round(
            standard_error_hedges_gav_true, 4
        )
        results["Standard Error of Hedges' gav (Morris)"] = round(
            standard_error_hedges_gav_morris, 4
        )
        results["Standard Error of Hedges' gav (Hedges)"] = round(
            standard_error_hedges_gav_hedges, 4
        )
        results["Standard Error of Hedges' gav (Hedges_Olkin)"] = round(
            standard_error_hedges_gav_hedges_olkin, 4
        )
        results["Standard Error of Hedges' gav (MLE)"] = round(
            standard_error_hedges_grm_MLE, 4
        )
        results["Standard Error of Hedges' gav (Large N)"] = round(
            standard_error_hedges_grm_Largen, 4
        )
        results["Standard Error of Hedges' gav (Small N)"] = round(
            standard_error_hedges_grm_Small_n, 4
        )
        results["Standard Error of Cohen's drm (True)"] = round(
            standard_error_cohens_drm_true, 4
        )
        results["Standard Error of Cohen's drm (Morris)"] = round(
            standard_error_cohens_drm_morris, 4
        )
        results["Standard Error of Cohen's drm (Hedges)"] = round(
            standard_error_cohens_drm_hedges, 4
        )
        results["Standard Error of Cohen's drm (Hedges_Olkin)"] = round(
            standard_error_cohens_drm_hedges_olkin, 4
        )
        results["Standard Error of Cohen's drm (MLE)"] = round(
            standard_error_cohens_drm_MLE, 4
        )
        results["Standard Error of Cohen's drm (Large N)"] = round(
            standard_error_cohens_drm_Largen, 4
        )
        results["Standard Error of Cohen's drm (Small N)"] = round(
            standard_error_cohens_drm_Small_n, 4
        )
        results["Standard Error of Hedges' grm (True)"] = round(
            standard_error_hedges_grm_true, 4
        )
        results["Standard Error of Hedges' grm (Morris)"] = round(
            standard_error_hedges_grm_morris, 4
        )
        results["Standard Error of Hedges' grm (Hedges)"] = round(
            standard_error_hedges_grm_hedges, 4
        )
        results["Standard Error of Hedges' grm (Hedges_Olkin)"] = round(
            standard_error_hedges_grm_hedges_olkin, 4
        )
        results["Standard Error of Hedges' grm (MLE)"] = round(
            standard_error_hedges_grm_MLE, 4
        )
        results["Standard Error of Hedges' grm (Large N)"] = round(
            standard_error_hedges_grm_Largen, 4
        )
        results["Standard Error of Hedges' grm (Small N)"] = round(
            standard_error_hedges_grm_Small_n, 4
        )

        results["Pooled Standard Error of Cohen's d (True)"] = round(
            standard_error_cohens_dz_true_pooled, 4
        )
        results["Pooled Standard Error of Cohen's d (Morris)"] = round(
            standard_error_cohens_dz_morris_pooled, 4
        )
        results["Pooled Standard Error of Cohen's d (Hedges)"] = round(
            standard_error_cohens_dz_hedges_pooled, 4
        )
        results["Pooled Standard Error of Cohen's d (Hedges_Olkin)"] = round(
            standard_error_cohens_dz_hedges_olkin_pooled, 4
        )
        results["Pooled Standard Error of Cohen's d (MLE)"] = round(
            standard_error_cohens_dz_MLE_pooled, 4
        )
        results["Pooled Standard Error of Cohen's d (Large N)"] = round(
            standard_error_cohens_dz_Largen_pooled, 4
        )
        results["Pooled Standard Error of Cohen's d (Small N)"] = round(
            standard_error_cohens_dz_Small_n_pooled, 4
        )
        results["Pooled Standard Error of Hedges' g (True)"] = round(
            standard_error_hedges_gz_true_pooled, 4
        )
        results["Pooled Standard Error of Hedges' g (Morris)"] = round(
            standard_error_hedges_gz_morris_pooled, 4
        )
        results["Pooled Standard Error of Hedges' g (Hedges)"] = round(
            standard_error_hedges_gz_hedges_pooled, 4
        )
        results["Pooled Standard Error of Hedges' g (Hedges' Olkin)"] = round(
            standard_error_hedges_gz_hedges_olkin_pooled, 4
        )
        results["Pooled Standard Error of Hedges' g (MLE)"] = round(
            standard_error_hedges_gz_MLE_pooled, 4
        )
        results["Pooled Standard Error of Hedges' g (Large N)"] = round(
            standard_error_hedges_gz_Largen_pooled, 4
        )
        results["Pooled Standard Error of Hedges' g (Small N)"] = round(
            standard_error_hedges_gz_Small_n_pooled, 4
        )
        results["Pooled Standard Error of Cohen's dav (True)"] = round(
            standard_error_cohens_dav_true_pooled, 4
        )
        results["Pooled Standard Error of Cohen's dav (Morris)"] = round(
            standard_error_cohens_dav_morris_pooled, 4
        )
        results["Pooled Standard Error of Cohen's dav (Hedges)"] = round(
            standard_error_cohens_dav_hedges_pooled, 4
        )
        results["Pooled Standard Error of Cohen's dav (Hedges_Olkin)"] = round(
            standard_error_cohens_dav_hedges_olkin_pooled, 4
        )
        results["Pooled Standard Error of Cohen's dav (MLE)"] = round(
            standard_error_cohens_dav_MLE_pooled, 4
        )
        results["Pooled Standard Error of Cohen's dav (Large N)"] = round(
            standard_error_cohens_dav_Largen_pooled, 4
        )
        results["Pooled Standard Error of Cohen's dav (Small N)"] = round(
            standard_error_cohens_dav_Small_n_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gav (True)"] = round(
            standard_error_hedges_gav_true_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gav (Morris)"] = round(
            standard_error_hedges_gav_morris_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gav (Hedges)"] = round(
            standard_error_hedges_gav_hedges_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gav (Hedges_Olkin)"] = round(
            standard_error_hedges_gav_hedges_olkin_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gav (MLE)"] = round(
            standard_error_hedges_grm_MLE_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gav (Large N)"] = round(
            standard_error_hedges_grm_Largen_pooled, 4
        )
        results["Pooled Standard Error of Hedges' gav (Small N)"] = round(
            standard_error_hedges_grm_Small_n_pooled, 4
        )
        results["Pooled Standard Error of Cohen's drm (True)"] = round(
            standard_error_cohens_drm_true_pooled, 4
        )
        results["Pooled Standard Error of Cohen's drm (Morris)"] = round(
            standard_error_cohens_drm_morris_pooled, 4
        )
        results["Pooled Standard Error of Cohen's drm (Hedges)"] = round(
            standard_error_cohens_drm_hedges_pooled, 4
        )
        results["Pooled Standard Error of Cohen's drm (Hedges_Olkin)"] = round(
            standard_error_cohens_drm_hedges_olkin_pooled, 4
        )
        results["Pooled Standard Error of Cohen's drm (MLE)"] = round(
            standard_error_cohens_drm_MLE_pooled, 4
        )
        results["Pooled Standard Error of Cohen's drm (Large N)"] = round(
            standard_error_cohens_drm_Largen_pooled, 4
        )
        results["Pooled Standard Error of Cohen's drm (Small N)"] = round(
            standard_error_cohens_drm_Small_n_pooled, 4
        )
        results["Pooled Standard Error of Hedges' grm (True)"] = round(
            standard_error_hedges_grm_true_pooled, 4
        )
        results["Pooled Standard Error of Hedges' grm (Morris)"] = round(
            standard_error_hedges_grm_morris_pooled, 4
        )
        results["Pooled Standard Error of Hedges' grm (Hedges)"] = round(
            standard_error_hedges_grm_hedges_pooled, 4
        )
        results["Pooled Standard Error of Hedges' grm (Hedges_Olkin)"] = round(
            standard_error_hedges_grm_hedges_olkin_pooled, 4
        )
        results["Pooled Standard Error of Hedges' grm (MLE)"] = round(
            standard_error_hedges_grm_MLE_pooled, 4
        )
        results["Pooled Standard Error of Hedges' grm (Large N)"] = round(
            standard_error_hedges_grm_Largen_pooled, 4
        )
        results["Pooled Standard Error of Hedges' grm (Small N)"] = round(
            standard_error_hedges_grm_Small_n_pooled, 4
        )

        results["Sample Mean 1"] = round(sample_mean_1, 4)
        results["Sample Mean 2"] = round(sample_mean_2, 4)
        results["Difference Between Samples"] = round(sample_mean_1 - sample_mean_2, 4)
        results["Difference in the Population"] = round(population_mean_diff, 4)
        results["Number of Pairs (n)"] = sample_size
        results["Pearson Correlation of the two Measures"] = round(correlation, 4)
        results["Pearson Correlation p-value "] = round(p_value_correlation, 4)
        results["Total Sample Size"] = sample_size
        results["Sample Standard Deviation 1"] = round(sample_sd_1, 4)
        results["Sample Standard Deviation 2"] = round(sample_sd_2, 4)
        results["Lower Central CI's Cohen's dz"] = round(ci_lower_cohens_dz_central, 4)
        results["Upper Central CI's Cohen's dz"] = round(ci_upper_cohens_dz_central, 4)
        results["Lower Central CI's Hedges' gz"] = round(ci_lower_hedges_gz_central, 4)
        results["Upper Central CI's Hedges' gz"] = round(ci_upper_hedges_gz_central, 4)
        results["Lower Central CI's Cohen's dav"] = round(
            ci_lower_cohens_dav_central, 4
        )
        results["Upper Central CI's Cohen's dav"] = round(
            ci_upper_cohens_dav_central, 4
        )
        results["Lower Central CI's Hedges' gav"] = round(
            ci_lower_hedges_gav_central, 4
        )
        results["Upper Central CI's Hedges' gav"] = round(
            ci_upper_hedges_gav_central, 4
        )
        results["Lower Central CI's Cohen's drm"] = round(
            ci_lower_cohens_drm_central, 4
        )
        results["Upper Central CI's Cohen's drm"] = round(
            ci_upper_cohens_drm_central, 4
        )
        results["Lower Central CI's Hedges' grm"] = round(
            ci_lower_hedges_grm_central, 4
        )
        results["Upper Central CI's Hedges' grm"] = round(
            ci_upper_hedges_grm_central, 4
        )
        results["Lower Pivotal CI's Cohen's dz"] = round(ci_lower_cohens_dz_Pivotal, 4)
        results["Upper Pivotal CI's Cohen's dz"] = round(ci_upper_cohens_dz_Pivotal, 4)
        results["Lower Pivotal CI's Hedges' gz"] = round(
            ci_lower_cohens_dz_Pivotal * correction, 4
        )
        results["Upper Pivotal CI's Hedges' gz"] = round(
            ci_upper_cohens_dz_Pivotal * correction, 4
        )
        results["Lower Pivotal CI's Cohen's dav"] = round(
            ci_lower_cohens_dav_Pivotal, 4
        )
        results["Upper Pivotal CI's Cohen's dav"] = round(
            ci_upper_cohens_dav_Pivotal, 4
        )
        results["Lower Pivotal CI's Hedges' gav"] = round(
            ci_lower_cohens_dav_Pivotal * correction, 4
        )
        results["Upper Pivotal CI's Hedges' gav"] = round(
            ci_upper_cohens_dav_Pivotal * correction, 4
        )
        results["Lower Pivotal CI's Cohen's drm"] = round(
            ci_lower_cohens_drm_Pivotal * (np.sqrt(2 * (1 - correlation))) / correction,
            4,
        )
        results["Upper Pivotal CI's Cohen's drm"] = round(
            ci_upper_cohens_drm_Pivotal * (np.sqrt(2 * (1 - correlation))) / correction,
            4,
        )
        results["Lower Pivotal CI's Hedges' grm"] = round(
            ci_lower_cohens_drm_Pivotal * (np.sqrt(2 * (1 - correlation))), 4
        )
        results["Upper Pivotal CI's Hedges' grm"] = round(
            ci_upper_cohens_drm_Pivotal * (np.sqrt(2 * (1 - correlation))), 4
        )
        results["Lower tprime CI's Cohen's dav"] = np.around(lower_ci_tprime_dav, 4)
        results["Upper tprime CI's Cohen's dav"] = np.around(upper_ci_tprime_dav, 4)
        results["Lower tprime CI's Hedges' gav"] = np.around(lower_ci_tprime_gav, 4)
        results["Upper tprime CI's Hedges' gav"] = np.around(upper_ci_tprime_gav, 4)
        results["Lower lambdaprime CI's Cohen's dav"] = np.around(
            lower_ci_lambda_prime_dav, 4
        )
        results["Upper lambdaprime CI's Cohen's dav"] = np.around(
            upper_ci_lambda_prime_dav, 4
        )
        results["Lower lambdaprime CI's Hedges' gav"] = np.around(
            lower_ci_lambda_prime_gav, 4
        )
        results["Upper lambdaprime CI's Hedges' gav"] = np.around(
            upper_ci_lambda_prime_gav, 4
        )
        results["Lower MAG CI's Cohen's dav"] = np.around(lower_ci_MAG_dav, 4)
        results["Upper MAG CI's Cohen's dav"] = np.around(upper_ci_MAG_dav, 4)

        results["Lower Algina Keselman CI's Cohen's dav"] = np.around(
            lower_ci_algina_keselman, 4
        )
        results["Upper Algina Keselman CI's Cohen's dav"] = np.around(
            upper_ci_algina_kesselman, 4
        )
        results["Lower Algina Keselman CI's Cohen's gav"] = np.around(
            lower_ci_algina_keselman * correction, 4
        )
        results["Upper Algina Keselman CI's Cohen's gav"] = np.around(
            upper_ci_algina_kesselman * correction, 4
        )

        results["Lower MAG CI's Hedges' gav"] = np.around(lower_ci_MAG_gav, 4)
        results["Upper MAG CI's Hedges' gav"] = np.around(upper_ci_MAG_gav, 4)
        results["Lower Morris CI's Cohen's dav"] = np.around(lower_ci_Morris_dav, 4)
        results["Upper Morris CI's Cohen's dav"] = np.around(upper_ci_Morris_dav, 4)
        results["Lower Morris CI's Hedges' gav"] = np.around(lower_ci_Morris_gav, 4)
        results["Upper Morris CI's Hedges' gav"] = np.around(upper_ci_Morris_gav, 4)
        results["Lower becker's CI's Cohen's drm"] = np.around(CI_lower_dunplap, 4)
        results["Upper becker's CI's Cohen's drm"] = np.around(CI_upper_dunplap, 4)
        results["Lower becker's CI's Hedges' grm"] = np.around(
            CI_lower_dunplap * correction, 4
        )
        results["Upper becker's CI's Hedges' grm"] = np.around(
            CI_upper_dunplap * correction, 4
        )
        results["Lower bird's CI's Cohen's drm"] = np.around(birds_ci[0], 4)
        results["Upper bird's CI's Cohen's drm"] = np.around(birds_ci[1], 4)
        results["Lower bird's CI's Hedges' grm"] = np.around(
            birds_ci[0] * correction, 4
        )
        results["Upper bird's CI's Hedges' grm"] = np.around(
            birds_ci[1] * correction, 4
        )

        # Ratio of Means
        results["Ratio of Means"] = round(ratio_of_means, 4)
        results["Standard Error of Ratio of Means"] = round(
            Standard_Error_of_means_ratio, 4
        )
        results["Lower CI's Ratio of Means"] = round(Lower_CI_Means_Ratio, 4)
        results["Upper CI's Ratio of Means"] = round(Upper_CI_Means_Ratio, 4)

        formatted_p_value = (
            "{:.3f}".format(p_value).lstrip("0")
            if p_value >= 0.001
            else "\033[3mp\033[0m < .001"
        )
        results["Statistical Line Cohen's dz"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's dz = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                df,
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                cohens_dz,
                confidence_level_percentages,
                ci_lower_cohens_dz_Pivotal * correction,
                ci_upper_cohens_dz_Pivotal,
            )
        )
        results["Statistical Line Hedges' gz"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Hedges' gz = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                df,
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                hedges_gz,
                confidence_level_percentages,
                ci_lower_cohens_dz_Pivotal * correction * correction,
                ci_upper_cohens_dz_Pivotal * correction,
            )
        )
        results["Statistical Line Cohen's drm"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's drm = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                df,
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                cohens_drm,
                confidence_level_percentages,
                ci_lower_cohens_drm_Pivotal,
                ci_upper_cohens_drm_Pivotal,
            )
        )
        results["Statistical Line Hedges' grm"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Hedges' grm = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                df,
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                hedges_grm,
                confidence_level_percentages,
                ci_lower_cohens_drm_Pivotal * correction,
                ci_upper_cohens_drm_Pivotal * correction,
            )
        )
        results["Statistical Line Cohen's dav"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Cohen's dav = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                df,
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                cohens_dav,
                confidence_level_percentages,
                ci_lower_cohens_dav_Pivotal,
                ci_upper_cohens_dav_Pivotal,
            )
        )
        results["Statistical Line Hedges' gav"] = (
            " \033[3mt\033[0m({}) = {:.3f}, {}{}, Hedges' gav = {:.3f}, {}% CI(Pivotal) [{:.3f}, {:.3f}]".format(
                df,
                t_score,
                "\033[3mp = \033[0m" if p_value >= 0.001 else "",
                formatted_p_value,
                hedges_gav,
                confidence_level_percentages,
                ci_lower_cohens_dav_Pivotal * correction,
                ci_upper_cohens_dav_Pivotal * correction,
            )
        )
        results["Correction Factor"] = round(correction, 4)
        return results

    # Things to Consider
    # 1. Allow users to put Variances and not only Standard Deviations
    # 2. Adding Algina & Kesselman (2003) CI for dav
    # 3. I get the same CI's as in SPSS - verify it is the same for other softwares for cohend's drm (whoever calcualte the corrected version of it)
    # 4. Check Denis CI's number 8 in the function prepration
    # 5. A lot of work is needed to be able to implement cousineau and fitts CI's - I need to explore more and consult them for more information:
    # A: Check which df exactly is preferable to use (allow both for the paired pooled desing)
    # B: whether we can apply the non-central method to all types of effect sizes


def paired_samples_t_test(x, y):
    """
    Perform paired samples t-test and calculate effect size.

    Parameters
    ----------
    x : array-like
        First sample measurements
    y : array-like
        Second sample measurements

    Returns
    -------
    dict
        Dictionary containing:
        - 't_statistic': T-test statistic
        - 'p_value': P-value of the test
        - 'effect_size': Cohen's d effect size
        - 'ci_lower': Lower bound of confidence interval
        - 'ci_upper': Upper bound of confidence interval
        - 'interpretation': Text interpretation of results
    """
    # ... existing code ...
