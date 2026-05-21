"""TDD contract tests for statistical functions currently implemented on origin/dev."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_FUNCTIONS_BY_FILE = {
    "stats/CI_Constructor/1_CohensDFamily/CI_Constructor_cohen_d.py": [
        "calculate_central_ci_from_cohens_d_one_sample",
        "Pivotal_ci_t",
        "NCT_ci_t",
        "calculate_central_ci_paired_samples_t_test",
        "calculate_SE_pooled_paired_samples_t_test",
        "CI_NCP_one_Sample",
        "CI_adjusted_lambda_prime_Paired_Samples",
        "CI_MAG_Paired_Samples",
        "CI_Morris_Paired_Samples",
        "CI_t_prime_Paired_Samples",
        "CI_t_Algina_Keselman",
        "calculate_central_ci_from_cohens_d_two_samples",
        "calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test",
    ],
    "stats/CI_Constructor/2_MeasuresOfAssociationsAndCorrelationsCI/Associations_and_Correlations.py": [
        "ncp_ci",
        "NonCentralCiF",
    ],
    "stats/CI_Constructor/3_EtaSquareFamily/CI_Constructor_eta.py": [
        "Confidence_Interval_Partial_Eta_Square_Family",
    ],
    "stats/CI_Constructor/CI_Constructor.py": [
        "calculate_central_ci_from_cohens_d_one_sample",
        "Pivotal_ci_t",
        "NCT_ci_t",
        "calculate_central_ci_paired_samples_t_test",
        "calculate_SE_pooled_paired_samples_t_test",
        "CI_NCP_one_Sample",
        "CI_adjusted_lambda_prime_Paired_Samples",
        "CI_MAG_Paired_Samples",
        "CI_Morris_Paired_Samples",
        "CI_t_prime_Paired_Samples",
        "CI_t_Algina_Keselman",
        "calculate_central_ci_from_cohens_d_two_samples",
        "calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test",
    ],
    "stats/Calculator/AssociationCorrelations/IntervalRatioCorrelation.py": [
        "Non_Central_CI_F",
        "pearson_correlation",
        "Rsquare_Estimation",
        "Robust_Measures_Interval",
        "skipped_Correlation",
    ],
    "stats/Calculator/AssociationCorrelations/MultipleCorrelation/Multiple_R_Square.py": [
        "NonCentralCiF",
        "Rsquare_Estimation",
    ],
    "stats/Calculator/AssociationCorrelations/NominalByInterval/Nominal_by_Interval.py": [
        "NonCentralCiF",
        "point_biserial_correlation",
        "Eta_Correlation_Ratio",
    ],
    "stats/Calculator/AssociationCorrelations/NominalByNominal/Nominal_By_Nominal.py": [
        "ncp_ci",
        "Berry_Mielke_Maximum_Corrected_Cramer_V_output_matrix",
        "goodman_kruskal_lamda_correlation",
        "Goodman_Kruskal_Tau",
        "Columns_to_Contingency",
        "multilevel_contingency_tables",
    ],
    "stats/Calculator/AssociationCorrelations/NominalByOrdinal/NominalByOrdinal.py": [
        "Freemans_Theta",
        "Rank_Biserial_Correlation",
    ],
    "stats/Calculator/AssociationCorrelations/OrdinalByInterval/Ordinal_By_Interval.py": [
        "skipped_Correlation",
        "Spearman_Correlation",
        "Gausian_Rank_Correlation",
        "bsmahal",
        "shepherd",
        "ginis_gamma",
    ],
    "stats/Calculator/AssociationCorrelations/OrdinalByOrdinal/Ordinal_By_Ordinal.py": [
        "skipped_Correlation",
        "Spearman_Correlation",
        "Gausian_Rank_Correlation",
        "bsmahal",
        "shepherd",
        "ginis_gamma",
        "Gamma_Family_Measures",
    ],
    "stats/Calculator/Medians/Multi_Dep_Medians.py": [
        "WinsorizedVariance",
    ],
    "stats/Calculator/Medians/One_Sample_Median.py": [
        "effect_sizes_for_one_sample_median",
    ],
    "stats/Calculator/Medians/TwoIndependentMedians/Two_Ind_Medians.py": [
        "effect_sizes_for_Indpednent_medians",
    ],
    "stats/Calculator/Medians/TwoIndependentMedians/Two_Ind_Medians2.py": [
        "effect_sizes_for_Indpednent_medians",
    ],
    "stats/Calculator/Medians/Two_Paired_Medians.py": [
        "effect_sizes_for_paired_medians",
    ],
    "stats/Calculator/OneSampleMean/OneSampleAparametric.py": [
        "apermetric_effect_size_one_sample",
    ],
    "stats/Calculator/OneSampleMean/OneSampleCLES.py": [
        "pivotal_ci_t",
        "calculate_central_ci_from_cohens_d_one_sample_t_test",
        "ci_ncp_one_sample",
        "density",
        "area_under_function",
        "WinsorizedVariance",
        "WinsorizedCorrelation",
    ],
    "stats/Calculator/OneSampleMean/OneSampleT.py": [
        "pivotal_ci_t",
        "calculate_central_ci_one_sample_t_test",
        "ci_ncp_one_sample",
        "one_sample_from_t_score",
        "one_sample_from_params",
    ],
    "stats/Calculator/OneSampleMean/OneSampleZ.py": [
        "calculate_central_ci_from_cohens_d_one_sample",
        "one_sample_from_z_score",
        "one_sample_from_parameters",
        "one_sample_from_data",
    ],
    "stats/Calculator/Poportions/MultipleProportions.py": [
        "Cochran_Q_based_Effect_Size",
        "goodness_of_fit_from_frequency",
    ],
    "stats/Calculator/Poportions/OneSampleProportion.py": [
        "main_one_sample_proportion",
    ],
    "stats/Calculator/Poportions/TwoIndProportions.py": [
        "main_Two_sample_proportions",
    ],
    "stats/Calculator/Poportions/TwoPairedProportions.py": [
        "Main_Two_Dep_Proportions_From_Parameters",
    ],
    "stats/Calculator/TwoIndpendentMeans/EffectSizeWithControlGroup.py": [
        "Pivotal_ci_t",
        "calculate_central_ci_from_cohens_d_one_sample_t_test",
    ],
    "stats/Calculator/TwoIndpendentMeans/IndependentSamplesCLES.py": [
        "Pivotal_ci_t",
        "calculate_Central_ci_from_cohens_d_two_indpednent_sample_t_test",
    ],
    "stats/Calculator/TwoIndpendentMeans/Robust_Independent_Samples.py": [
        "density",
        "area_under_function",
        "WinsorizedVariance",
        "WinsorizedCorrelation",
    ],
    "stats/Calculator/TwoIndpendentMeans/TwoIndependentSamplesT.py": [
        "Pivotal_ci_t",
        "calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test",
    ],
    "stats/Calculator/TwoIndpendentMeans/TwoIndependentSamplesZ.py": [
        "calculate_central_ci_from_cohens_d_two_samples",
    ],
    "stats/Calculator/TwoIndpendentMeans/UnequalVariances.py": [
        "Pivotal_ci_t",
        "calculate_central_ci_from_cohens_d_two_indpednent_sample_t_test",
    ],
    "stats/Calculator/TwoPairedMeans/ControlGroupPrePost.py": [
        "Pivotal_ci_t",
        "calculate_central_ci_from_cohens_d_one_sample_t_test",
    ],
    "stats/Calculator/TwoPairedMeans/PairedSamplesT.py": [
        "pivotal_ci_t",
        "nct_ci_t",
        "calculate_central_ci_paired_samples_t_test",
        "calculate_se_pooled_paired_samples_t_test",
        "ci_ncp_paired_samples_difference",
        "ci_adjusted_lambda_prime_Paired_Samples",
        "ci_mag_paired_samples",
        "ci_morris_paired_samples",
        "ci_t_prime_paired_samples",
        "ci_ncp_paired_samples_pooled",
        "ci_t_algina_keselman",
        "paired_samples_t_test",
    ],
    "stats/Calculator/TwoPairedMeans/PairedSamplesZ.py": [
        "calculate_central_ci_from_cohens_d_one_sample",
    ],
    "stats/Calculator/TwoPairedMeans/RobustPairedSamples.py": [
        "density",
        "area_under_function",
        "WinsorizedVariance",
        "WinsorizedCorrelation",
        "robust_paired_samples",
    ],
    "stats/Calculator/TwoPairedMeans/TwoPairedSamplesCLES.py": [
        "Pivotal_ci_t",
        "calculate_central_ci_from_cohens_d_two_paired_sample_t_test",
        "paired_samples_cles",
    ],
    "stats/Differecnes/DifferencesBetweenCorrelations/Nominal Variables/Diference_Categorical.py": [
        "goodman_kruskal_lamda_correlation",
        "Goodman_Kruskal_Tau",
        "Columns_to_Contingency",
    ],
}


@pytest.mark.parametrize("relative_path", sorted(EXPECTED_FUNCTIONS_BY_FILE))
def test_dev_statistical_module_file_exists(relative_path: str) -> None:
    module_path = REPO_ROOT / relative_path
    assert module_path.exists(), (
        f"TDD failure: expected dev module not found on main yet: {relative_path}"
    )


@pytest.mark.parametrize("relative_path", sorted(EXPECTED_FUNCTIONS_BY_FILE))
def test_dev_statistical_module_defines_expected_functions(relative_path: str) -> None:
    module_path = REPO_ROOT / relative_path
    assert module_path.exists(), (
        f"TDD failure: expected dev module not found on main yet: {relative_path}"
    )

    source = module_path.read_text(encoding="utf-8")
    parsed = ast.parse(source)

    actual_functions = {
        node.name
        for node in parsed.body
        if isinstance(node, ast.FunctionDef)
    }
    expected_functions = set(EXPECTED_FUNCTIONS_BY_FILE[relative_path])

    missing_functions = sorted(expected_functions - actual_functions)
    assert not missing_functions, (
        "TDD failure: missing function definitions in "
        f"{relative_path}: {missing_functions}"
    )
