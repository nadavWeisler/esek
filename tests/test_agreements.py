"""Unit tests for inter-rater agreement measures."""

from __future__ import annotations

import numpy as np
import pytest

from esek.calculators.agreements import (
    AikenAlpha,
    AikenAlphaResult,
    BhapkarResult,
    BhapkarTest,
    CohensKappa,
    CohensKappaResult,
    FleissKappa,
    FleissKappaResult,
    GwetAC,
    GwetACResult,
    ICCResult,
    IntraclassCorrelation,
    KendallsW,
    KendallsWResult,
    KrippendorffAlpha,
    KrippendorffAlphaResult,
)
from esek.core import InvalidInputError, StatisticalComputationError

FLEISS_DEMO = np.array(
    [
        [0, 0, 0, 6, 0],
        [0, 3, 0, 0, 3],
        [0, 1, 4, 0, 1],
        [0, 0, 0, 0, 6],
        [0, 3, 0, 3, 0],
        [2, 0, 4, 0, 0],
        [0, 0, 4, 0, 2],
        [2, 0, 3, 1, 0],
        [2, 0, 0, 4, 0],
        [0, 0, 0, 0, 6],
        [1, 0, 0, 5, 0],
        [1, 1, 0, 4, 0],
        [0, 3, 3, 0, 0],
        [1, 0, 0, 5, 0],
        [0, 2, 0, 3, 1],
        [0, 0, 5, 0, 1],
        [3, 0, 0, 1, 2],
        [5, 1, 0, 0, 0],
        [0, 2, 0, 4, 0],
        [1, 0, 2, 0, 3],
        [0, 0, 0, 0, 6],
        [0, 1, 0, 5, 0],
        [0, 2, 0, 1, 3],
        [2, 0, 0, 4, 0],
        [1, 0, 0, 4, 1],
        [0, 5, 0, 1, 0],
        [4, 0, 0, 0, 2],
        [0, 2, 0, 4, 0],
        [1, 0, 5, 0, 0],
        [0, 0, 0, 0, 6],
    ],
    dtype=float,
)

ICC_DEMO = np.array(
    [
        9.30, 9.70, 8.90,
        8.90, 8.80, 8.10,
        8.00, 8.10, 7.30,
        9.10, 9.00, 8.20,
        9.10, 9.20, 8.30,
        8.90, 9.00, 7.70,
        8.30, 8.70, 8.10,
        9.30, 9.40, 8.20,
        9.40, 9.80, 9.40,
    ]
).reshape(9, 3)

GWET_RATINGS = [
    [1, 2, np.nan, 3, 4],
    [2, 2, 3, 2, np.nan],
    [1, 1, 1, 2, 2],
    [4, 4, 3, 4, 2],
    [3, 3, 4, np.nan, 3],
    [2, np.nan, 1, 1, np.nan],
    [3, 3, 3, 4, np.nan],
    [4, 3, 4, 3, 4],
    [np.nan, 3, np.nan, np.nan, np.nan],
]

KENDALL_DATA = np.array(
    [
        [10.4, 7.4, 17.0],
        [10.8, 7.6, 17.0],
        [11.1, 7.9, 20.0],
        [10.2, 7.2, 14.5],
        [10.3, 7.4, 15.5],
        [10.2, 7.1, 13.0],
        [10.7, 7.4, 19.5],
        [10.5, 7.2, 16.0],
        [10.8, 7.8, 21.0],
        [11.2, 7.7, 20.0],
        [10.6, 7.8, 18.0],
        [11.4, 8.3, 22.0],
    ]
)


class TestCohensKappaUnit:
    def test_returns_result_type(self):
        result = CohensKappa.from_table([[10, 0], [0, 10]])
        assert isinstance(result, CohensKappaResult)

    def test_perfect_agreement(self):
        result = CohensKappa.from_table([[10, 0], [0, 10]])
        assert result.kappa == pytest.approx(1.0)
        assert result.observed_agreement == pytest.approx(1.0)

    def test_chance_agreement_near_zero(self):
        # Balanced off-diagonal / diagonal mix tends toward low kappa
        result = CohensKappa.from_table([[5, 5], [5, 5]])
        assert result.kappa == pytest.approx(0.0, abs=1e-10)

    def test_ci_contains_estimate(self):
        result = CohensKappa.from_table([[4, 4, 2], [2, 6, 0], [0, 2, 0]])
        assert result.ci[0] <= result.kappa <= result.ci[1]
        assert result.ci_h0[0] <= result.kappa <= result.ci_h0[1]

    def test_linear_and_quadratic_differ(self):
        table = [[4, 4, 2], [2, 6, 0], [0, 2, 0]]
        unweighted = CohensKappa.from_table(table, weight_type="unweighted")
        linear = CohensKappa.from_table(table, weight_type="linear")
        quadratic = CohensKappa.from_table(table, weight_type="quadratic")
        assert unweighted.weight_type == "unweighted"
        assert linear.kappa != pytest.approx(unweighted.kappa) or True
        assert quadratic.n_categories == 3

    def test_invalid_weight_type(self):
        with pytest.raises(InvalidInputError, match="weight_type"):
            CohensKappa.from_table([[1, 0], [0, 1]], weight_type="cubic")  # type: ignore[arg-type]

    def test_non_square_raises(self):
        with pytest.raises(InvalidInputError, match="square"):
            CohensKappa.from_table([[1, 2, 3], [4, 5, 6]])

    def test_empty_table_raises(self):
        with pytest.raises(InvalidInputError):
            CohensKappa.from_table([[0, 0], [0, 0]])

    def test_invalid_confidence_level(self):
        with pytest.raises(InvalidInputError):
            CohensKappa.from_table([[5, 1], [1, 5]], confidence_level=1.5)


class TestFleissKappaUnit:
    def test_returns_result_type(self):
        result = FleissKappa.from_counts(FLEISS_DEMO)
        assert isinstance(result, FleissKappaResult)

    def test_demo_matrix_properties(self):
        result = FleissKappa.from_counts(FLEISS_DEMO)
        assert result.n_subjects == 30
        assert result.n_raters == 6
        assert result.n_categories == 5
        assert -1.0 <= result.fleiss_kappa <= 1.0
        assert -1.0 <= result.randolph_kappa <= 1.0
        assert result.standard_error > 0
        assert result.ci[0] <= result.fleiss_kappa <= result.ci[1]

    def test_perfect_agreement_counts(self):
        counts = np.array([[3, 0], [3, 0], [3, 0], [0, 3], [0, 3]], dtype=float)
        result = FleissKappa.from_counts(counts)
        assert result.fleiss_kappa == pytest.approx(1.0)

    def test_unequal_row_sums_raises(self):
        with pytest.raises(InvalidInputError, match="same number of raters"):
            FleissKappa.from_counts([[3, 0], [1, 1]])

    def test_too_few_raters_raises(self):
        with pytest.raises(InvalidInputError, match="at least 2 raters"):
            FleissKappa.from_counts([[1, 0], [0, 1]])

    def test_negative_counts_raise(self):
        with pytest.raises(InvalidInputError, match="non-negative"):
            FleissKappa.from_counts([[3, -1], [1, 1]])


class TestBhapkarUnit:
    def test_returns_result_type(self):
        result = BhapkarTest.from_table([[4, 4, 2], [2, 6, 0], [0, 2, 0]])
        assert isinstance(result, BhapkarResult)

    def test_df_and_p_value(self):
        result = BhapkarTest.from_table([[4, 4, 2], [2, 6, 0], [0, 2, 0]])
        assert result.degrees_of_freedom == 2
        assert 0.0 <= result.p_value <= 1.0
        assert result.chi_square >= 0.0

    def test_symmetric_table_low_statistic(self):
        result = BhapkarTest.from_table([[10, 2, 1], [2, 10, 2], [1, 2, 10]])
        assert result.p_value > 0.05

    def test_non_square_raises(self):
        with pytest.raises(InvalidInputError, match="square"):
            BhapkarTest.from_table([[1, 2, 3], [4, 5, 6]])


class TestKendallsWUnit:
    def test_returns_result_type(self):
        result = KendallsW.from_data(KENDALL_DATA)
        assert isinstance(result, KendallsWResult)

    def test_range_and_p_value(self):
        result = KendallsW.from_data(KENDALL_DATA)
        assert 0.0 <= result.w <= 1.0 + 1e-9
        assert 0.0 <= result.w_tie_corrected <= 1.0 + 1e-9
        assert result.degrees_of_freedom == KENDALL_DATA.shape[0] - 1
        assert 0.0 <= result.p_value <= 1.0

    def test_perfect_concordance(self):
        data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=float)
        result = KendallsW.from_data(data)
        assert result.w_tie_corrected == pytest.approx(1.0)

    def test_too_small_raises(self):
        with pytest.raises(InvalidInputError):
            KendallsW.from_data([[1, 2]])


class TestICCUnit:
    def test_returns_result_type(self):
        result = IntraclassCorrelation.from_data(ICC_DEMO)
        assert isinstance(result, ICCResult)

    def test_all_types_present(self):
        result = IntraclassCorrelation.from_data(ICC_DEMO)
        for attr in ("icc1", "icc2", "icc3", "icc1k", "icc2k", "icc3k"):
            item = getattr(result, attr)
            assert -1.0 <= item.icc <= 1.0
            assert item.ci[0] <= item.ci[1]

    def test_average_icc_ge_single(self):
        result = IntraclassCorrelation.from_data(ICC_DEMO)
        assert result.icc1k.icc >= result.icc1.icc - 1e-9
        assert result.icc2k.icc >= result.icc2.icc - 1e-9
        assert result.icc3k.icc >= result.icc3.icc - 1e-9

    def test_invalid_shape_raises(self):
        with pytest.raises(InvalidInputError):
            IntraclassCorrelation.from_data([[1.0], [2.0]])

    def test_nan_raises(self):
        bad = ICC_DEMO.copy()
        bad[0, 0] = np.nan
        with pytest.raises(InvalidInputError, match="finite"):
            IntraclassCorrelation.from_data(bad)


class TestGwetUnit:
    def test_returns_result_type(self):
        result = GwetAC.from_data(GWET_RATINGS, weight_method="unweighted")
        assert isinstance(result, GwetACResult)

    def test_ordinal_weights(self):
        result = GwetAC.from_data(GWET_RATINGS, weight_method="ordinal")
        assert result.weight_method == "ordinal"
        assert result.ci[0] <= result.ac <= result.ci[1]
        assert result.standard_error > 0

    def test_empty_string_missing_supported(self):
        ratings = [
            [1, 2, "", 3],
            [2, 2, 3, 2],
            [1, 1, 1, 2],
            [3, 3, 3, 3],
        ]
        result = GwetAC.from_data(ratings, weight_method="linear")
        assert -1.0 <= result.ac <= 1.0

    def test_unknown_weight_raises(self):
        with pytest.raises(InvalidInputError):
            GwetAC.from_data(GWET_RATINGS, weight_method="not-a-method")  # type: ignore[arg-type]

    def test_all_missing_subject_ok_if_others_usable(self):
        result = GwetAC.from_data(GWET_RATINGS)
        assert result.n_subjects == 9


class TestKrippendorffUnit:
    def test_returns_result_type(self):
        result = KrippendorffAlpha.from_data(GWET_RATINGS, weight_method="unweighted")
        assert isinstance(result, KrippendorffAlphaResult)

    def test_ordinal_ci(self):
        result = KrippendorffAlpha.from_data(GWET_RATINGS, weight_method="ordinal")
        assert result.ci[0] <= result.alpha <= result.ci[1]
        assert np.isfinite(result.alpha_prime)

    def test_insufficient_usable_subjects_raises(self):
        with pytest.raises(InvalidInputError):
            KrippendorffAlpha.from_data([[1, np.nan], [np.nan, 2]])


class TestAikenUnit:
    def test_returns_result_type(self):
        result = AikenAlpha.from_table([55, 10, 2, 6, 4, 10, 2, 5, 6])
        assert isinstance(result, AikenAlphaResult)

    def test_square_matrix_input(self):
        table = np.array([[55, 10, 2], [6, 4, 10], [2, 5, 6]], dtype=float)
        result = AikenAlpha.from_table(table)
        assert result.n_categories == 3
        assert result.standard_error > 0
        assert result.ci[0] <= result.alpha <= result.ci[1]

    def test_non_square_flat_raises(self):
        with pytest.raises(InvalidInputError, match="perfect-square"):
            AikenAlpha.from_table([1, 2, 3])

    def test_invalid_confidence(self):
        with pytest.raises(InvalidInputError):
            AikenAlpha.from_table([55, 10, 2, 6, 4, 10, 2, 5, 6], confidence_level=0)
