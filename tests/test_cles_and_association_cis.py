"""Unit tests for TwoIndependentCLES factories and association CIs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from esek.calculators.two_independent_mean import (
    TwoIndependentCLES,
    TwoIndependentCLESResult,
)
from esek.confidence_intervals import (
    CohensWCIResult,
    ContingencyCoefficientCIResult,
    CramerVCIResult,
    cohens_w_ci,
    contingency_coefficient_ci,
    cramer_v_ci,
)
from esek.core import InvalidInputError, StatisticalComputationError


class TestTwoIndependentCLESUnit:
    def test_from_t_score_result_type(self):
        result = TwoIndependentCLES.from_t_score(2.5, 30, 28)
        assert isinstance(result, TwoIndependentCLESResult)
        assert result.degrees_of_freedom == 56
        assert 0.0 < result.p_value <= 0.99999

    def test_from_parameters_matches_from_t_score(self):
        mean1, mean2, sd1, sd2, n1, n2 = 10.0, 8.0, 2.0, 2.5, 30, 28
        df = n1 + n2 - 2
        pooled = math.sqrt((((n1 - 1) * sd1**2) + ((n2 - 1) * sd2**2)) / df)
        se = pooled * math.sqrt(1 / n1 + 1 / n2)
        t_score = (mean1 - mean2) / se
        from_t = TwoIndependentCLES.from_t_score(t_score, n1, n2)
        from_p = TwoIndependentCLES.from_parameters(mean1, mean2, sd1, sd2, n1, n2)
        assert from_p.t_score == pytest.approx(from_t.t_score, abs=1e-12)
        assert from_p.cohens_ds.effect_size == pytest.approx(
            from_t.cohens_ds.effect_size, abs=1e-12
        )
        assert from_p.hedges_gs.effect_size == pytest.approx(
            from_t.hedges_gs.effect_size, abs=1e-12
        )
        assert from_p.cohens_dpop.effect_size == pytest.approx(
            from_t.cohens_dpop.effect_size, abs=1e-12
        )

    def test_from_data_matches_from_parameters(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(10, 2, size=40)
        g2 = rng.normal(8, 2.5, size=35)
        from_data = TwoIndependentCLES.from_data(g1, g2)
        from_params = TwoIndependentCLES.from_parameters(
            float(np.mean(g1)),
            float(np.mean(g2)),
            float(np.std(g1, ddof=1)),
            float(np.std(g2, ddof=1)),
            len(g1),
            len(g2),
        )
        assert from_data.t_score == pytest.approx(from_params.t_score, abs=1e-12)
        assert from_data.cohens_ds.cl.value == pytest.approx(
            from_params.cohens_ds.cl.value, abs=1e-12
        )

    def test_cles_measures_in_unit_interval(self):
        result = TwoIndependentCLES.from_parameters(12, 10, 2, 2, 25, 25)
        for std in (result.cohens_ds, result.hedges_gs, result.cohens_dpop):
            for measure in (std.u1, std.u2, std.u3, std.cl, std.pov):
                assert 0.0 <= measure.value <= 1.0
                assert 0.0 <= measure.central_ci.lower <= 1.0
                assert 0.0 <= measure.central_ci.upper <= 1.0

    def test_pop_diff_changes_t(self):
        base = TwoIndependentCLES.from_parameters(10, 8, 2, 2, 30, 30, pop_diff=0.0)
        shifted = TwoIndependentCLES.from_parameters(10, 8, 2, 2, 30, 30, pop_diff=2.0)
        assert base.t_score != pytest.approx(shifted.t_score)

    def test_invalid_n(self):
        with pytest.raises(InvalidInputError):
            TwoIndependentCLES.from_parameters(1, 0, 1, 1, 1, 10)

    def test_zero_sd_raises(self):
        with pytest.raises(InvalidInputError):
            TwoIndependentCLES.from_parameters(1, 0, 0.0, 1.0, 20, 20)

    def test_constant_groups_raise_computation_error(self):
        with pytest.raises((InvalidInputError, StatisticalComputationError)):
            TwoIndependentCLES.from_data([1, 1, 1, 1], [2, 2, 2, 2])

    def test_empty_group_raises(self):
        with pytest.raises(InvalidInputError):
            TwoIndependentCLES.from_data([], [1, 2, 3])

    def test_non_finite_raises(self):
        with pytest.raises(InvalidInputError, match="finite"):
            TwoIndependentCLES.from_data([1, 2, np.nan], [3, 4, 5])


class TestCohensWCIUnit:
    def test_result_type_and_transform(self):
        result = cohens_w_ci(0.3, n=100, df=2)
        assert isinstance(result, CohensWCIResult)
        assert result.chi_square == pytest.approx(0.3**2 * 100)
        assert result.ci[0] <= result.cohens_w <= result.ci[1]

    def test_wider_at_higher_confidence(self):
        r95 = cohens_w_ci(0.25, n=80, df=3, confidence_level=0.95)
        r80 = cohens_w_ci(0.25, n=80, df=3, confidence_level=0.80)
        assert (r95.ci[1] - r95.ci[0]) > (r80.ci[1] - r80.ci[0])

    def test_zero_w(self):
        result = cohens_w_ci(0.0, n=50, df=1)
        assert result.ci[0] == pytest.approx(0.0, abs=1e-6)
        assert result.ci[1] >= 0.0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            cohens_w_ci(-0.1, n=50, df=1)
        with pytest.raises(ValueError):
            cohens_w_ci(0.2, n=1, df=1)
        with pytest.raises(ValueError):
            cohens_w_ci(0.2, n=50, df=0)


class TestContingencyCoefficientCIUnit:
    def test_result_type_and_transform(self):
        c = 0.4
        n = 120
        result = contingency_coefficient_ci(c, n=n, df=4)
        assert isinstance(result, ContingencyCoefficientCIResult)
        expected_chi = (c**2 * n) / (1 - c**2)
        assert result.chi_square == pytest.approx(expected_chi)
        assert result.ci[0] <= result.contingency_coefficient <= result.ci[1]

    def test_near_zero(self):
        result = contingency_coefficient_ci(0.05, n=200, df=2)
        assert result.ci[0] >= 0.0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            contingency_coefficient_ci(1.0, n=50, df=1)
        with pytest.raises(ValueError):
            contingency_coefficient_ci(-0.1, n=50, df=1)
        with pytest.raises(ValueError):
            contingency_coefficient_ci(0.2, n=50, df=1, confidence_level=1.5)

    def test_consistency_with_cramer_family(self):
        # Sanity: related NCP CIs remain ordered for positive association strength
        w = cohens_w_ci(0.3, n=100, df=2)
        v = cramer_v_ci(0.3 / math.sqrt(2), n=100, df=2)
        assert isinstance(v, CramerVCIResult)
        assert w.ci[1] > 0
        assert v.ci[1] > 0


class TestOrdinalByOrdinalTypedUnit:
    def test_typed_and_legacy(self):
        from esek.calculators.correlations import OrdinalByOrdinal, OrdinalByOrdinalResult

        x = [1, 2, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1]
        y = [1, 2, 2, 3, 1, 3, 2, 2, 4, 2, 3, 1]
        typed = OrdinalByOrdinal.from_data(x, y, confidence_level=0.95, n_bootstrap=10)
        assert isinstance(typed, OrdinalByOrdinalResult)
        assert typed.confidence_level == 0.95

        legacy = OrdinalByOrdinal.from_data(
            params={
                "Variable 1": x,
                "Variable 2": y,
                "Confidence Level": 95,
                "Number Of Bootstraps Samples": 10,
            }
        )
        assert isinstance(legacy, dict)
        assert "Spearman Correlation" in legacy

    def test_typed_from_contingency_table(self):
        from esek.calculators.correlations import OrdinalByOrdinal, OrdinalByOrdinalResult

        table = [[5, 2, 1], [1, 4, 2], [0, 2, 6]]
        result = OrdinalByOrdinal.from_contingency_table(
            table, confidence_level=0.95, n_bootstrap=10
        )
        assert isinstance(result, OrdinalByOrdinalResult)
        assert result.metadata["n"] == 23

    def test_missing_args_raise(self):
        from esek.calculators.correlations import OrdinalByOrdinal
        from esek.core import InvalidInputError

        with pytest.raises(InvalidInputError):
            OrdinalByOrdinal.from_data()
        with pytest.raises(InvalidInputError):
            OrdinalByOrdinal.from_contingency_table()
