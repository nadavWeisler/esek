"""Tests for PearsonCorrelationDifference."""
from __future__ import annotations

import math
import pytest

from esek.calculators.correlations import (
    PearsonCorrelationDifference,
    CorrelationDifferenceResult,
)


# ---------------------------------------------------------------------------
# Independent correlations
# ---------------------------------------------------------------------------


class TestIndependentCorrelations:
    def test_basic_output_type(self):
        result = PearsonCorrelationDifference.independent(0.5, 80, 0.3, 90)
        assert isinstance(result, CorrelationDifferenceResult)
        assert result.design == "independent"

    def test_n1_n2_preserved(self):
        result = PearsonCorrelationDifference.independent(0.4, 60, 0.2, 70)
        assert result.n1 == 60
        assert result.n2 == 70

    def test_r_values_preserved(self):
        result = PearsonCorrelationDifference.independent(0.5, 100, 0.3, 100)
        assert result.r1 == pytest.approx(0.5, abs=1e-5)
        assert result.r2 == pytest.approx(0.3, abs=1e-5)

    def test_difference(self):
        result = PearsonCorrelationDifference.independent(0.5, 100, 0.3, 100)
        assert result.difference == pytest.approx(0.2, abs=1e-5)

    def test_cohens_q(self):
        result = PearsonCorrelationDifference.independent(0.5, 100, 0.3, 100)
        expected_q = math.atanh(0.5) - math.atanh(0.3)
        assert result.cohens_q == pytest.approx(expected_q, abs=1e-4)

    def test_ci_zou_contains_difference_for_large_n(self):
        result = PearsonCorrelationDifference.independent(0.5, 300, 0.3, 300)
        lo, hi = result.ci_zou
        assert lo <= result.difference <= hi

    def test_ci_zou_ordering(self):
        result = PearsonCorrelationDifference.independent(0.5, 80, 0.3, 80)
        assert result.ci_zou[0] <= result.ci_zou[1]

    def test_r1_ci_ordering(self):
        result = PearsonCorrelationDifference.independent(0.5, 80, 0.3, 80)
        assert result.r1_ci[0] <= result.r1_ci[1]

    def test_p_values_in_range(self):
        result = PearsonCorrelationDifference.independent(0.5, 80, 0.3, 80)
        stat, p = result.tests["fisher_z"]
        assert 0.0 <= p <= 1.0

    def test_large_n_significant(self):
        result = PearsonCorrelationDifference.independent(0.7, 500, 0.2, 500)
        _, p = result.tests["fisher_z"]
        assert p < 0.001

    def test_equal_r_not_significant(self):
        result = PearsonCorrelationDifference.independent(0.5, 100, 0.5, 100)
        _, p = result.tests["fisher_z"]
        assert p == pytest.approx(1.0, abs=0.01)

    def test_invalid_r1(self):
        with pytest.raises(ValueError, match="r1"):
            PearsonCorrelationDifference.independent(1.1, 100, 0.5, 100)

    def test_invalid_r2(self):
        with pytest.raises(ValueError, match="r2"):
            PearsonCorrelationDifference.independent(0.5, 100, -1.5, 100)

    def test_invalid_n_too_small(self):
        with pytest.raises(ValueError):
            PearsonCorrelationDifference.independent(0.5, 2, 0.3, 80)

    def test_invalid_confidence_level(self):
        with pytest.raises(ValueError, match="confidence_level"):
            PearsonCorrelationDifference.independent(0.5, 100, 0.3, 100, confidence_level=0.0)

    def test_negative_r(self):
        result = PearsonCorrelationDifference.independent(-0.4, 100, 0.4, 100)
        assert result.difference == pytest.approx(-0.8, abs=1e-5)
        assert result.cohens_q < 0

    def test_confidence_level_99_wider(self):
        r95 = PearsonCorrelationDifference.independent(0.5, 80, 0.3, 80, confidence_level=0.95)
        r99 = PearsonCorrelationDifference.independent(0.5, 80, 0.3, 80, confidence_level=0.99)
        w95 = r95.ci_zou[1] - r95.ci_zou[0]
        w99 = r99.ci_zou[1] - r99.ci_zou[0]
        assert w99 > w95


# ---------------------------------------------------------------------------
# Dependent non-overlapping correlations
# ---------------------------------------------------------------------------


class TestDependentNonOverlapping:
    # Reference inputs from Steiger (1980): r12=.8, r34=.6, n=103
    # with cross-correlations consistent with the original example
    DEFAULT = dict(
        r12=0.5, r34=0.3, r13=0.4, r14=0.3, r23=0.3, r24=0.2, n=120
    )

    def test_basic_output_type(self):
        result = PearsonCorrelationDifference.dependent_non_overlapping(**self.DEFAULT)
        assert isinstance(result, CorrelationDifferenceResult)
        assert result.design == "dependent_non_overlapping"

    def test_n_preserved(self):
        result = PearsonCorrelationDifference.dependent_non_overlapping(**self.DEFAULT)
        assert result.n == 120

    def test_difference(self):
        result = PearsonCorrelationDifference.dependent_non_overlapping(**self.DEFAULT)
        assert result.difference == pytest.approx(0.2, abs=1e-5)

    def test_five_tests_present(self):
        result = PearsonCorrelationDifference.dependent_non_overlapping(**self.DEFAULT)
        expected_keys = {
            "pearson_filon_1898", "dunn_clark_1969", "steiger_1980",
            "raghunathan_1996", "silver_2004",
        }
        assert expected_keys == set(result.tests.keys())

    def test_p_values_in_range(self):
        result = PearsonCorrelationDifference.dependent_non_overlapping(**self.DEFAULT)
        for name, (stat, p) in result.tests.items():
            assert 0.0 <= p <= 1.0, f"{name} p={p} out of range"

    def test_ci_zou_ordering(self):
        result = PearsonCorrelationDifference.dependent_non_overlapping(**self.DEFAULT)
        assert result.ci_zou[0] <= result.ci_zou[1]

    def test_invalid_n_too_small(self):
        kwargs = {**self.DEFAULT, "n": 4}
        with pytest.raises(ValueError):
            PearsonCorrelationDifference.dependent_non_overlapping(**kwargs)

    def test_ci_meng_is_none(self):
        result = PearsonCorrelationDifference.dependent_non_overlapping(**self.DEFAULT)
        assert result.ci_meng is None

    def test_large_n_significant(self):
        result = PearsonCorrelationDifference.dependent_non_overlapping(
            r12=0.8, r34=0.3, r13=0.3, r14=0.2, r23=0.2, r24=0.1, n=200
        )
        for _, (_, p) in result.tests.items():
            assert p < 0.05


# ---------------------------------------------------------------------------
# Dependent overlapping correlations
# ---------------------------------------------------------------------------


class TestDependentOverlapping:
    # Meng et al. (1992) example: r12=.4, r13=.2, r23=.3, n=100
    DEFAULT = dict(r12=0.5, r13=0.3, r23=0.4, n=100)

    def test_basic_output_type(self):
        result = PearsonCorrelationDifference.dependent_overlapping(**self.DEFAULT)
        assert isinstance(result, CorrelationDifferenceResult)
        assert result.design == "dependent_overlapping"

    def test_n_preserved(self):
        result = PearsonCorrelationDifference.dependent_overlapping(**self.DEFAULT)
        assert result.n == 100

    def test_difference(self):
        result = PearsonCorrelationDifference.dependent_overlapping(**self.DEFAULT)
        assert result.difference == pytest.approx(0.2, abs=1e-5)

    def test_nine_tests_present(self):
        result = PearsonCorrelationDifference.dependent_overlapping(**self.DEFAULT)
        expected_keys = {
            "pearson_filon_1898", "hotelling_1940", "williams_1959",
            "olkin_1967", "dunn_clark_1969", "hendrickson_1970",
            "steiger_1980", "meng_1992", "hittner_2003",
        }
        assert expected_keys == set(result.tests.keys())

    def test_p_values_in_range(self):
        result = PearsonCorrelationDifference.dependent_overlapping(**self.DEFAULT)
        for name, (stat, p) in result.tests.items():
            assert 0.0 <= p <= 1.0, f"{name} p={p} out of range"

    def test_ci_zou_ordering(self):
        result = PearsonCorrelationDifference.dependent_overlapping(**self.DEFAULT)
        assert result.ci_zou[0] <= result.ci_zou[1]

    def test_ci_meng_is_not_none(self):
        result = PearsonCorrelationDifference.dependent_overlapping(**self.DEFAULT)
        assert result.ci_meng is not None
        assert result.ci_meng[0] <= result.ci_meng[1]

    def test_large_n_significant_for_large_difference(self):
        result = PearsonCorrelationDifference.dependent_overlapping(
            r12=0.8, r13=0.2, r23=0.4, n=200
        )
        # Most tests should be significant
        sig_count = sum(1 for _, (_, p) in result.tests.items() if p < 0.05)
        assert sig_count >= 5

    def test_equal_r_not_significant(self):
        result = PearsonCorrelationDifference.dependent_overlapping(
            r12=0.5, r13=0.5, r23=0.4, n=100
        )
        # All p-values should be > 0.5 (no difference)
        for _, (_, p) in result.tests.items():
            assert p > 0.1

    def test_invalid_n_too_small(self):
        with pytest.raises(ValueError):
            PearsonCorrelationDifference.dependent_overlapping(0.5, 0.3, 0.4, n=3)

    def test_invalid_r_range(self):
        with pytest.raises(ValueError):
            PearsonCorrelationDifference.dependent_overlapping(0.5, 0.3, 0.4, n=100, confidence_level=2.0)

    def test_cohens_q_formula(self):
        result = PearsonCorrelationDifference.dependent_overlapping(
            r12=0.6, r13=0.4, r23=0.3, n=150
        )
        expected_q = math.atanh(0.6) - math.atanh(0.4)
        assert result.cohens_q == pytest.approx(expected_q, abs=1e-4)

    def test_confidence_level_preserved(self):
        result = PearsonCorrelationDifference.dependent_overlapping(
            **self.DEFAULT, confidence_level=0.99
        )
        assert result.confidence_level == 0.99

    def test_99_ci_wider_than_95(self):
        r95 = PearsonCorrelationDifference.dependent_overlapping(**self.DEFAULT, confidence_level=0.95)
        r99 = PearsonCorrelationDifference.dependent_overlapping(**self.DEFAULT, confidence_level=0.99)
        assert r99.ci_zou[1] - r99.ci_zou[0] > r95.ci_zou[1] - r95.ci_zou[0]
