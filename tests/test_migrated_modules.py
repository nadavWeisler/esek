"""Tests for the new modules migrated from the dev branch.

Covers:
    - PearsonCorrelation (interval_by_interval)
    - StatisticToEffectSize (statistic_to_effect_size)
    - ContingencyTable2x2 (contingency_tables)
    - Nominal/Ordinal correlation classes (correlations sub-package)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from esek import ContingencyTable2x2, PearsonCorrelation, StatisticToEffectSize


# ============================================================
# PearsonCorrelation
# ============================================================


class TestPearsonCorrelation:
    """Tests for PearsonCorrelation.from_data."""

    @pytest.fixture()
    def data(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 80)
        y = 0.6 * x + rng.normal(0, 1, 80)
        return x, y

    def test_returns_correct_types(self, data):
        x, y = data
        result = PearsonCorrelation.from_data(x, y)
        assert isinstance(result.r, float)
        assert isinstance(result.r_squared, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.ci_fisher, tuple) and len(result.ci_fisher) == 2
        assert isinstance(result.ci_bonett, tuple) and len(result.ci_bonett) == 2
        assert isinstance(result.ci_bootstrap, tuple) and len(result.ci_bootstrap) == 2
        assert isinstance(result.ci_ncp, tuple) and len(result.ci_ncp) == 2

    def test_r_in_range(self, data):
        x, y = data
        result = PearsonCorrelation.from_data(x, y)
        assert -1.0 <= result.r <= 1.0

    def test_r_squared_consistency(self, data):
        x, y = data
        result = PearsonCorrelation.from_data(x, y)
        assert result.r_squared == pytest.approx(result.r**2, abs=1e-10)

    def test_fisher_ci_contains_r(self, data):
        x, y = data
        result = PearsonCorrelation.from_data(x, y)
        assert result.ci_fisher[0] < result.r < result.ci_fisher[1]

    def test_bonett_ci_contains_r(self, data):
        x, y = data
        result = PearsonCorrelation.from_data(x, y)
        assert result.ci_bonett[0] < result.r < result.ci_bonett[1]

    def test_perfect_positive_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = 2.0 * x + 1.0
        result = PearsonCorrelation.from_data(x, y)
        assert result.r == pytest.approx(1.0, abs=1e-6)
        assert result.p_value == pytest.approx(0.0, abs=1e-6)
        # For r=1, Fisher CI should be (1.0, 1.0) or (nan, nan) — just check it doesn't crash
        assert not math.isnan(result.r)

    def test_known_numerical_value(self):
        """Known example: r≈0.4 for given seed."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 60)
        y = 0.5 * x + rng.normal(0, 1, 60)
        result = PearsonCorrelation.from_data(x, y, bootstrap_random_state=99)
        assert result.r == pytest.approx(0.4013, abs=5e-4)

    def test_wider_ci_at_lower_level(self, data):
        x, y = data
        r95 = PearsonCorrelation.from_data(x, y, confidence_level=0.95)
        r80 = PearsonCorrelation.from_data(x, y, confidence_level=0.80)
        width95 = r95.ci_fisher[1] - r95.ci_fisher[0]
        width80 = r80.ci_fisher[1] - r80.ci_fisher[0]
        assert width95 > width80

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            PearsonCorrelation.from_data(np.array([1, 2, 3]), np.array([1, 2]))

    def test_too_small_n_raises(self):
        with pytest.raises(ValueError, match="at least 4"):
            PearsonCorrelation.from_data(np.array([1, 2, 3]), np.array([1, 2, 3]))

    def test_invalid_ci_level_raises(self):
        x = np.arange(10, dtype=float)
        with pytest.raises(ValueError, match="confidence_level"):
            PearsonCorrelation.from_data(x, x, confidence_level=1.5)

    def test_standard_errors_present(self, data):
        x, y = data
        result = PearsonCorrelation.from_data(x, y)
        for key in ("fisher_1896", "bonett", "regression"):
            assert key in result.standard_errors
            assert isinstance(result.standard_errors[key], float)

    def test_metadata_present(self, data):
        x, y = data
        result = PearsonCorrelation.from_data(x, y)
        assert "slope" in result.metadata
        assert "common_language_effect_size_dunlap" in result.metadata


# ============================================================
# StatisticToEffectSize
# ============================================================


class TestStatisticToEffectSize:
    """Tests for StatisticToEffectSize — z→d and t→d conversions."""

    # --- z → d ---

    def test_z_one_sample_known(self):
        # d = z / sqrt(n) = 3 / sqrt(25) = 0.6
        result = StatisticToEffectSize.from_z_one_sample(z=3.0, n=25)
        assert result.cohens_d == pytest.approx(0.6, abs=1e-6)
        assert result.design == "one_sample"
        assert result.input_statistic == "z"

    def test_z_paired_known(self):
        result = StatisticToEffectSize.from_z_paired(z=2.0, n=16)
        assert result.cohens_d == pytest.approx(0.5, abs=1e-6)
        assert result.design == "paired"

    def test_z_independent_equal_n(self):
        # For equal n1=n2=n, simplifies to d = z * sqrt(2/n) when n1=n2
        n1, n2 = 30, 30
        z = 2.5
        total = n1 + n2
        harmonic = 2.0 * n1 * n2 / total  # = n = 30 when equal
        expected_d = (2.0 * z / math.sqrt(total)) * math.sqrt(harmonic / (2.0 * n1 * n2 / total))
        result = StatisticToEffectSize.from_z_independent(z=z, n1=n1, n2=n2)
        assert result.cohens_d == pytest.approx(expected_d, abs=1e-6)
        assert result.design == "independent"

    # --- t → d ---

    def test_t_one_sample_known(self):
        # d = t / sqrt(df) where df = n - 1 = 29
        result = StatisticToEffectSize.from_t_one_sample(t=2.0, n=30)
        expected_d = 2.0 / math.sqrt(29)
        assert result.cohens_d == pytest.approx(expected_d, abs=1e-6)
        assert result.hedges_g is not None
        assert abs(result.hedges_g) < abs(result.cohens_d)  # bias correction shrinks

    def test_t_paired_known(self):
        # d = t / sqrt(n) = 2 / sqrt(30)
        result = StatisticToEffectSize.from_t_paired(t=2.0, n=30)
        assert result.cohens_d == pytest.approx(2.0 / math.sqrt(30), abs=1e-6)

    def test_t_independent_known(self):
        # d = t * sqrt(1/n1 + 1/n2) for equal n = sqrt(2/30) * t
        n1, n2 = 30, 30
        t = 2.5
        expected_d = t * math.sqrt(1.0 / n1 + 1.0 / n2)
        result = StatisticToEffectSize.from_t_independent(t=t, n1=n1, n2=n2)
        assert result.cohens_d == pytest.approx(expected_d, abs=1e-6)
        assert result.hedges_g is not None

    def test_t_independent_unequal_n(self):
        result = StatisticToEffectSize.from_t_independent(t=2.0, n1=20, n2=40)
        expected_d = 2.0 * math.sqrt(1.0 / 20 + 1.0 / 40)
        assert result.cohens_d == pytest.approx(expected_d, abs=1e-6)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            StatisticToEffectSize.from_z_one_sample(z=1.0, n=1)
        with pytest.raises(ValueError):
            StatisticToEffectSize.from_t_one_sample(t=1.0, n=0)

    def test_negative_t_gives_negative_d(self):
        result = StatisticToEffectSize.from_t_one_sample(t=-2.0, n=30)
        assert result.cohens_d < 0
        assert result.hedges_g < 0

    def test_hedges_g_smaller_than_d(self):
        """Hedges' g should be smaller in magnitude than d (bias correction)."""
        result = StatisticToEffectSize.from_t_independent(t=2.0, n1=10, n2=10)
        assert abs(result.hedges_g) < abs(result.cohens_d)

    def test_via_effect_size_converter(self):
        """EffectSizeConverter should expose the same functionality."""
        from esek import EffectSizeConverter
        result = EffectSizeConverter.from_t_one_sample(t=3.0, n=30)
        expected = StatisticToEffectSize.from_t_one_sample(t=3.0, n=30)
        assert result.cohens_d == pytest.approx(expected.cohens_d, abs=1e-10)


# ============================================================
# ContingencyTable2x2
# ============================================================


class TestContingencyTable2x2:
    """Tests for ContingencyTable2x2.from_table."""

    @pytest.fixture()
    def table_4_4(self):
        """Balanced 2×2 table [[40,20],[15,35]]."""
        return np.array([[40, 20], [15, 35]])

    def test_phi_sign(self, table_4_4):
        """More a-d than a-b/c-d → positive phi."""
        result = ContingencyTable2x2.from_table(table_4_4)
        assert result.phi > 0

    def test_phi_range(self, table_4_4):
        result = ContingencyTable2x2.from_table(table_4_4)
        assert -1.0 <= result.phi <= 1.0

    def test_odds_ratio_positive(self, table_4_4):
        result = ContingencyTable2x2.from_table(table_4_4)
        assert result.odds_ratio > 0

    def test_chi_square_positive(self, table_4_4):
        result = ContingencyTable2x2.from_table(table_4_4)
        assert result.chi_square > 0

    def test_p_value_range(self, table_4_4):
        result = ContingencyTable2x2.from_table(table_4_4)
        assert 0.0 <= result.p_value <= 1.0

    def test_known_phi_value(self):
        """Known phi: for [[10,10],[10,10]] phi=0."""
        result = ContingencyTable2x2.from_table(np.array([[10, 10], [10, 10]]))
        assert result.phi == pytest.approx(0.0, abs=1e-6)
        assert result.chi_square == pytest.approx(0.0, abs=1e-6)
        assert result.p_value == pytest.approx(1.0, abs=1e-6)

    def test_known_or_perfect_association(self):
        """[[10,0],[0,10]]: perfect association → phi=1, OR=inf."""
        result = ContingencyTable2x2.from_table(np.array([[10, 0], [0, 10]]))
        assert result.phi == pytest.approx(1.0, abs=1e-4)
        assert math.isnan(result.odds_ratio)  # b*c = 0 → undefined

    def test_tetrachoric_r_range(self, table_4_4):
        result = ContingencyTable2x2.from_table(table_4_4)
        assert -1.0 <= result.tetrachoric_r <= 1.0

    def test_ci_swing_d_covers_estimate(self, table_4_4):
        result = ContingencyTable2x2.from_table(table_4_4)
        lo, hi = result.ci_swing_d_cols
        if not math.isnan(lo):
            assert lo < result.wallis_swing_d_cols < hi

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="2×2"):
            ContingencyTable2x2.from_table(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_negative_cell_raises(self):
        with pytest.raises(ValueError, match="≥ 0"):
            ContingencyTable2x2.from_table(np.array([[10, -1], [5, 8]]))

    def test_all_zeros_raises(self):
        with pytest.raises(ValueError, match="all zeros"):
            ContingencyTable2x2.from_table(np.array([[0, 0], [0, 0]]))

    def test_metadata_cell_counts(self, table_4_4):
        result = ContingencyTable2x2.from_table(table_4_4)
        assert result.metadata["a"] == 40
        assert result.metadata["b"] == 20
        assert result.metadata["c"] == 15
        assert result.metadata["d"] == 35

    def test_n_correct(self, table_4_4):
        result = ContingencyTable2x2.from_table(table_4_4)
        assert result.n == 40 + 20 + 15 + 35

    def test_cramer_equals_phi_for_2x2(self, table_4_4):
        result = ContingencyTable2x2.from_table(table_4_4)
        assert result.cramer_v == pytest.approx(result.phi, abs=1e-10)

    def test_confidence_level_stored(self, table_4_4):
        result = ContingencyTable2x2.from_table(table_4_4, confidence_level=0.99)
        assert result.confidence_level == 0.99

    def test_via_esek_import(self, table_4_4):
        """ContingencyTable2x2 is accessible from esek top-level."""
        from esek import ContingencyTable2x2 as CT
        result = CT.from_table(table_4_4)
        assert isinstance(result.phi, float)


# ============================================================
# Nominal/Ordinal correlation classes (smoke tests)
# ============================================================


class TestNominalByNominal:
    """Smoke tests for NominalByNominal."""

    def test_import(self):
        from esek.calculators.correlations import NominalByNominal
        assert NominalByNominal is not None

    def test_has_from_contingency_table(self):
        from esek.calculators.correlations import NominalByNominal
        assert hasattr(NominalByNominal, "from_contingency_table")


class TestPearsonCorrelationModule:
    """Additional regression tests against scipy.stats.pearsonr."""

    def test_r_matches_scipy(self):
        from scipy.stats import pearsonr
        rng = np.random.default_rng(7)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 1, 50)
        result = PearsonCorrelation.from_data(x, y, bootstrap_n_resamples=200, bootstrap_random_state=7)
        r_scipy, p_scipy = pearsonr(x, y)
        assert result.r == pytest.approx(r_scipy, abs=1e-10)
        assert result.p_value == pytest.approx(p_scipy, abs=1e-10)

    def test_fisher_ci_matches_manual(self):
        """Manually compute Fisher CI and compare."""
        from scipy.stats import norm
        rng = np.random.default_rng(13)
        x = rng.normal(0, 1, 100)
        y = 0.4 * x + rng.normal(0, 1, 100)
        result = PearsonCorrelation.from_data(x, y)
        r = result.r
        n = result.n
        zr = 0.5 * math.log((1 + r) / (1 - r))
        se = 1.0 / math.sqrt(n - 3)
        z_crit = norm.ppf(0.975)
        expected_lo = math.tanh(zr - z_crit * se)
        expected_hi = math.tanh(zr + z_crit * se)
        assert result.ci_fisher[0] == pytest.approx(expected_lo, abs=1e-10)
        assert result.ci_fisher[1] == pytest.approx(expected_hi, abs=1e-10)
