"""Tests for median-based effect size calculators.

Covers:
- OneSampleMedian
- TwoPairedMedians
- TwoIndependentMedians
- MultipleDependentMedians
- nonparametric CI helpers (sign_test_ci, hodges_lehmann_ci, wilcoxon_location_ci)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from esek.calculators.medians import (
    MultipleDependentMedians,
    OneSampleMedian,
    TwoIndependentMedians,
    TwoPairedMedians,
)
from esek.calculators.medians.one_sample_median import OneSampleMedianResult
from esek.calculators.medians.two_paired_medians import TwoPairedMediansResult
from esek.calculators.medians.two_independent_medians import TwoIndependentMediansResult
from esek.calculators.medians.multiple_dependent_medians import MultipleDependentMediansResult
from esek.utils.nonparametric_ci import (
    hodges_lehmann_ci,
    sign_test_ci,
    wilcoxon_location_ci,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def normal_sample() -> np.ndarray:
    np.random.seed(0)
    return np.random.normal(loc=5.0, scale=1.5, size=40)


@pytest.fixture
def skewed_sample() -> np.ndarray:
    np.random.seed(1)
    return np.random.exponential(scale=2.0, size=30)


@pytest.fixture
def paired_samples() -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(2)
    g1 = np.random.normal(5.0, 1.5, 30)
    g2 = g1 * 0.8 + np.random.normal(0, 0.5, 30)
    return g1, g2


@pytest.fixture
def two_groups() -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(3)
    return np.random.normal(0.0, 1.0, 25), np.random.normal(0.5, 1.0, 25)


# ─────────────────────────────────────────────────────────────────────────────
# Nonparametric CI helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestSignTestCI:
    def test_returns_four_values(self, normal_sample: np.ndarray) -> None:
        result = sign_test_ci(normal_sample, mu0=0.0)
        assert len(result) == 4

    def test_ci_contains_true_median(self, normal_sample: np.ndarray) -> None:
        true_median = float(np.median(normal_sample))
        _, _, lo, hi = sign_test_ci(normal_sample, mu0=0.0, confidence_level=0.95)
        assert lo < true_median < hi

    def test_p_value_in_unit_interval(self, normal_sample: np.ndarray) -> None:
        _, p, _, _ = sign_test_ci(normal_sample, mu0=0.0)
        assert 0.0 <= p <= 1.0

    def test_rejects_far_off_null(self, normal_sample: np.ndarray) -> None:
        # True median ≈ 5; testing against mu0=100 should give very small p
        _, p, _, _ = sign_test_ci(normal_sample, mu0=100.0)
        assert p < 0.01

    def test_fails_to_reject_null_near_median(self, normal_sample: np.ndarray) -> None:
        med = float(np.median(normal_sample))
        _, p, _, _ = sign_test_ci(normal_sample, mu0=med)
        assert p > 0.05

    def test_90_percent_ci_narrower_than_95(self, normal_sample: np.ndarray) -> None:
        _, _, lo_90, hi_90 = sign_test_ci(normal_sample, mu0=0.0, confidence_level=0.90)
        _, _, lo_95, hi_95 = sign_test_ci(normal_sample, mu0=0.0, confidence_level=0.95)
        assert (hi_90 - lo_90) <= (hi_95 - lo_95)


class TestHodgesLehmannCI:
    def test_returns_three_values(self, normal_sample: np.ndarray) -> None:
        result = hodges_lehmann_ci(normal_sample)
        assert len(result) == 3

    def test_point_estimate_near_median(self, normal_sample: np.ndarray) -> None:
        est, _, _ = hodges_lehmann_ci(normal_sample)
        sample_median = float(np.median(normal_sample))
        # Pseudomedian ≈ median for symmetric distributions
        assert abs(est - sample_median) < 1.0

    def test_ci_contains_point_estimate(self, normal_sample: np.ndarray) -> None:
        est, lo, hi = hodges_lehmann_ci(normal_sample)
        assert lo <= est <= hi

    def test_ci_width_decreases_with_n(self) -> None:
        np.random.seed(42)
        small = np.random.normal(0, 1, 15)
        large = np.random.normal(0, 1, 100)
        _, lo_s, hi_s = hodges_lehmann_ci(small)
        _, lo_l, hi_l = hodges_lehmann_ci(large)
        assert (hi_l - lo_l) < (hi_s - lo_s)


class TestWilcoxonLocationCI:
    def test_returns_four_values(self, normal_sample: np.ndarray) -> None:
        result = wilcoxon_location_ci(normal_sample, mu0=0.0)
        assert len(result) == 4

    def test_stat_and_p_value_types(self, normal_sample: np.ndarray) -> None:
        stat, p, _, _ = wilcoxon_location_ci(normal_sample, mu0=0.0)
        assert isinstance(stat, float)
        assert 0 <= p <= 1.0

    def test_ci_contains_estimate(self, normal_sample: np.ndarray) -> None:
        _, _, lo, hi = wilcoxon_location_ci(normal_sample, mu0=0.0, confidence_level=0.95)
        est, _, _ = hodges_lehmann_ci(normal_sample, confidence_level=0.95)
        assert lo <= est <= hi

    def test_approx_and_exact_similar(self, normal_sample: np.ndarray) -> None:
        _, _, lo_ap, hi_ap = wilcoxon_location_ci(normal_sample, method="approx")
        _, _, lo_ex, hi_ex = wilcoxon_location_ci(normal_sample, method="exact")
        # Should be within 0.5 of each other
        assert abs(lo_ap - lo_ex) < 0.5
        assert abs(hi_ap - hi_ex) < 0.5


# ─────────────────────────────────────────────────────────────────────────────
# OneSampleMedian
# ─────────────────────────────────────────────────────────────────────────────

class TestOneSampleMedian:
    def test_returns_correct_type(self, normal_sample: np.ndarray) -> None:
        result = OneSampleMedian.from_data(normal_sample, population_median=0.0)
        assert isinstance(result, OneSampleMedianResult)

    def test_sample_size_correct(self, normal_sample: np.ndarray) -> None:
        result = OneSampleMedian.from_data(normal_sample, population_median=0.0)
        assert result.sample_size == len(normal_sample)

    def test_median_correct(self, normal_sample: np.ndarray) -> None:
        result = OneSampleMedian.from_data(normal_sample, population_median=0.0)
        assert abs(result.median - float(np.median(normal_sample))) < 1e-9

    def test_effect_sizes_keys(self, normal_sample: np.ndarray) -> None:
        result = OneSampleMedian.from_data(normal_sample, population_median=0.0)
        expected = {"delta_iqr", "delta_mad", "delta_mad_corrected", "delta_bw",
                    "delta_s", "delta_qn", "median_shift"}
        assert expected.issubset(set(result.effect_sizes.keys()))

    def test_effect_sizes_are_numeric(self, normal_sample: np.ndarray) -> None:
        result = OneSampleMedian.from_data(normal_sample, population_median=0.0)
        for key, val in result.effect_sizes.items():
            assert isinstance(val, float), f"{key} is not float: {val!r}"

    def test_ci_keys_present(self, normal_sample: np.ndarray) -> None:
        result = OneSampleMedian.from_data(normal_sample, population_median=0.0,
                                           n_bootstrap=100)
        ci = result.confidence_intervals
        assert "median_hodges_lehmann" in ci
        assert "median_sign_test_exact" in ci
        assert "median_wilcoxon_exact" in ci
        assert "median_mad_based" in ci
        assert "median_bootstrap_basic" in ci

    def test_ci_hodges_lehmann_contains_median(self, normal_sample: np.ndarray) -> None:
        result = OneSampleMedian.from_data(normal_sample, population_median=0.0,
                                           n_bootstrap=100)
        lo, hi = result.confidence_intervals["median_hodges_lehmann"]
        med = result.pseudo_median
        assert lo <= med <= hi

    def test_inferential_keys_present(self, normal_sample: np.ndarray) -> None:
        result = OneSampleMedian.from_data(normal_sample, population_median=0.0,
                                           n_bootstrap=100)
        inf = result.inferential
        assert "p_wilcoxon_exact" in inf
        assert "p_sign_exact" in inf
        assert "t_mad_based" in inf

    def test_sign_test_rejects_far_null(self, normal_sample: np.ndarray) -> None:
        # Median of normal_sample ≈ 5; testing against 100 should reject
        result = OneSampleMedian.from_data(normal_sample, population_median=100.0,
                                           n_bootstrap=100)
        assert result.inferential["p_sign_exact"] < 0.05

    def test_dispersion_keys(self, normal_sample: np.ndarray) -> None:
        result = OneSampleMedian.from_data(normal_sample, population_median=0.0,
                                           n_bootstrap=100)
        assert "iqr" in result.dispersion
        assert "median_absolute_deviation" in result.dispersion
        assert "qn" in result.dispersion

    def test_too_few_observations_raises(self) -> None:
        with pytest.raises(ValueError, match="3"):
            OneSampleMedian.from_data(np.array([1.0, 2.0]), population_median=0.0)

    def test_invalid_confidence_level_raises(self, normal_sample: np.ndarray) -> None:
        with pytest.raises(ValueError):
            OneSampleMedian.from_data(normal_sample, confidence_level=1.5)

    def test_effect_sizes_zero_when_median_equals_null(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = OneSampleMedian.from_data(data, population_median=3.0, n_bootstrap=50)
        # delta_s should be (3-3)/sd = 0
        assert abs(result.effect_sizes["delta_s"]) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# TwoPairedMedians
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoPairedMedians:
    def test_returns_correct_type(self, paired_samples: tuple) -> None:
        g1, g2 = paired_samples
        result = TwoPairedMedians.from_data(g1, g2, n_bootstrap=100)
        assert isinstance(result, TwoPairedMediansResult)

    def test_sample_size_correct(self, paired_samples: tuple) -> None:
        g1, g2 = paired_samples
        result = TwoPairedMedians.from_data(g1, g2, n_bootstrap=100)
        assert result.sample_size == len(g1)

    def test_effect_sizes_keys(self, paired_samples: tuple) -> None:
        g1, g2 = paired_samples
        result = TwoPairedMedians.from_data(g1, g2, n_bootstrap=100)
        expected = {"delta_iqr", "delta_mad", "delta_mad_corrected",
                    "delta_bw", "delta_s", "delta_qn", "median_shift"}
        assert expected.issubset(set(result.effect_sizes.keys()))

    def test_ci_keys_present(self, paired_samples: tuple) -> None:
        g1, g2 = paired_samples
        result = TwoPairedMedians.from_data(g1, g2, n_bootstrap=100)
        ci = result.confidence_intervals
        assert "diff_median_hodges_lehmann" in ci
        assert "diff_wilcoxon_exact" in ci
        assert "diff_sign_test" in ci
        assert "diff_price_bonett" in ci
        assert "diff_mad_based" in ci

    def test_group_stats_populated(self, paired_samples: tuple) -> None:
        g1, g2 = paired_samples
        result = TwoPairedMedians.from_data(g1, g2, n_bootstrap=100)
        assert "median" in result.group1_stats
        assert "sd" in result.group2_stats

    def test_unequal_lengths_raises(self) -> None:
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="same length"):
            TwoPairedMedians.from_data(g1, g2, n_bootstrap=10)

    def test_too_few_pairs_raises(self) -> None:
        g1 = np.array([1.0, 2.0])
        g2 = np.array([2.0, 3.0])
        with pytest.raises(ValueError, match="3"):
            TwoPairedMedians.from_data(g1, g2, n_bootstrap=10)

    def test_inferential_keys_present(self, paired_samples: tuple) -> None:
        g1, g2 = paired_samples
        result = TwoPairedMedians.from_data(g1, g2, n_bootstrap=100)
        inf = result.inferential
        assert "p_wilcoxon_exact" in inf
        assert "p_sign_exact" in inf

    def test_price_bonett_ci_finite(self, paired_samples: tuple) -> None:
        g1, g2 = paired_samples
        result = TwoPairedMedians.from_data(g1, g2, n_bootstrap=100)
        lo, hi = result.confidence_intervals["diff_price_bonett"]
        assert math.isfinite(lo) and math.isfinite(hi)
        assert lo < hi


# ─────────────────────────────────────────────────────────────────────────────
# TwoIndependentMedians
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoIndependentMedians:
    def test_returns_correct_type(self, two_groups: tuple) -> None:
        g1, g2 = two_groups
        result = TwoIndependentMedians.from_data(g1, g2)
        assert isinstance(result, TwoIndependentMediansResult)

    def test_effect_sizes_present(self, two_groups: tuple) -> None:
        g1, g2 = two_groups
        result = TwoIndependentMedians.from_data(g1, g2)
        es = result.effect_sizes
        assert "d_mdns" in es
        assert "d_mad_pooled" in es
        assert "quantile_symmetric" in es

    def test_d_mdns_direction(self) -> None:
        """If group1 > group2, d_mdns should be positive."""
        g1 = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        g2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = TwoIndependentMedians.from_data(g1, g2)
        assert result.effect_sizes["d_mdns"] > 0

    def test_ci_present(self, two_groups: tuple) -> None:
        g1, g2 = two_groups
        result = TwoIndependentMedians.from_data(g1, g2)
        assert "diff_price_bonett" in result.confidence_intervals
        assert "ratio_price_bonett" in result.confidence_intervals
        assert "diff_hodges_lehmann" in result.confidence_intervals

    def test_price_bonett_ci_is_tuple(self, two_groups: tuple) -> None:
        g1, g2 = two_groups
        result = TwoIndependentMedians.from_data(g1, g2)
        lo, hi = result.confidence_intervals["diff_price_bonett"]
        assert lo < hi

    def test_moods_test_present(self, two_groups: tuple) -> None:
        g1, g2 = two_groups
        result = TwoIndependentMedians.from_data(g1, g2)
        assert "chi2_moods_median" in result.inferential
        assert "p_moods_median" in result.inferential
        p = result.inferential["p_moods_median"]
        assert 0.0 <= p <= 1.0

    def test_group_stats_populated(self, two_groups: tuple) -> None:
        g1, g2 = two_groups
        result = TwoIndependentMedians.from_data(g1, g2)
        assert "median" in result.group1_stats
        assert "sd" in result.group2_stats

    def test_too_few_observations_raises(self) -> None:
        with pytest.raises(ValueError):
            TwoIndependentMedians.from_data(np.array([1.0]), np.array([2.0, 3.0]))

    def test_group_sizes_can_differ(self) -> None:
        """Should handle unequal group sizes."""
        g1 = np.random.normal(0, 1, 15)
        g2 = np.random.normal(0.5, 1, 25)
        result = TwoIndependentMedians.from_data(g1, g2)
        assert isinstance(result, TwoIndependentMediansResult)


# ─────────────────────────────────────────────────────────────────────────────
# MultipleDependentMedians
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleDependentMedians:
    def test_returns_correct_type(self) -> None:
        np.random.seed(0)
        groups = {"A": np.random.normal(0, 1, 20), "B": np.random.normal(0.3, 1, 20)}
        result = MultipleDependentMedians.from_data(groups)
        assert isinstance(result, MultipleDependentMediansResult)

    def test_descriptives_dataframe_shape(self) -> None:
        groups = {f"G{i}": np.random.normal(i, 1, 15) for i in range(4)}
        result = MultipleDependentMedians.from_data(groups)
        assert result.descriptives.shape[0] == 4

    def test_descriptives_columns_present(self) -> None:
        groups = {"A": np.arange(1, 21, dtype=float), "B": np.arange(2, 22, dtype=float)}
        result = MultipleDependentMedians.from_data(groups)
        assert "median" in result.descriptives.columns
        assert "trimmed_mean" in result.descriptives.columns
        assert "mad" in result.descriptives.columns
        assert "iqr" in result.descriptives.columns

    def test_robust_anova_keys(self) -> None:
        groups = {"A": np.random.normal(0, 1, 20), "B": np.random.normal(1, 1, 20)}
        result = MultipleDependentMedians.from_data(groups)
        anova = result.robust_anova
        assert "gc_statistic" in anova
        assert "grand_trimmed_mean" in anova
        assert "trimmed_means" in anova

    def test_accepts_sequence(self) -> None:
        arrays = [np.random.normal(i, 1, 10) for i in range(3)]
        result = MultipleDependentMedians.from_data(arrays)
        assert result.descriptives.shape[0] == 3

    def test_unequal_group_lengths_raises(self) -> None:
        groups = {"A": np.array([1, 2, 3, 4, 5], dtype=float),
                  "B": np.array([1, 2, 3], dtype=float)}
        with pytest.raises(ValueError, match="same number"):
            MultipleDependentMedians.from_data(groups)

    def test_invalid_trimming_raises(self) -> None:
        groups = {"A": np.random.normal(0, 1, 20), "B": np.random.normal(0, 1, 20)}
        with pytest.raises(ValueError, match="trimming"):
            MultipleDependentMedians.from_data(groups, trimming=0.6)

    def test_gc_positive_for_different_groups(self) -> None:
        groups = {"low": np.zeros(20), "high": np.ones(20) * 10}
        result = MultipleDependentMedians.from_data(groups)
        assert result.robust_anova["gc_statistic"] > 0

    def test_gc_zero_for_identical_groups(self) -> None:
        x = np.ones(20)
        groups = {"A": x, "B": x}
        result = MultipleDependentMedians.from_data(groups)
        assert abs(result.robust_anova["gc_statistic"]) < 1e-9
