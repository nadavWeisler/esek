"""Tests for confidence interval methods."""

import math
import pytest
from esek.confidence_intervals import (
    central_ci_one_sample,
    central_ci_paired,
    central_ci_two_samples,
    fisher_z_ci,
    log_scale_ci,
)


class TestCentralCIOneSample:
    def test_returns_three_values(self):
        result = central_ci_one_sample(0.5, 30, 0.95)
        assert len(result) == 3

    def test_ci_lower_lt_upper(self):
        ci_low, ci_high, _ = central_ci_one_sample(0.5, 30, 0.95)
        assert ci_low < ci_high

    def test_effect_size_within_ci(self):
        d = 0.5
        ci_low, ci_high, _ = central_ci_one_sample(d, 30, 0.95)
        assert ci_low < d < ci_high

    def test_larger_n_gives_narrower_ci(self):
        _, _, se_small = central_ci_one_sample(0.5, 20, 0.95)
        _, _, se_large = central_ci_one_sample(0.5, 200, 0.95)
        assert se_small > se_large

    def test_higher_confidence_gives_wider_ci(self):
        ci_low_95, ci_high_95, _ = central_ci_one_sample(0.5, 30, 0.95)
        ci_low_99, ci_high_99, _ = central_ci_one_sample(0.5, 30, 0.99)
        width_95 = ci_high_95 - ci_low_95
        width_99 = ci_high_99 - ci_low_99
        assert width_99 > width_95

    def test_known_se_formula(self):
        # SE = sqrt(1/n + d^2/(2n))
        d, n = 0.5, 50
        expected_se = math.sqrt(1.0 / n + d**2 / (2.0 * n))
        _, _, se = central_ci_one_sample(d, n, 0.95)
        assert se == pytest.approx(expected_se, abs=1e-10)


class TestCentralCIPaired:
    def test_returns_nine_values(self):
        result = central_ci_paired(0.5, 30, 0.95)
        assert len(result) == 9

    def test_ci_lower_lt_upper(self):
        ci_low, ci_high, *_ = central_ci_paired(0.5, 30, 0.95)
        assert ci_low < ci_high

    def test_small_sample_raises(self):
        with pytest.raises(ValueError):
            central_ci_paired(0.5, 3, 0.95)


class TestCentralCITwoSamples:
    def test_returns_three_values(self):
        result = central_ci_two_samples(0.5, 30, 30, 0.95)
        assert len(result) == 3

    def test_ci_contains_effect_size(self):
        d = 0.5
        ci_low, ci_high, _ = central_ci_two_samples(d, 30, 30, 0.95)
        assert ci_low < d < ci_high


class TestFisherZCI:
    def test_returns_tuple(self):
        result = fisher_z_ci(0.5, 30)
        assert len(result) == 2

    def test_ci_lower_lt_upper(self):
        ci_low, ci_high = fisher_z_ci(0.5, 30)
        assert ci_low < ci_high

    def test_r_within_ci(self):
        r = 0.5
        ci_low, ci_high = fisher_z_ci(r, 30)
        assert ci_low < r < ci_high

    def test_small_n_raises(self):
        with pytest.raises(ValueError):
            fisher_z_ci(0.5, 3)

    def test_ci_bounded_to_minus_1_to_1(self):
        ci_low, ci_high = fisher_z_ci(0.99, 10)
        assert ci_low >= -1.0
        assert ci_high <= 1.0

    def test_larger_n_narrower_ci(self):
        _, ci_high_small = fisher_z_ci(0.5, 20)
        ci_low_small, _ = fisher_z_ci(0.5, 20)
        _, ci_high_large = fisher_z_ci(0.5, 200)
        ci_low_large, _ = fisher_z_ci(0.5, 200)
        width_small = ci_high_small - ci_low_small
        width_large = ci_high_large - ci_low_large
        assert width_large < width_small


class TestLogScaleCI:
    def test_returns_tuple(self):
        result = log_scale_ci(2.0, 0.3)
        assert len(result) == 2

    def test_ci_lower_lt_upper(self):
        ci_low, ci_high = log_scale_ci(2.0, 0.3)
        assert ci_low < ci_high

    def test_or_within_ci(self):
        or_ = 2.0
        ci_low, ci_high = log_scale_ci(or_, 0.3)
        assert ci_low < or_ < ci_high

    def test_or_zero_raises(self):
        with pytest.raises(ValueError):
            log_scale_ci(0.0, 0.3)

    def test_or_1_symmetric_ci(self):
        # OR=1 → log-OR=0 → CI symmetric around 1
        ci_low, ci_high = log_scale_ci(1.0, 0.5)
        assert ci_low == pytest.approx(1.0 / ci_high, rel=1e-6)
