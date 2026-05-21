"""Tests for dispersion measure CIs (MAD, SD)."""
from __future__ import annotations

import math
import pytest
import numpy as np

from esek.confidence_intervals import mad_ci, sd_ci, MADCIResult, SDCIResult


RNG = np.random.default_rng(42)


class TestMADCI:
    def test_basic_return_type(self):
        data = RNG.normal(size=50)
        result = mad_ci(data)
        assert isinstance(result, MADCIResult)

    def test_ci_ordering(self):
        data = RNG.normal(size=80)
        result = mad_ci(data)
        assert result.ci_low <= result.mad_corrected <= result.ci_high

    def test_n_preserved(self):
        data = list(range(1, 41))
        result = mad_ci(data)
        assert result.n == 40

    def test_mad_corrected_formula(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mad_ci(data)
        n = 5
        median = 3.0
        mad = float(np.mean(np.abs(data - median)))
        expected_corrected = mad * n / (n - 1)
        assert result.mad == pytest.approx(mad, abs=1e-6)
        assert result.mad_corrected == pytest.approx(expected_corrected, abs=1e-6)

    def test_confidence_level_preserved(self):
        data = RNG.normal(size=60)
        result = mad_ci(data, confidence_level=0.99)
        assert result.confidence_level == 0.99

    def test_99_ci_wider_than_95(self):
        data = RNG.normal(size=80)
        r95 = mad_ci(data, confidence_level=0.95)
        r99 = mad_ci(data, confidence_level=0.99)
        assert r99.ci_high - r99.ci_low > r95.ci_high - r95.ci_low

    def test_larger_n_narrower_ci(self):
        data_small = RNG.normal(size=30)
        data_large = RNG.normal(size=300)
        r_small = mad_ci(data_small)
        r_large = mad_ci(data_large)
        w_small = r_small.ci_high - r_small.ci_low
        w_large = r_large.ci_high - r_large.ci_low
        # Not guaranteed by law but very likely for normal data
        assert w_large < w_small

    def test_invalid_ndim(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            mad_ci(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_too_small_n(self):
        with pytest.raises(ValueError, match="≥ 3"):
            mad_ci([1.0, 2.0])

    def test_zero_mad_raises(self):
        with pytest.raises(ValueError, match="MAD is zero"):
            mad_ci([5.0, 5.0, 5.0, 5.0, 5.0])

    def test_invalid_ci_level(self):
        data = RNG.normal(size=50)
        with pytest.raises(ValueError, match="confidence_level"):
            mad_ci(data, confidence_level=1.5)

    def test_metadata_keys(self):
        data = RNG.normal(size=50)
        result = mad_ci(data)
        assert "mean" in result.metadata
        assert "median" in result.metadata
        assert "sd" in result.metadata

    def test_list_input_accepted(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = mad_ci(data)
        assert isinstance(result, MADCIResult)


class TestSDCI:
    def test_basic_return_type(self):
        data = RNG.normal(size=50)
        result = sd_ci(data)
        assert isinstance(result, SDCIResult)

    def test_sd_matches_numpy(self):
        data = RNG.normal(size=60)
        result = sd_ci(data)
        expected_sd = float(np.std(data, ddof=1))
        assert result.sd == pytest.approx(expected_sd, abs=1e-5)

    def test_ci_contains_true_sd_large_n(self):
        # For large n, CI should reliably contain true σ=1.0
        rng = np.random.default_rng(77)
        data = rng.normal(loc=5.0, scale=1.0, size=1000)
        result = sd_ci(data)
        assert result.ci_low <= 1.0 <= result.ci_high

    def test_ci_ordering(self):
        data = RNG.normal(size=80)
        result = sd_ci(data)
        assert result.ci_low <= result.sd <= result.ci_high

    def test_variance_equals_sd_squared(self):
        data = RNG.normal(size=50)
        result = sd_ci(data)
        assert result.variance == pytest.approx(result.sd**2, rel=1e-6)

    def test_var_ci_equals_sd_ci_squared(self):
        data = RNG.normal(size=50)
        result = sd_ci(data)
        assert result.var_ci_low == pytest.approx(result.ci_low**2, rel=1e-5)
        assert result.var_ci_high == pytest.approx(result.ci_high**2, rel=1e-5)

    def test_df_is_n_minus_1(self):
        data = RNG.normal(size=40)
        result = sd_ci(data)
        assert result.df == 39
        assert result.n == 40

    def test_confidence_level_preserved(self):
        data = RNG.normal(size=50)
        result = sd_ci(data, confidence_level=0.99)
        assert result.confidence_level == 0.99

    def test_99_ci_wider_than_95(self):
        data = RNG.normal(size=80)
        r95 = sd_ci(data, confidence_level=0.95)
        r99 = sd_ci(data, confidence_level=0.99)
        assert r99.ci_high - r99.ci_low > r95.ci_high - r95.ci_low

    def test_invalid_n_too_small(self):
        with pytest.raises(ValueError, match="≥ 2"):
            sd_ci([5.0])

    def test_invalid_ci_level(self):
        data = RNG.normal(size=50)
        with pytest.raises(ValueError, match="confidence_level"):
            sd_ci(data, confidence_level=0.0)

    def test_list_input_accepted(self):
        data = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        result = sd_ci(data)
        assert isinstance(result, SDCIResult)

    def test_constant_data(self):
        # SD=0, CI should be [0, 0]
        data = [3.0] * 10
        result = sd_ci(data)
        assert result.sd == pytest.approx(0.0, abs=1e-10)
        assert result.ci_low == pytest.approx(0.0, abs=1e-10)
