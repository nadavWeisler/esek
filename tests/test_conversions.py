"""Tests for effect-size conversion functions."""

import math
import pytest
from esek import EffectSizeConverter
from esek.converters.d_conversions import d_to_r, d_to_odds_ratio, d_to_cohens_f, d_to_r_equal_n
from esek.converters.r_conversions import r_to_d, r_to_fisher_z, fisher_z_to_r
from esek.converters.odds_ratio_conversions import odds_ratio_to_d, odds_ratio_to_r
from esek.results.base import ConversionResult
from esek.core.exceptions import InvalidInputError


# ---------------------------------------------------------------------------
# d → r
# ---------------------------------------------------------------------------

class TestDToR:
    def test_returns_conversion_result(self):
        result = d_to_r(0.5, 30, 30)
        assert isinstance(result, ConversionResult)

    def test_input_output_types(self):
        result = d_to_r(0.5, 30, 30)
        assert result.input_type == "d"
        assert result.output_type == "r"

    def test_known_value_equal_n(self):
        # For equal n, d_to_r should match: r = d / sqrt(d^2 + 4)
        d = 0.5
        expected_r = d / math.sqrt(d**2 + 4)
        result = d_to_r_equal_n(d, 30)
        assert result.output_value == pytest.approx(expected_r, abs=1e-10)

    def test_zero_d_gives_zero_r(self):
        result = d_to_r(0.0, 30, 30)
        assert result.output_value == pytest.approx(0.0, abs=1e-10)

    def test_positive_d_gives_positive_r(self):
        result = d_to_r(1.0, 50, 50)
        assert result.output_value > 0

    def test_negative_d_gives_negative_r(self):
        result = d_to_r(-0.5, 30, 30)
        assert result.output_value < 0

    def test_invalid_n1_raises(self):
        with pytest.raises(InvalidInputError):
            d_to_r(0.5, 0, 30)

    def test_invalid_n2_raises(self):
        with pytest.raises(InvalidInputError):
            d_to_r(0.5, 30, -1)

    def test_converter_class_matches(self):
        direct = d_to_r(0.5, 30, 30)
        via_class = EffectSizeConverter.d_to_r(0.5, 30, 30)
        assert direct.output_value == pytest.approx(via_class.output_value)


# ---------------------------------------------------------------------------
# d → OR
# ---------------------------------------------------------------------------

class TestDToOddsRatio:
    def test_returns_conversion_result(self):
        result = d_to_odds_ratio(0.5)
        assert isinstance(result, ConversionResult)

    def test_known_value(self):
        # d=0 → OR=1
        result = d_to_odds_ratio(0.0)
        assert result.output_value == pytest.approx(1.0, abs=1e-10)

    def test_positive_d_gives_or_gt_1(self):
        result = d_to_odds_ratio(0.5)
        assert result.output_value > 1.0

    def test_negative_d_gives_or_lt_1(self):
        result = d_to_odds_ratio(-0.5)
        assert result.output_value < 1.0

    def test_log_or_metadata(self):
        result = d_to_odds_ratio(1.0)
        assert "log_OR" in result.metadata


# ---------------------------------------------------------------------------
# d → f
# ---------------------------------------------------------------------------

class TestDToCohensF:
    def test_known_value(self):
        result = d_to_cohens_f(1.0)
        assert result.output_value == pytest.approx(0.5, abs=1e-10)

    def test_zero(self):
        result = d_to_cohens_f(0.0)
        assert result.output_value == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# r → d
# ---------------------------------------------------------------------------

class TestRToD:
    def test_inverse_of_d_to_r_equal_n(self):
        d_original = 0.5
        r_result = d_to_r_equal_n(d_original, 30)
        r = r_result.output_value
        # r_to_d_equal_n should recover d
        from esek.converters.r_conversions import r_to_d_equal_n
        back = r_to_d_equal_n(r)
        assert back.output_value == pytest.approx(d_original, abs=1e-6)

    def test_r_equals_one_raises(self):
        with pytest.raises(InvalidInputError):
            r_to_d(1.0, 30, 30)

    def test_invalid_n_raises(self):
        with pytest.raises(InvalidInputError):
            r_to_d(0.3, 0, 30)


# ---------------------------------------------------------------------------
# Fisher z
# ---------------------------------------------------------------------------

class TestFisherZ:
    def test_r_zero_gives_z_zero(self):
        result = r_to_fisher_z(0.0)
        assert result.output_value == pytest.approx(0.0, abs=1e-10)

    def test_r_equals_one_raises(self):
        with pytest.raises(InvalidInputError):
            r_to_fisher_z(1.0)

    def test_round_trip(self):
        r_orig = 0.6
        z = r_to_fisher_z(r_orig).output_value
        r_back = fisher_z_to_r(z).output_value
        assert r_back == pytest.approx(r_orig, abs=1e-10)

    def test_positive_r_gives_positive_z(self):
        result = r_to_fisher_z(0.5)
        assert result.output_value > 0

    def test_known_value(self):
        # atanh(0.5) ≈ 0.5493
        result = r_to_fisher_z(0.5)
        assert result.output_value == pytest.approx(math.atanh(0.5), abs=1e-10)


# ---------------------------------------------------------------------------
# OR → d
# ---------------------------------------------------------------------------

class TestOddsRatioToD:
    def test_or_one_gives_d_zero(self):
        result = odds_ratio_to_d(1.0)
        assert result.output_value == pytest.approx(0.0, abs=1e-10)

    def test_zero_or_raises(self):
        with pytest.raises(InvalidInputError):
            odds_ratio_to_d(0.0)

    def test_negative_or_raises(self):
        with pytest.raises(InvalidInputError):
            odds_ratio_to_d(-1.0)

    def test_round_trip_with_d_to_or(self):
        d_orig = 0.5
        or_result = d_to_odds_ratio(d_orig)
        back = odds_ratio_to_d(or_result.output_value)
        assert back.output_value == pytest.approx(d_orig, abs=1e-6)


# ---------------------------------------------------------------------------
# ConversionResult str representation
# ---------------------------------------------------------------------------

class TestConversionResultStr:
    def test_str_contains_types(self):
        result = d_to_r(0.5, 30, 30)
        s = str(result)
        assert "d" in s
        assert "r" in s
