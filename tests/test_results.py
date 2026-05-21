"""Tests for result objects (frozen dataclasses)."""

import pytest
from esek.results.base import EffectSizeResult, ConversionResult, ConfidenceIntervalResult
from esek.results.effect_sizes import CohensDResult, HedgesGResult, CliffsDeltaResult
from esek.results.proportions import CohenHResult, CohenGResult


class TestEffectSizeResult:
    def test_basic_construction(self):
        r = EffectSizeResult(value=0.5, effect_size_type="Cohen's d", method="from_score")
        assert r.value == 0.5
        assert r.effect_size_type == "Cohen's d"
        assert r.method == "from_score"

    def test_frozen(self):
        r = EffectSizeResult(value=0.5, effect_size_type="d", method="test")
        with pytest.raises((TypeError, AttributeError)):
            r.value = 1.0  # type: ignore[misc]

    def test_ci_property_returns_tuple(self):
        r = EffectSizeResult(value=0.5, effect_size_type="d", method="test", ci_low=0.1, ci_high=0.9)
        assert r.ci == (0.1, 0.9)

    def test_ci_property_none_when_missing(self):
        r = EffectSizeResult(value=0.5, effect_size_type="d", method="test")
        assert r.ci is None

    def test_str_contains_value(self):
        r = EffectSizeResult(value=0.5, effect_size_type="Cohen's d", method="test")
        assert "0.5" in str(r)

    def test_metadata_default_empty_dict(self):
        r = EffectSizeResult(value=0.5, effect_size_type="d", method="test")
        assert r.metadata == {}

    def test_metadata_not_shared_between_instances(self):
        r1 = EffectSizeResult(value=0.5, effect_size_type="d", method="test")
        r2 = EffectSizeResult(value=0.6, effect_size_type="d", method="test")
        assert r1.metadata is not r2.metadata


class TestCohensDResult:
    def test_construction(self):
        r = CohensDResult(value=0.5, effect_size_type="Cohen's d", method="from_parameters")
        assert isinstance(r, EffectSizeResult)

    def test_is_frozen(self):
        r = CohensDResult(value=0.5, effect_size_type="Cohen's d", method="test")
        with pytest.raises((TypeError, AttributeError)):
            r.value = 1.0  # type: ignore[misc]


class TestConversionResult:
    def test_basic_construction(self):
        r = ConversionResult(
            input_type="d",
            output_type="r",
            input_value=0.5,
            output_value=0.243,
            method="Cohen formula",
        )
        assert r.input_type == "d"
        assert r.output_type == "r"
        assert r.output_value == pytest.approx(0.243)

    def test_frozen(self):
        r = ConversionResult(
            input_type="d", output_type="r",
            input_value=0.5, output_value=0.24,
            method="test",
        )
        with pytest.raises((TypeError, AttributeError)):
            r.output_value = 0.5  # type: ignore[misc]

    def test_str_contains_types(self):
        r = ConversionResult(
            input_type="d", output_type="r",
            input_value=0.5, output_value=0.24,
            method="test",
        )
        s = str(r)
        assert "d" in s
        assert "r" in s


class TestConfidenceIntervalResult:
    def test_ci_property(self):
        r = ConfidenceIntervalResult(
            lower=0.1, upper=0.9, confidence_level=0.95, method="Wilson"
        )
        assert r.ci == (0.1, 0.9)

    def test_str_contains_pct(self):
        r = ConfidenceIntervalResult(
            lower=0.1, upper=0.9, confidence_level=0.95, method="Wilson"
        )
        assert "95%" in str(r)


class TestProportionResults:
    def test_cohen_h_result(self):
        r = CohenHResult(value=0.3, effect_size_type="Cohen's h", method="test")
        assert isinstance(r, EffectSizeResult)

    def test_cohen_g_result(self):
        r = CohenGResult(value=0.1, effect_size_type="Cohen's g", method="test")
        assert isinstance(r, EffectSizeResult)
