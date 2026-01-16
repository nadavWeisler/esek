"""Tests for the Two Paired Common Language calculator."""

from src.esek.Calculator.two_paired_mean.two_paired_common_lang import (
    TwoPairedCommonLangResults,
    TwoPairedCommonLangTests,
)


def test_two_paired_common_lang_from_score():
    """Test Two Paired Common Language from t-score."""

    data = {
        "t_score": 1.96,
        "sample_size": 30,
        "confidence_level": 0.95,
    }
    results = TwoPairedCommonLangTests.from_score(**data)

    assert isinstance(results, TwoPairedCommonLangResults)


def test_two_paired_common_lang_from_parameters():
    """Test Two Paired Common Language from parameters."""

    data = {
        "sample_mean_1": 100,
        "sample_mean_2": 75,
        "sample_sd_1": 10,
        "sample_sd_2": 7.5,
        "sample_size": 30,
        "population_mean_diff": 5,
        "correlation": 0.8,
        "confidence_level": 0.95,
    }
    results = TwoPairedCommonLangTests.from_parameters(**data)

    assert isinstance(results, TwoPairedCommonLangResults)


def test_two_paired_common_lang_from_data():
    """Test Two Paired Common Language from data."""

    data = {
        "column": [[1, 2, 3], [4, 5, 6]],
        "reps": 1000,
        "confidence_level": 0.95,
    }
    results = TwoPairedCommonLangTests.from_data(**data)

    assert isinstance(results, TwoPairedCommonLangResults)
