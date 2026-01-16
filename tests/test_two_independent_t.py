"""Tests for the Two Independent T calculator."""

from src.esek.Calculator.two_independent_mean.two_independent_t import (
    TwoIndependentTResults,
    TwoIndependentTTests,
)


def test_two_independent_t_from_score():
    """Test Two Independent T from score."""
    data = {
        "t_score": 2.5,
        "sample_size_1": 30,
        "sample_size_2": 30,
        "confidence_level": 0.95,
    }
    results = TwoIndependentTTests.from_score(**data)

    assert isinstance(results, TwoIndependentTResults)


def test_two_independent_t_from_parameters():
    """Test Two Independent T from parameters."""
    data = {
        "sample_mean_1": 5.0,
        "sample_mean_2": 4.0,
        "sample_sd_1": 1.0,
        "sample_sd_2": 1.0,
        "sample_size_1": 30,
        "sample_size_2": 30,
        "population_mean_diff": 0,
        "confidence_level": 0.95,
    }
    results = TwoIndependentTTests.from_parameters(**data)

    assert isinstance(results, TwoIndependentTResults)


def test_two_independent_t_from_data():
    """Test Two Independent T from data."""
    data = {
        "columns": [[1, 2, 3], [4, 5, 6]],
        "population_mean_diff": 0,
        "confidence_level": 0.95,
    }
    results = TwoIndependentTTests.from_data(**data)

    assert isinstance(results, TwoIndependentTResults)
