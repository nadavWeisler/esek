"""Tests for the Two Independent Z calculator."""

from src.esek.Calculator.two_independent_mean.two_independent_z import (
    TwoIndependentZResults,
    TwoIndependentZTests,
)


def test_two_independent_z_from_score():
    """Test Two Independent Z from score."""

    data = {
        "z_score": 2.5,
        "sample_size_1": 30,
        "sample_size_2": 30,
        "confidence_level": 0.95,
    }
    results = TwoIndependentZTests.from_score(**data)

    assert isinstance(results, TwoIndependentZResults)


def test_two_independent_z_from_parameters():
    """Test Two Independent Z from parameters."""

    data = {
        "sample_mean_1": 5.0,
        "sample_mean_2": 4.0,
        "population_sd_1": 1.0,
        "population_sd_2": 1.0,
        "sample_size_1": 30,
        "sample_size_2": 30,
        "population_diff": 0,
        "confidence_level": 0.95,
    }
    results = TwoIndependentZTests.from_parameters(**data)

    assert isinstance(results, TwoIndependentZResults)


def test_two_independent_z_from_data():
    """Test Two Independent Z from data."""
    data = {
        "columns": [[1, 2, 3], [4, 5, 6]],
        "population_diff": 0,
        "population_sd_1": 1.0,
        "population_sd_2": 1.0,
        "confidence_level": 0.95,
    }
    results = TwoIndependentZTests.from_data(**data)

    assert isinstance(results, TwoIndependentZResults)
