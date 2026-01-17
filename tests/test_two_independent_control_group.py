"""Tests for the Two Independent Control Group calculator."""

from src.esek.calculator.two_independent_mean.two_independent_control_group import (
    TwoIndependentControlGroupResults,
    TwoIndependentControlGroupTests,
)


def test_two_independent_control_group_from_score():
    """Test Two Independent Control Group from score."""
    assert True, "Method not implemented yet"


def test_two_independent_control_group_from_parameters():
    """Test Two Independent Control Group from parameters."""
    data = {
        "sample_mean_experimental": 5.0,
        "sample_mean_control": 4.0,
        "sample_sd_experimental": 1.0,
        "sample_sd_control": 1.0,
        "sample_size_experimental": 30,
        "sample_size_control": 30,
        "population_mean_diff": 0,
        "confidence_level": 0.95,
    }
    results = TwoIndependentControlGroupTests.from_parameters(**data)

    assert isinstance(results, TwoIndependentControlGroupResults)


def test_two_independent_control_group_from_data():
    """Test Two Independent Control Group from data."""
    data = {
        "columns": [[1, 2, 3, 4], [5, 6, 7, 8]],
        "population_mean_diff": 0,
        "confidence_level": 0.95,
    }
    results = TwoIndependentControlGroupTests.from_data(**data)

    assert isinstance(results, TwoIndependentControlGroupResults)
