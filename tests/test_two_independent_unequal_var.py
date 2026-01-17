"""Tests for the Two Independent Unequal Variance calculator."""

from src.esek.calculator.two_independent_mean.two_independent_unequal_var import (
    TwoIndependentUnequalVarResults,
    TwoIndependentUnequalVarTests,
)


def test_two_independent_unequal_var_from_score():
    """Test Two Independent Unequal Var from score."""
    assert True, "Method not implemented yet"


def test_two_independent_unequal_var_from_parameters():
    """Test Two Independent Unequal Var from parameters."""
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
    results = TwoIndependentUnequalVarTests.from_parameters(**data)

    assert isinstance(results, TwoIndependentUnequalVarResults)


def test_two_independent_unequal_var_from_data():
    """Test Two Independent Unequal Var from data."""
    data = {
        "columns": [[1, 2, 3], [4, 5, 6]],
        "population_mean_diff": 0,
        "confidence_level": 0.95,
    }
    results = TwoIndependentUnequalVarTests.from_data(**data)

    assert isinstance(results, TwoIndependentUnequalVarResults)
