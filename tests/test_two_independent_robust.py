"""Tests for the Two Independent Robust calculator."""

from src.esek.Calculator.two_independent_mean.two_independent_robust import (
    TwoIndependentRobustResults,
    TwoIndependentRobustTests,
)


def test_two_independent_robust_from_score():
    """Test Two Independent Robust from score."""
    assert True, "Method not implemented yet"


def test_two_independent_robust_from_parameters():
    """Test Two Independent Robust from parameters."""
    assert True, "Method not implemented yet"

def test_two_independent_robust_from_data():
    """Test Two Independent Robust from data."""
    data = {
        "columns": [[1, 2, 3], [4, 5, 6]],
        "reps": 10000,
        "trimming_level": 0.2,
        "population_difference": 0.2,
        "confidence_level": 0.95,
    }
    results = TwoIndependentRobustTests.from_data(**data)

    assert isinstance(results, TwoIndependentRobustResults)
