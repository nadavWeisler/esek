"""Tests for the Two Independent Aparametric calculator."""

from src.esek.Calculator.two_independent_mean.two_independent_aparametric import (
    TwoIndependentAparametricResults,
    TwoIndependentAparametricTests,
)


def test_two_independent_aparametric_from_score():
    """Test Two Independent Aparametric from score."""
    assert True, "Method not implemented yet"


def test_two_independent_aparametric_from_parameters():
    """Test Two Independent Aparametric from parameters."""
    assert True, "Method not implemented yet"


def test_two_independent_aparametric_from_data():
    """Test Two Independent Aparametric from data."""
    data = {
        "columns": [[1, 2, 3], [4, 5, 6]],
        "confidence_level": 0.95,
    }
    results = TwoIndependentAparametricTests.from_data(**data)

    assert isinstance(results, TwoIndependentAparametricResults)
