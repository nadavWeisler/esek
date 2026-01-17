"""Tests for the Two Paired Aparametric calculator."""

from src.esek.calculator.two_paired_mean.two_paired_aparametric import (
    TwoPairedAparametricResults,
    TwoPairedAparametricTests,
)


def test_two_paired_aparametric_from_score():
    """Test Two Paired Aparametric from score."""
    assert True, "Method not implemented yet"


def test_two_paired_aparametric_from_parameters():
    """Test Two Paired Aparametric from parameters."""
    assert True, "Method not implemented yet"


def test_two_paired_aparametric_from_data():
    """Test Two Paired Aparametric from data."""
    data = {
        "columns": [[1, 2, 3], [4, 5, 6]],
        "population_difference": 0.5,
        "confidence_level": 0.95,
    }
    results = TwoPairedAparametricTests.from_data(**data)

    assert isinstance(results, TwoPairedAparametricResults)
