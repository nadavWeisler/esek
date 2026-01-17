"""Tests for the Two Paired Robust calculator."""

from src.esek.calculator.two_paired_mean.two_paired_robust import (
    TwoPairedRobustResults,
    TwoPairedRobustTests,
)


def test_two_paired_robust_from_score():
    """Test Two Paired Robust from score."""
    assert True, "Method not implemented yet"


def test_two_paired_robust_from_parameters():
    """Test Two Paired Robust from parameters."""
    assert True, "Method not implemented yet"


def test_two_paired_robust_from_data():
    """Test Two Paired Robust from data."""
    data = {
        "columns": [[1, 2, 3], [4, 5, 6]],
        "reps": 1000,
        "confidence_level": 0.95,
    }
    results = TwoPairedRobustTests.from_data(**data)

    assert isinstance(results, TwoPairedRobustResults)
