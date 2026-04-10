"""Tests for the Nominal by Ordinal correlation module."""

import numpy as np
import pytest

from esek.calculator.correlations.nominal_by_ordinal import NominalByOrdinal


def test_from_data_returns_dict():
    """NominalByOrdinal.from_data returns a dict with expected keys."""
    params = {
        "Nominal Variable": ["A", "A", "A", "B", "B", "B"],
        "Ordinal Variable": [1, 2, 3, 2, 3, 4],
        "Number of Bootstrapping Samples": 50,
        "Confidence Level": 95,
    }
    result = NominalByOrdinal.from_data(params)
    assert isinstance(result, dict)
    assert "H Statistic" in result
    assert "Degrees of Freedom of the Kruskal Wallis Test" in result
    assert "p-value of the Kruskal Wallis Test" in result
    assert "Epsilon Square" in result
    assert "Freeman's Theta" in result


def test_from_data_binary_nominal_includes_rank_biserial():
    """Binary nominal variable yields rank-biserial correlation."""
    params = {
        "Nominal Variable": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "Ordinal Variable": [1, 2, 2, 3, 3, 4, 4, 5],
        "Number of Bootstrapping Samples": 50,
        "Confidence Level": 95,
    }
    result = NominalByOrdinal.from_data(params)
    assert "Rank Biserial Correlation" in result
    assert isinstance(result["Rank Biserial Correlation"], float)


def test_from_data_multiclass_no_rank_biserial():
    """Multi-class nominal variable does not include rank-biserial."""
    params = {
        "Nominal Variable": ["A", "A", "B", "B", "C", "C"],
        "Ordinal Variable": [1, 2, 3, 4, 5, 6],
        "Number of Bootstrapping Samples": 50,
        "Confidence Level": 95,
    }
    result = NominalByOrdinal.from_data(params)
    assert "Rank Biserial Correlation" not in result


def test_from_contingency_table_returns_dict():
    """NominalByOrdinal.from_contingency_table returns expected structure."""
    ct = np.array([[5, 3, 2], [2, 4, 6]])
    params = {
        "Contingency Table": ct,
        "Number of Bootstrapping Samples": 50,
        "Confidence Level": 95,
    }
    result = NominalByOrdinal.from_contingency_table(params)
    assert isinstance(result, dict)
    assert "H Statistic" in result
    assert "Epsilon Square" in result
    assert "Freeman's Theta" in result
    # Binary nominal → should have rank-biserial
    assert "Rank Biserial Correlation" in result


def test_from_contingency_table_epsilon_range():
    """Epsilon Square should be in [0, 1]."""
    ct = np.array([[10, 5, 2], [3, 8, 9]])
    params = {
        "Contingency Table": ct,
        "Number of Bootstrapping Samples": 50,
        "Confidence Level": 95,
    }
    result = NominalByOrdinal.from_contingency_table(params)
    assert 0.0 <= result["Epsilon Square"] <= 1.0
