"""Tests for the Ordinal by Interval correlation module."""

import numpy as np
import pytest

from esek.calculator.correlations.ordinal_by_interval import OrdinalByInterval


_PARAMS_DATA = {
    "Variable 1": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float),
    "Variable 2": np.array([1.5, 3.0, 2.5, 5.0, 4.5, 6.0, 7.5, 8.0, 9.5, 11.0]),
    "Confidence Level": 95,
    "Number Of Bootstraps Samples": 50,
}


def test_from_data_returns_dict():
    """OrdinalByInterval.from_data returns a dict with expected keys."""
    result = OrdinalByInterval.from_data(_PARAMS_DATA)
    assert isinstance(result, dict)
    assert "Spearman Correlation" in result
    assert "Gaussian Rank Correlation" in result
    assert "Skipped Correlation" in result
    assert "Ginni's Gamma" in result
    assert "Shepherd's Pi" in result


def test_from_data_spearman_is_string():
    """Spearman result is a formatted string."""
    result = OrdinalByInterval.from_data(_PARAMS_DATA)
    assert isinstance(result["Spearman Correlation"], str)
    assert "Spearman" in result["Spearman Correlation"]


def test_from_contingency_table_returns_dict():
    """OrdinalByInterval.from_contingency_table returns expected structure."""
    ct = np.array([[5, 3, 1], [2, 4, 6], [1, 3, 8]])
    params = {
        "Contingency Table": ct,
        "Confidence Level": 95,
        "Number Of Bootstraps Samples": 50,
    }
    result = OrdinalByInterval.from_contingency_table(params)
    assert isinstance(result, dict)
    assert "Contingency Table" in result
    assert "Spearman Correlation" in result
    assert "Gaussian Rank Correlation" in result
    assert "Ginni's Gamma" in result
    assert "Shepherd's Pi" in result
