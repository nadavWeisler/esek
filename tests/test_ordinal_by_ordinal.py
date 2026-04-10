"""Tests for the Ordinal by Ordinal correlation module."""

import numpy as np
import pytest

from esek.calculator.correlations.ordinal_by_ordinal import OrdinalByOrdinal


_PARAMS_DATA = {
    "Variable 1": np.array([1, 2, 2, 3, 3, 4, 5, 5, 6, 7], dtype=float),
    "Variable 2": np.array([2, 2, 3, 3, 4, 4, 5, 6, 6, 7], dtype=float),
    "Confidence Level": 95,
    "Number Of Bootstraps Samples": 50,
}


def test_from_data_returns_dict():
    """OrdinalByOrdinal.from_data returns a dict with expected keys."""
    result = OrdinalByOrdinal.from_data(_PARAMS_DATA)
    assert isinstance(result, dict)
    assert "Spearman Correlation" in result
    assert "Gaussian Rank Correlation" in result
    assert "Skipped Correlation" in result
    assert "Gini's Gamma" in result
    assert "Shepherd's Pi" in result
    assert "The Gamma Family Measures" in result


def test_from_data_gamma_family_present():
    """Gamma-family measures are included in OrdinalByOrdinal output."""
    result = OrdinalByOrdinal.from_data(_PARAMS_DATA)
    gamma_str = result["The Gamma Family Measures"]
    assert isinstance(gamma_str, str)
    assert "Kendall's Tau" in gamma_str or "Tau" in gamma_str


def test_from_contingency_table_returns_dict():
    """OrdinalByOrdinal.from_contingency_table returns expected structure."""
    ct = np.array([[4, 2, 0], [1, 5, 2], [0, 2, 6]])
    params = {
        "Contingency Table": ct,
        "Confidence Level": 95,
        "Number Of Bootstraps Samples": 50,
    }
    result = OrdinalByOrdinal.from_contingency_table(params)
    assert isinstance(result, dict)
    assert "Contingency Table" in result
    assert "The Gamma Family Measures" in result
    assert "Spearman Correlation" in result
