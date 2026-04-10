"""Tests for the Nominal by Interval correlation module."""

import numpy as np
import pytest

from esek.calculator.correlations.nominal_by_interval import NominalByInterval


def test_from_data_returns_dict():
    """NominalByInterval.from_data returns a dict with expected keys."""
    params = {
        "Nominal": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        "Interval": np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]),
        "Confidence Level": 95,
    }
    result = NominalByInterval.from_data(params)
    assert isinstance(result, dict)
    assert "eta_output" in result
    assert "Point Biserial Correlation" in result


def test_from_data_binary_nominal_has_point_biserial():
    """Binary nominal variable yields a point-biserial correlation string."""
    params = {
        "Nominal": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        "Interval": np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        "Confidence Level": 95,
    }
    result = NominalByInterval.from_data(params)
    # For a binary nominal, point-biserial should be a string (formatted results)
    assert isinstance(result["Point Biserial Correlation"], str)
    assert "Point Biserial Correlation" in result["Point Biserial Correlation"]


def test_from_data_multiclass_nominal_message():
    """Multi-class nominal variable returns info message for point-biserial."""
    params = {
        "Nominal": np.array([0, 0, 1, 1, 2, 2]),
        "Interval": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "Confidence Level": 95,
    }
    result = NominalByInterval.from_data(params)
    assert "Point Biserial Correlation only relevant" in result["Point Biserial Correlation"]


def test_from_contingency_table_returns_dict():
    """NominalByInterval.from_contingency_table returns expected structure."""
    ct = np.array([[5, 3, 2], [2, 4, 6]])
    params = {"Table": ct, "Confidence Level": 95}
    result = NominalByInterval.from_contingency_table(params)
    assert isinstance(result, dict)
    assert "eta_output" in result
    assert "Point Biserial Correlation" in result
