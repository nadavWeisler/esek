"""Tests for the Nominal by Nominal correlation module."""

import numpy as np
import pytest

from esek.calculator.correlations.nominal_by_nominal import NominalByNominal


def test_from_chi_score_returns_dict():
    """NominalByNominal.from_chi_score returns a dict with expected keys."""
    params = {
        "Chi Sqaure Score": 9.5,
        "Sample Size": 100,
        "Degrees of Freedom": 2,
        "Confidence Level": 95,
    }
    result = NominalByNominal.from_chi_score(params)
    assert isinstance(result, dict)
    assert "Cohen's w / Phi" in result
    assert "Cramer's V" in result
    assert "Contingency Coefficient" in result
    assert "p-value" in result


def test_from_chi_score_values():
    """NominalByNominal.from_chi_score produces correct Cramér's V."""
    params = {
        "Chi Sqaure Score": 10.0,
        "Sample Size": 100,
        "Degrees of Freedom": 1,
        "Confidence Level": 95,
    }
    result = NominalByNominal.from_chi_score(params)
    expected_w = np.sqrt(10.0 / 100)
    assert abs(result["Cohen's w / Phi"] - round(expected_w, 4)) < 1e-3


def test_from_contingency_table_returns_dict():
    """NominalByNominal.from_contingency_table returns expected structure."""
    ct = np.array([[20, 30], [15, 35]])
    params = {"Contingency Table": ct, "Confidence Level": 95}
    result = NominalByNominal.from_contingency_table(params)
    assert isinstance(result, dict)
    assert "Lambda Table" in result
    assert "Tau Table" in result
    assert "Nominal by Nominal Association" in result


def test_from_data_returns_dict():
    """NominalByNominal.from_data returns expected structure."""
    params = {
        "Column 1": np.array(["A", "A", "B", "B", "C", "C", "A", "B"]),
        "Column 2": np.array(["X", "Y", "X", "Y", "X", "Y", "X", "Y"]),
        "Confidence Level": 95,
    }
    result = NominalByNominal.from_data(params)
    assert isinstance(result, dict)
    assert "Contingency Table" in result
    assert "Lambda Table" in result
    assert "Tau Table" in result
