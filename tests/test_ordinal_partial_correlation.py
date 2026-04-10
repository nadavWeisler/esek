"""Tests for the Ordinal Partial Correlation module."""

import numpy as np
import pandas as pd
import pytest

from esek.calculator.correlations.ordinal_partial_correlation import OrdinalPartialCorrelation


def _make_data(n=20, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 8, size=n).astype(float)
    cov = rng.standard_normal(n)
    y = 0.5 * x + 0.3 * cov + rng.standard_normal(n)
    return pd.DataFrame(
        {
            "independent_variable": x,
            "dependnent_variable": y,
            "covariate": cov,
        }
    )


def test_from_data_returns_dict():
    """OrdinalPartialCorrelation.from_data returns a dict with expected keys."""
    data = _make_data()
    params = {"Data": data, "Confidence Level": 95}
    result = OrdinalPartialCorrelation.from_data(params)
    assert isinstance(result, dict)
    assert "Partial Correlation" in result
    assert "Semi Partial Correlation" in result
    assert "p-value partial correlation" in result
    assert "p-value Semi partial correlation" in result
    assert "Sample Size" in result


def test_from_data_partial_correlation_range():
    """Partial correlation is in [-1, 1]."""
    data = _make_data()
    params = {"Data": data, "Confidence Level": 95}
    result = OrdinalPartialCorrelation.from_data(params)
    assert -1.0 <= result["Partial Correlation"] <= 1.0


def test_from_data_ci_included():
    """Confidence intervals are returned."""
    data = _make_data()
    params = {"Data": data, "Confidence Level": 95}
    result = OrdinalPartialCorrelation.from_data(params)
    ci = result["Confidence Intervals Partial Correlation"]
    assert isinstance(ci, list)
    assert len(ci) == 2
    assert ci[0] <= ci[1]


def test_from_data_sample_size():
    """Sample size matches the data provided."""
    n = 30
    data = _make_data(n=n)
    params = {"Data": data, "Confidence Level": 95}
    result = OrdinalPartialCorrelation.from_data(params)
    assert result["Sample Size"] == n


def test_from_data_statistical_lines():
    """Statistical line strings are present in the result."""
    data = _make_data()
    params = {"Data": data, "Confidence Level": 95}
    result = OrdinalPartialCorrelation.from_data(params)
    assert "Statistical Line Partial Correlation" in result
    assert "Statistical Line Semi Partial Correlation" in result
    assert isinstance(result["Statistical Line Partial Correlation"], str)
