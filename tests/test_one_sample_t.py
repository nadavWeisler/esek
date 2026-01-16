"""Tests for One Sample T Test."""
from src.esek.Calculator.one_sample_mean.one_sample_t import (
    OneSampleTResults,
    OneSampleTTest,
)


def test_one_sample_t_from_score():
    """Test OneSampleTTest from t-score."""
    data = {
        "t_score": 1.96,
        "sample_size": 30,
        "confidence_level": 0.95,
    }
    results = OneSampleTTest.from_score(**data)

    assert isinstance(results, OneSampleTResults)


def test_one_sample_t_from_parameters():
    """Test OneSampleTTest from parameters."""
    data = {
        "sample_mean": 100,
        "sample_size": 30,
        "sample_sd": 0.5,
        "population_mean": 95,
        "confidence_level": 0.95,
    }
    results = OneSampleTTest.from_parameters(**data)

    assert isinstance(results, OneSampleTResults)


def test_one_sample_t_from_data():
    """Test OneSampleTTest from data."""
    assert True, "This test is not implemented yet"
