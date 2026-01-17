""" "Tests for One Sample Z Test."""

from src.esek.calculator.one_sample_mean.one_sample_z import (
    OneSampleZResults,
    OneSampleZTests,
)


def test_one_sample_z_from_score():
    """Test OneSampleZTests from z-score."""
    data = {
        "z_score": 1.96,
        "sample_size": 30,
        "confidence_level": 0.95,
    }
    results = OneSampleZTests.from_score(**data)
    experced_results = OneSampleZResults()
    experced_results.z_score = 1.96
    experced_results.sample_size = 30
    experced_results.p_value = 0.025
    experced_results.cohens_d = 0.025

    assert isinstance(results, OneSampleZResults)


def test_one_sample_z_from_parameters():
    """Test OneSampleZTests from parameters."""
    data = {
        "sample_mean": 100,
        "sample_size": 30,
        "population_mean": 95,
        "population_sd": 0.5,
        "confidence_level": 0.95,
    }
    results = OneSampleZTests.from_parameters(**data)
    assert isinstance(results, OneSampleZResults)


def test_one_sample_z_from_data():
    """Test OneSampleZTests from data."""
    data = {
        "column": [[1, 2, 3], [4, 5, 6]],
        "population_mean": 95,
        "population_sd": 0.5,
        "confidence_level": 0.95,
    }
    results = OneSampleZTests.from_data(**data)
    assert isinstance(results, OneSampleZResults)
