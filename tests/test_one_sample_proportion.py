# tests/test_one_sample_proportion.py
import numpy as np
import pytest

from src.esek.Calculator.Proportions.one_sample_proportion import (
    OneSampleProportions,
    OneSampleProportionResults
)

def test_from_parameters_basic():
    # p̂ = 0.60, n=100, p0=0.50, CL=0.95
    res: OneSampleProportionResults = OneSampleProportions.from_parameters(
        proportion_sample=0.60,
        sample_size=100,
        population_proportion=0.50,
        confidence_level=0.95,
    )

    # structure
    assert isinstance(res, OneSampleProportionResults)
    assert res.descriptive_statistics is None or res.descriptive_statistics  # allow either
    assert res.z_test is not None

    # z-score (score test, using p0 in SE): (0.6-0.5)/sqrt(0.5*0.5/100) = 2.0
    assert np.isclose(res.z_test.score, 2.0, rtol=0, atol=1e-6)

    # p-value is one-sided sf(|z|) in your code
    assert np.isclose(res.z_test.p_value, 0.0228, rtol=1e-3)

    # Cohen’s h ~ 0.20136 for 0.60 vs 0.50
    assert res.cohens_h is not None
    assert np.isclose(res.cohens_h.value, 0.20135792079, rtol=0, atol=1e-6)

def test_proportion_of_hits_returns_results_obj():
    # π task: 20 correct out of 40 trials, 4 choices, 95% CI
    res = OneSampleProportions.proportion_of_hits(
        number_correct_answers=20,
        number_of_trials=40,
        number_of_choices=4,
        confidence_level=0.95,
    )
    # must be results object per new scheme
    assert isinstance(res, OneSampleProportionResults)
    assert res.z_test is not None
    assert res.descriptive_statistics is not None
