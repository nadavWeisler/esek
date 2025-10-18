# tests/test_two_independent_proportions.py
import numpy as np
import pytest

from esek.Calculator.Proportions.two_independent_proportions import (
    TwoIndependentProportions,
    TwoIndependentProportionsResults
)


def test_two_ind_basic_wald_z():
    # p1=0.60 (n1=100), p2=0.50 (n2=100)
    # Wald SE = sqrt(0.6*0.4/100 + 0.5*0.5/100) = sqrt(0.0049) = 0.07
    # z = 0.10 / 0.07 ≈ 1.42857 ; one-sided p ≈ 0.0764
    res: TwoIndependentProportionsResults = TwoIndependentProportions.from_parameters(
        proportion_sample_1=0.60,
        proportion_sample_2=0.50,
        sample_size_1=100,
        sample_size_2=100,
        confidence_level=0.95,
        difference_in_population=0.0,
    )

    assert isinstance(res, TwoIndependentProportionsResults)
    assert res.z_wald is not None
    assert np.isclose(res.z_wald.score, 1.428571, atol=1e-4)
    assert np.isclose(res.z_wald.p_value, 0.0764, atol=5e-3)

    # Cohen's h between the two samples ~ |2*asin(sqrt(.6))-2*asin(sqrt(.5))| ≈ 0.20136
    assert res.cohens_h is not None
    assert np.isclose(res.cohens_h.value, 0.20135792079, atol=1e-6)
