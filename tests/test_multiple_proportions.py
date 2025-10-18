# tests/test_multiple_proportions.py
import numpy as np
import pandas as pd
import pytest

from esek.Calculator.Proportions.multiple_proportions import (
    MultipleProportions,
    CochranQResults,
    GoodnessOfFitResults
)

def test_cochran_q_trivial_equal_columns():
    # 3 identical measures across 8 subjects ⇒ Q = 0, p = 1
    X = pd.DataFrame({
        "t1": [1,0,1,1,0,1,0,0],
        "t2": [1,0,1,1,0,1,0,0],
        "t3": [1,0,1,1,0,1,0,0],
    })
    res: CochranQResults = MultipleProportions.cochran_q_from_wide_binary(X)
    assert isinstance(res, CochranQResults)
    assert np.isclose(res.q_stat, 0.0, atol=1e-12)
    assert np.isclose(res.p_value, 1.0, atol=1e-12)
    # effect size defined; in this trivial case variance=mean=0 → r handled as 0.0
    assert 0.0 <= res.r_effect_size <= 1.0

def test_gof_perfect_fit_expected_proportions():
    # counts 50,30,20 with expected proportions [0.5,0.3,0.2] → perfect fit
    data = (["A"]*50) + (["B"]*30) + (["C"]*20)
    res: GoodnessOfFitResults = MultipleProportions.goodness_of_fit_from_frequency(
        data,
        expected_proportions=[0.5, 0.3, 0.2],
    )
    assert isinstance(res, GoodnessOfFitResults)
    assert np.isclose(res.chi_square, 0.0, atol=1e-9)
    assert np.isclose(res.wilks_g, 0.0, atol=1e-9)
    assert np.isclose(res.cohens_w, 0.0, atol=1e-9)
    # p-value ~ 1
    assert res.p_value_chi_square > 0.99
