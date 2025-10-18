# tests/test_two_dependent_proportions.py
import numpy as np
import pytest

from src.esek.Calculator.Proportions.two_dependent_proportions import (
    TwoDependentProportions,
    TwoDependentProportionsResults,  # your dataclass
)


def test_paired_basic_mcnemar_and_diff_cis():
    # 2x2 paired table:
    # yes_yes=30, yes_no=10, no_yes=5, no_no=55  (n=100)
    # p1=(30+10)/100=0.40, p2=(30+5)/100=0.35, diff=0.05
    res: TwoDependentProportionsResults = TwoDependentProportions.from_frequencies(
        yes_yes=30,
        yes_no=10,
        no_yes=5,
        no_no=55,
        confidence_level=0.95
    )

    assert isinstance(res, TwoDependentProportionsResults)
    # McNemar chi-square = (|b-c|)^2/(b+c) = 25/15 = 1.6667; p ≈ 0.196
    assert np.isclose(res.mcnemar_chi2, 25 / 15, atol=1e-6)
    assert np.isclose(res.mcnemar_exact_p, 0.196, atol=2e-2)

    # difference between proportions should be ~ 0.05
    assert np.isclose(res.diff_p, 0.05, atol=1e-12)

    # Wald CI exists and is within [-1,1]
    lo, hi = res.diff_cis.ci
    assert -1.0 <= lo <= hi <= 1.0
