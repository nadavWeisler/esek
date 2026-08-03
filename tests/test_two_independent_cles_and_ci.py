"""Tests for completed TwoIndependentCLES entry points and association CIs."""

from __future__ import annotations

import numpy as np
import pytest

from esek.calculators.two_independent_mean import TwoIndependentCLES, TwoIndependentCLESResult
from esek.confidence_intervals import (
    CohensWCIResult,
    ContingencyCoefficientCIResult,
    cohens_w_ci,
    contingency_coefficient_ci,
)
from esek.core import InvalidInputError


class TestTwoIndependentCLES:
    def test_from_parameters_matches_from_t_score(self):
        mean1, mean2 = 10.0, 8.0
        sd1, sd2 = 2.0, 2.5
        n1, n2 = 30, 28
        df = n1 + n2 - 2
        pooled = np.sqrt((((n1 - 1) * sd1**2) + ((n2 - 1) * sd2**2)) / df)
        se = pooled * np.sqrt(1 / n1 + 1 / n2)
        t_score = (mean1 - mean2) / se

        from_t = TwoIndependentCLES.from_t_score(t_score, n1, n2)
        from_p = TwoIndependentCLES.from_parameters(mean1, mean2, sd1, sd2, n1, n2)
        assert isinstance(from_p, TwoIndependentCLESResult)
        assert from_p.t_score == pytest.approx(from_t.t_score, abs=1e-10)
        assert from_p.cohens_ds.effect_size == pytest.approx(
            from_t.cohens_ds.effect_size, abs=1e-10
        )

    def test_from_data_matches_from_parameters(self):
        rng = np.random.default_rng(0)
        g1 = rng.normal(10, 2, size=40)
        g2 = rng.normal(8, 2.5, size=35)
        from_data = TwoIndependentCLES.from_data(g1, g2)
        from_params = TwoIndependentCLES.from_parameters(
            float(np.mean(g1)),
            float(np.mean(g2)),
            float(np.std(g1, ddof=1)),
            float(np.std(g2, ddof=1)),
            len(g1),
            len(g2),
        )
        assert from_data.t_score == pytest.approx(from_params.t_score, abs=1e-10)

    def test_invalid_sample_size_raises(self):
        with pytest.raises(InvalidInputError):
            TwoIndependentCLES.from_parameters(1, 0, 1, 1, 1, 10)


class TestAssociationCIs:
    def test_cohens_w_ci_contains_estimate(self):
        result = cohens_w_ci(0.3, n=100, df=2)
        assert isinstance(result, CohensWCIResult)
        lo, hi = result.ci
        assert lo <= result.cohens_w <= hi

    def test_contingency_coefficient_ci_contains_estimate(self):
        result = contingency_coefficient_ci(0.4, n=120, df=4)
        assert isinstance(result, ContingencyCoefficientCIResult)
        lo, hi = result.ci
        assert lo <= result.contingency_coefficient <= hi

    def test_cohens_w_invalid(self):
        with pytest.raises(ValueError):
            cohens_w_ci(-0.1, n=50, df=1)

    def test_contingency_coefficient_invalid(self):
        with pytest.raises(ValueError):
            contingency_coefficient_ci(1.0, n=50, df=1)
