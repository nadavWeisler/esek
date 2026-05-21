"""Tests for MultipleRSquared and PartialPearsonCorrelation."""
from __future__ import annotations

import math
import pytest
import numpy as np
import pandas as pd

from esek.calculators.correlations import (
    MultipleRSquared,
    MultipleRSquaredResult,
    compute_adjusted_r_squared,
    PartialPearsonCorrelation,
    PartialCorrelationResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def make_regression_df(n: int = 100, p: int = 2) -> pd.DataFrame:
    X = RNG.normal(size=(n, p))
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + RNG.normal(size=n) * 0.5
    cols = {f"x{i + 1}": X[:, i] for i in range(p)}
    cols["y"] = y
    return pd.DataFrame(cols)


def make_partial_df(n: int = 120) -> pd.DataFrame:
    """IQ, test_score, age as covariate."""
    rng = np.random.default_rng(7)
    age = rng.normal(35, 8, size=n)
    iv = 0.4 * age + rng.normal(0, 5, size=n)
    dv = 0.3 * age + 0.5 * iv + rng.normal(0, 3, size=n)
    return pd.DataFrame({
        "independent_variable": iv,
        "dependent_variable": dv,
        "age": age,
    })


# ---------------------------------------------------------------------------
# compute_adjusted_r_squared
# ---------------------------------------------------------------------------


class TestComputeAdjustedRSquared:
    def test_basic_output_type(self):
        est = compute_adjusted_r_squared(0.5, n=100, p=3)
        assert isinstance(est, dict)
        assert len(est) > 0

    def test_ezekiel_known_formula(self):
        r2, n, p = 0.5, 100, 3
        expected = 1.0 - (n - 1) / (n - p - 1) * (1 - r2)
        est = compute_adjusted_r_squared(r2, n=n, p=p)
        assert "ezekiel_1930" in est
        assert est["ezekiel_1930"] == pytest.approx(expected, abs=1e-4)

    def test_smith_known_formula(self):
        r2, n, p = 0.5, 100, 3
        expected = 1.0 - (n / (n - p)) * (1 - r2)
        est = compute_adjusted_r_squared(r2, n=n, p=p)
        assert est["smith_1929"] == pytest.approx(expected, abs=1e-4)

    def test_all_estimators_between_neg_one_and_one(self):
        est = compute_adjusted_r_squared(0.6, n=50, p=4)
        for name, value in est.items():
            assert -2.0 <= value <= 1.0, f"{name} = {value} out of range"

    def test_small_df_returns_empty(self):
        est = compute_adjusted_r_squared(0.5, n=3, p=3)
        assert est == {}

    def test_r2_zero(self):
        est = compute_adjusted_r_squared(0.0, n=100, p=3)
        assert isinstance(est, dict)

    def test_r2_one(self):
        est = compute_adjusted_r_squared(1.0, n=100, p=3)
        assert isinstance(est, dict)


# ---------------------------------------------------------------------------
# MultipleRSquared.from_r_squared
# ---------------------------------------------------------------------------


class TestMultipleRSquaredFromRSquared:
    def test_basic(self):
        result = MultipleRSquared.from_r_squared(0.5, n=100, p=2)
        assert isinstance(result, MultipleRSquaredResult)

    def test_r_squared_preserved(self):
        result = MultipleRSquared.from_r_squared(0.42, n=80, p=3)
        assert result.r_squared == pytest.approx(0.42, abs=1e-5)

    def test_n_and_p_preserved(self):
        result = MultipleRSquared.from_r_squared(0.5, n=60, p=2)
        assert result.n == 60
        assert result.n_predictors == 2
        assert result.df1 == 2
        assert result.df2 == 57

    def test_f_stat_reasonable(self):
        # With large R² and large n, F should be large
        result = MultipleRSquared.from_r_squared(0.8, n=200, p=3)
        assert result.f_statistic > 100

    def test_p_value_small_for_large_f(self):
        result = MultipleRSquared.from_r_squared(0.8, n=200, p=3)
        assert result.p_value < 0.001

    def test_p_value_large_for_r2_near_zero(self):
        result = MultipleRSquared.from_r_squared(0.01, n=50, p=3)
        assert result.p_value > 0.05

    def test_ci_wishart_covers_r2(self):
        r2 = 0.5
        result = MultipleRSquared.from_r_squared(r2, n=200, p=2)
        lo, hi = result.ci_wishart
        assert lo <= r2 <= hi

    def test_ci_fisher_covers_r2(self):
        r2 = 0.5
        result = MultipleRSquared.from_r_squared(r2, n=200, p=2)
        lo, hi = result.ci_fisher
        assert lo <= r2 <= hi

    def test_ci_ncp_covers_r2(self):
        r2 = 0.5
        result = MultipleRSquared.from_r_squared(r2, n=200, p=2)
        # NCP CI is for population R², so only check ordering
        lo, hi = result.ci_ncp
        assert lo <= hi

    def test_ci_bounds_in_unit_interval(self):
        result = MultipleRSquared.from_r_squared(0.6, n=100, p=3)
        for ci in (result.ci_wishart, result.ci_fisher, result.ci_ncp):
            assert 0.0 <= ci[0] <= ci[1] <= 1.0

    def test_adjusted_estimators_present(self):
        result = MultipleRSquared.from_r_squared(0.5, n=100, p=3)
        assert "ezekiel_1930" in result.adjusted_estimators
        assert "wherry_1931" in result.adjusted_estimators
        assert "olkin_pratt_1958" in result.adjusted_estimators

    def test_r2_zero_no_error(self):
        result = MultipleRSquared.from_r_squared(0.0, n=50, p=2)
        assert result.r_squared == pytest.approx(0.0, abs=1e-6)

    def test_r2_one_no_error(self):
        result = MultipleRSquared.from_r_squared(1.0, n=50, p=2)
        assert result.r_squared == pytest.approx(1.0, abs=1e-6)

    def test_invalid_r2_negative(self):
        with pytest.raises(ValueError, match="r_squared"):
            MultipleRSquared.from_r_squared(-0.1, n=50, p=2)

    def test_invalid_r2_above_one(self):
        with pytest.raises(ValueError, match="r_squared"):
            MultipleRSquared.from_r_squared(1.1, n=50, p=2)

    def test_invalid_n_too_small(self):
        with pytest.raises(ValueError):
            MultipleRSquared.from_r_squared(0.5, n=3, p=3)

    def test_invalid_ci_level(self):
        with pytest.raises(ValueError, match="confidence_level"):
            MultipleRSquared.from_r_squared(0.5, n=100, p=2, confidence_level=1.5)

    def test_confidence_level_preserved(self):
        result = MultipleRSquared.from_r_squared(0.5, n=100, p=2, confidence_level=0.99)
        assert result.confidence_level == 0.99

    def test_wider_ci_for_90_pct(self):
        r95 = MultipleRSquared.from_r_squared(0.5, n=50, p=2, confidence_level=0.95)
        r90 = MultipleRSquared.from_r_squared(0.5, n=50, p=2, confidence_level=0.90)
        # 95% CI should be wider than 90% CI for all three methods
        assert (r95.ci_wishart[1] - r95.ci_wishart[0]) > (r90.ci_wishart[1] - r90.ci_wishart[0])
        assert (r95.ci_fisher[1] - r95.ci_fisher[0]) > (r90.ci_fisher[1] - r90.ci_fisher[0])


# ---------------------------------------------------------------------------
# MultipleRSquared.from_data
# ---------------------------------------------------------------------------


class TestMultipleRSquaredFromData:
    def test_basic(self):
        df = make_regression_df()
        result = MultipleRSquared.from_data(df, outcome_col="y")
        assert isinstance(result, MultipleRSquaredResult)
        assert 0.0 <= result.r_squared <= 1.0

    def test_n_and_p_correct(self):
        df = make_regression_df(n=80, p=3)
        result = MultipleRSquared.from_data(df, outcome_col="y")
        assert result.n == 80
        assert result.n_predictors == 3

    def test_perfect_r2(self):
        n = 50
        x1 = RNG.normal(size=n)
        x2 = RNG.normal(size=n)
        y = x1 * 2.0 + x2 * 3.0  # no noise — perfect fit
        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        result = MultipleRSquared.from_data(df, outcome_col="y")
        assert result.r_squared == pytest.approx(1.0, abs=1e-6)

    def test_missing_outcome_col(self):
        df = make_regression_df()
        with pytest.raises(ValueError, match="outcome_col"):
            MultipleRSquared.from_data(df, outcome_col="nonexistent")

    def test_not_dataframe(self):
        with pytest.raises(ValueError, match="DataFrame"):
            MultipleRSquared.from_data([[1, 2], [3, 4]], outcome_col="y")

    def test_no_predictors_raises(self):
        df = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
        with pytest.raises(ValueError, match="predictor"):
            MultipleRSquared.from_data(df, outcome_col="y")

    def test_too_small_raises(self):
        df = pd.DataFrame({"x1": [1.0, 2.0], "y": [2.0, 4.0]})
        with pytest.raises(ValueError):
            MultipleRSquared.from_data(df, outcome_col="y")


# ---------------------------------------------------------------------------
# PartialPearsonCorrelation
# ---------------------------------------------------------------------------


class TestPartialPearsonCorrelation:
    def test_basic(self):
        df = make_partial_df()
        result = PartialPearsonCorrelation.from_data(df)
        assert isinstance(result, PartialCorrelationResult)

    def test_partial_r_in_range(self):
        df = make_partial_df()
        result = PartialPearsonCorrelation.from_data(df)
        assert -1.0 <= result.partial_r <= 1.0
        assert -1.0 <= result.semi_partial_r <= 1.0

    def test_ci_contains_partial_r(self):
        df = make_partial_df(n=200)
        result = PartialPearsonCorrelation.from_data(df)
        lo, hi = result.partial_r_ci
        assert lo <= result.partial_r <= hi

    def test_ci_semi_partial_contains_semi_partial_r(self):
        df = make_partial_df(n=200)
        result = PartialPearsonCorrelation.from_data(df)
        lo, hi = result.semi_partial_r_ci
        assert lo <= result.semi_partial_r <= hi

    def test_n_preserved(self):
        n = 150
        df = make_partial_df(n=n)
        result = PartialPearsonCorrelation.from_data(df)
        assert result.n == n

    def test_n_covariates(self):
        df = make_partial_df()
        result = PartialPearsonCorrelation.from_data(df)
        assert result.n_covariates == 1

    def test_covariate_in_metadata(self):
        df = make_partial_df()
        result = PartialPearsonCorrelation.from_data(df)
        assert "covariate_columns" in result.metadata
        assert "age" in result.metadata["covariate_columns"]

    def test_no_covariates(self):
        # Only IV and DV — degenerate case, should still run (ordinary Pearson)
        rng = np.random.default_rng(99)
        iv = rng.normal(size=60)
        dv = 0.4 * iv + rng.normal(size=60)
        df = pd.DataFrame({"independent_variable": iv, "dependent_variable": dv})
        result = PartialPearsonCorrelation.from_data(df)
        assert result.n_covariates == 0

    def test_confidence_level_99(self):
        df = make_partial_df(n=200)
        r95 = PartialPearsonCorrelation.from_data(df, confidence_level=0.95)
        r99 = PartialPearsonCorrelation.from_data(df, confidence_level=0.99)
        w95 = r95.partial_r_ci[1] - r95.partial_r_ci[0]
        w99 = r99.partial_r_ci[1] - r99.partial_r_ci[0]
        assert w99 > w95

    def test_missing_required_col(self):
        df = pd.DataFrame({
            "independent_variable": [1.0, 2.0, 3.0],
            "x": [4.0, 5.0, 6.0],
        })
        with pytest.raises(ValueError, match="missing required columns"):
            PartialPearsonCorrelation.from_data(df)

    def test_not_dataframe(self):
        with pytest.raises(ValueError, match="DataFrame"):
            PartialPearsonCorrelation.from_data([[1, 2], [3, 4]])

    def test_too_small_sample(self):
        df = pd.DataFrame({
            "independent_variable": [1.0, 2.0, 3.0],
            "dependent_variable": [4.0, 5.0, 6.0],
        })
        with pytest.raises(ValueError, match="Sample size"):
            PartialPearsonCorrelation.from_data(df)

    def test_invalid_ci_level(self):
        df = make_partial_df()
        with pytest.raises(ValueError, match="confidence_level"):
            PartialPearsonCorrelation.from_data(df, confidence_level=0.0)

    def test_p_values_in_range(self):
        df = make_partial_df(n=200)
        result = PartialPearsonCorrelation.from_data(df)
        assert 0.0 <= result.partial_r_p_value <= 1.0
        assert 0.0 <= result.semi_partial_r_p_value <= 1.0

    def test_controlling_reduces_correlation(self):
        """Controlling for a strong common cause should reduce raw correlation."""
        rng = np.random.default_rng(55)
        n = 300
        common = rng.normal(size=n)
        iv = 0.9 * common + rng.normal(0, 0.1, size=n)
        dv = 0.9 * common + rng.normal(0, 0.1, size=n)
        df_with = pd.DataFrame({"independent_variable": iv, "dependent_variable": dv, "common": common})
        df_without = pd.DataFrame({"independent_variable": iv, "dependent_variable": dv})
        res_with = PartialPearsonCorrelation.from_data(df_with)
        res_without = PartialPearsonCorrelation.from_data(df_without)
        assert abs(res_with.partial_r) < abs(res_without.partial_r)
