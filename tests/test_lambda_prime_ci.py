"""Tests for lambda-prime and t-prime paired CI methods in CohensDCI.

These replace the R/rpy2 sadists::qlambdap-based methods from the dev branch.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from esek.confidence_intervals.ci_cohens_d import CohensDCI, CohensDCIResult
from esek.utils.distribution_helpers import plambdap, qlambdap


class TestLambdaPrimeDistribution:
    """Tests for the pure-Python lambda-prime distribution implementation."""

    def test_plambdap_at_mean(self) -> None:
        """CDF at the mean of the lambda-prime distribution should be ~0.5."""
        # Mean of lambda-prime(df=30, t=3) ≈ 3 * sqrt(2/30) * Γ(15.5)/Γ(15) ≈ 2.97
        # So plambdap(mean, 30, 3) should be close to 0.5
        cdf_val = plambdap(2.97, df=30, t=3.0)
        assert abs(cdf_val - 0.5) < 0.05

    def test_plambdap_bounds(self) -> None:
        """CDF should be in [0, 1] and monotonically increasing."""
        qs = [-5.0, -1.0, 0.0, 1.0, 3.0, 5.0, 10.0]
        cdfs = [plambdap(q, df=30, t=3.0) for q in qs]
        assert all(0.0 <= c <= 1.0 for c in cdfs)
        assert all(cdfs[i] <= cdfs[i + 1] for i in range(len(cdfs) - 1))

    def test_plambdap_small_t(self) -> None:
        """For t≈0, lambda-prime reduces to approximately a standard normal."""
        # plambdap(1.96, df=30, t=0.001) ≈ norm.cdf(1.96) ≈ 0.975
        from scipy.stats import norm
        cdf_val = plambdap(1.96, df=30, t=0.001)
        assert abs(cdf_val - norm.cdf(1.96)) < 0.02

    def test_qlambdap_is_inverse_of_plambdap(self) -> None:
        """qlambdap should invert plambdap for several (df, t) pairs."""
        test_cases = [
            (0.025, 20.0, 2.0),
            (0.975, 20.0, 2.0),
            (0.05,  40.0, 1.5),
            (0.95,  40.0, 1.5),
            (0.5,   30.0, 3.0),
        ]
        for p, df, t in test_cases:
            q = qlambdap(p, df=df, t=t)
            p_back = plambdap(q, df=df, t=t)
            assert abs(p_back - p) < 1e-5, f"Round-trip failed for p={p}, df={df}, t={t}"

    def test_qlambdap_quantile_order(self) -> None:
        """Quantiles must be monotone: q(0.025) < q(0.5) < q(0.975)."""
        df, t = 30.0, 2.5
        q025 = qlambdap(0.025, df=df, t=t)
        q500 = qlambdap(0.5, df=df, t=t)
        q975 = qlambdap(0.975, df=df, t=t)
        assert q025 < q500 < q975

    def test_qlambdap_invalid_probability(self) -> None:
        """p=0 or p=1 should raise."""
        with pytest.raises(ValueError):
            qlambdap(0.0, df=30.0, t=2.0)
        with pytest.raises(ValueError):
            qlambdap(1.0, df=30.0, t=2.0)

    def test_qlambdap_invalid_df(self) -> None:
        """df ≤ 0 should raise."""
        with pytest.raises(ValueError):
            qlambdap(0.5, df=0.0, t=1.0)
        with pytest.raises(ValueError):
            qlambdap(0.5, df=-5.0, t=1.0)

    def test_qlambdap_monte_carlo(self) -> None:
        """qlambdap should match empirical quantiles from simulation."""
        np.random.seed(42)
        df, t_param = 30.0, 3.0
        chi2_s = np.random.chisquare(df, 200_000)
        z_s = np.random.normal(0, 1, 200_000)
        samples = z_s + t_param * np.sqrt(chi2_s / df)

        for p in [0.025, 0.1, 0.5, 0.9, 0.975]:
            empirical = float(np.percentile(samples, p * 100))
            theoretical = qlambdap(p, df=df, t=t_param)
            assert abs(theoretical - empirical) < 0.1, (
                f"Monte Carlo mismatch at p={p}: theoretical={theoretical:.4f}, "
                f"empirical={empirical:.4f}"
            )


class TestPairedTLambdaPrime:
    """Tests for CohensDCI.paired_t_lambda_prime."""

    def test_returns_correct_type(self) -> None:
        result = CohensDCI.paired_t_lambda_prime(
            d=0.5, sd1=1.0, sd2=1.0, n=30, r=0.6
        )
        assert isinstance(result, CohensDCIResult)

    def test_ci_contains_d_for_equal_sds(self) -> None:
        """95% CI should contain the observed d value."""
        d = 0.5
        result = CohensDCI.paired_t_lambda_prime(d=d, sd1=1.0, sd2=1.0, n=30, r=0.6)
        assert result.ci_low < d < result.ci_high

    def test_ci_width_decreases_with_n(self) -> None:
        """Larger n should give narrower CI."""
        d, sd1, sd2, r = 0.5, 1.0, 1.0, 0.6
        small = CohensDCI.paired_t_lambda_prime(d=d, sd1=sd1, sd2=sd2, n=20, r=r)
        large = CohensDCI.paired_t_lambda_prime(d=d, sd1=sd1, sd2=sd2, n=100, r=r)
        w_small = small.ci_high - small.ci_low
        w_large = large.ci_high - large.ci_low
        assert w_large < w_small

    def test_symmetric_for_equal_sds_and_zero_d(self) -> None:
        """For d=0 the CI should be centred near zero."""
        result = CohensDCI.paired_t_lambda_prime(d=0.0, sd1=1.0, sd2=1.0, n=40, r=0.5)
        # CI midpoint should be near 0
        mid = (result.ci_low + result.ci_high) / 2
        assert abs(mid) < 0.1

    def test_negative_d(self) -> None:
        """CI for negative d should have negative high bound (large effect)."""
        result = CohensDCI.paired_t_lambda_prime(d=-0.8, sd1=1.0, sd2=1.0, n=50, r=0.7)
        assert result.ci_low < result.ci_high
        assert result.ci_high < 0

    def test_method_name(self) -> None:
        result = CohensDCI.paired_t_lambda_prime(d=0.5, sd1=1.0, sd2=1.0, n=30, r=0.5)
        assert result.method == "t_lambda_prime"
        assert result.design == "paired"

    def test_metadata_contains_expected_keys(self) -> None:
        result = CohensDCI.paired_t_lambda_prime(d=0.5, sd1=1.5, sd2=1.0, n=30, r=0.4)
        meta = result.metadata
        assert "n" in meta
        assert "df" in meta
        assert "df_corrected" in meta
        assert "corrected_correlation" in meta

    def test_unequal_sds(self) -> None:
        """Should work for sd1 ≠ sd2 (corrected correlation ≠ r)."""
        result = CohensDCI.paired_t_lambda_prime(d=0.6, sd1=2.0, sd2=1.0, n=40, r=0.5)
        assert result.ci_low < 0.6 < result.ci_high

    def test_validation_n_too_small(self) -> None:
        with pytest.raises(Exception):
            CohensDCI.paired_t_lambda_prime(d=0.5, sd1=1.0, sd2=1.0, n=2, r=0.5)

    def test_validation_invalid_confidence_level(self) -> None:
        with pytest.raises(Exception):
            CohensDCI.paired_t_lambda_prime(d=0.5, sd1=1.0, sd2=1.0, n=30, r=0.5,
                                             confidence_level=1.5)

    def test_validation_invalid_correlation(self) -> None:
        with pytest.raises(Exception):
            CohensDCI.paired_t_lambda_prime(d=0.5, sd1=1.0, sd2=1.0, n=30, r=1.5)

    def test_90_percent_ci_narrower_than_95(self) -> None:
        d, sd1, sd2, n, r = 0.5, 1.0, 1.0, 30, 0.6
        ci_90 = CohensDCI.paired_t_lambda_prime(d=d, sd1=sd1, sd2=sd2, n=n, r=r,
                                                confidence_level=0.90)
        ci_95 = CohensDCI.paired_t_lambda_prime(d=d, sd1=sd1, sd2=sd2, n=n, r=r,
                                                confidence_level=0.95)
        assert (ci_90.ci_high - ci_90.ci_low) < (ci_95.ci_high - ci_95.ci_low)


class TestPairedTTPrime:
    """Tests for CohensDCI.paired_t_t_prime."""

    def test_returns_correct_type(self) -> None:
        result = CohensDCI.paired_t_t_prime(d=0.5, sd1=1.0, sd2=1.0, n=30, r=0.6)
        assert isinstance(result, CohensDCIResult)

    def test_ci_contains_d_for_equal_sds(self) -> None:
        d = 0.5
        result = CohensDCI.paired_t_t_prime(d=d, sd1=1.0, sd2=1.0, n=30, r=0.6)
        assert result.ci_low < d < result.ci_high

    def test_ci_width_decreases_with_n(self) -> None:
        d, sd1, sd2, r = 0.5, 1.0, 1.0, 0.6
        small = CohensDCI.paired_t_t_prime(d=d, sd1=sd1, sd2=sd2, n=20, r=r)
        large = CohensDCI.paired_t_t_prime(d=d, sd1=sd1, sd2=sd2, n=100, r=r)
        assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)

    def test_method_name(self) -> None:
        result = CohensDCI.paired_t_t_prime(d=0.5, sd1=1.0, sd2=1.0, n=30, r=0.5)
        assert result.method == "t_t_prime"
        assert result.design == "paired"

    def test_compared_to_lambda_prime_similar_bounds(self) -> None:
        """For equal SDs, t_prime and lambda_prime should give similar bounds."""
        d, sd1, sd2, n, r = 0.5, 1.0, 1.0, 50, 0.6
        lp = CohensDCI.paired_t_lambda_prime(d=d, sd1=sd1, sd2=sd2, n=n, r=r)
        tp = CohensDCI.paired_t_t_prime(d=d, sd1=sd1, sd2=sd2, n=n, r=r)
        # Both should have d inside CI
        assert lp.ci_low < d < lp.ci_high
        assert tp.ci_low < d < tp.ci_high
        # They should be within 0.2 of each other
        assert abs(lp.ci_low - tp.ci_low) < 0.2
        assert abs(lp.ci_high - tp.ci_high) < 0.2

    def test_negative_d_gives_negative_upper_bound(self) -> None:
        result = CohensDCI.paired_t_t_prime(d=-0.8, sd1=1.0, sd2=1.0, n=50, r=0.7)
        assert result.ci_high < 0

    def test_90_percent_ci_narrower_than_95(self) -> None:
        d, sd1, sd2, n, r = 0.5, 1.0, 1.0, 30, 0.6
        ci_90 = CohensDCI.paired_t_t_prime(d=d, sd1=sd1, sd2=sd2, n=n, r=r,
                                            confidence_level=0.90)
        ci_95 = CohensDCI.paired_t_t_prime(d=d, sd1=sd1, sd2=sd2, n=n, r=r,
                                            confidence_level=0.95)
        assert (ci_90.ci_high - ci_90.ci_low) < (ci_95.ci_high - ci_95.ci_low)

    def test_validation_negative_sd(self) -> None:
        with pytest.raises(Exception):
            CohensDCI.paired_t_t_prime(d=0.5, sd1=-1.0, sd2=1.0, n=30, r=0.5)

    def test_unequal_sds(self) -> None:
        result = CohensDCI.paired_t_t_prime(d=0.7, sd1=2.0, sd2=1.0, n=40, r=0.4)
        assert result.ci_low < 0.7 < result.ci_high
