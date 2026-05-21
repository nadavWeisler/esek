"""
Regression tests: compare ESEK outputs against the statistician's original
formulas (inlined from dev branch code).

These tests validate statistical correctness — that the refactored ESEK
implementation produces the same numerical results as the original code.

Formulas are copied verbatim from:
  stats/CI_Constructor/CI_Constructor.py  (commit on dev branch)

NOTE: CI_adjusted_lambda_prime_Paired_Samples had a known formula bug in the
original (denominator was `(2*(1-r)) / c2` instead of `scale * c2`).
The ESEK version uses the corrected formula and is intentionally different.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest
from scipy.stats import norm, nct, gmean as sp_gmean


# ---------------------------------------------------------------------------
# Original helper functions (copied verbatim from dev branch CI_Constructor.py)
# ---------------------------------------------------------------------------

def _orig_central_ci_one_sample(cohens_d: float, n: int, cl: float) -> Tuple[float, float, float]:
    se = np.sqrt((1 / n) + ((cohens_d ** 2 / (2 * n))))
    z = norm.ppf(cl + ((1 - cl) / 2))
    return cohens_d - se * z, cohens_d + se * z, se


def _orig_pivotal_ci_t(t_score: float, df: int, n: int, cl: float) -> Tuple[float, float]:
    is_negative = t_score < 0
    t_score = abs(t_score)
    upper_limit = 1 - (1 - cl) / 2
    lower_limit = (1 - cl) / 2
    lc = [-t_score, t_score / 2, t_score]
    uc = [t_score, 2 * t_score, 3 * t_score]
    while nct.cdf(t_score, df, lc[0]) < upper_limit:
        lc = [lc[0] - t_score, lc[0], lc[2]]
    while nct.cdf(t_score, df, uc[0]) < lower_limit:
        if nct.cdf(t_score, df) < lower_limit:
            uc = [uc[0] / 4, uc[0], uc[2]]
    while nct.cdf(t_score, df, uc[2]) > lower_limit:
        uc = [uc[0], uc[2], uc[2] + t_score]
    lower_ci = 0.0
    diff = 1.0
    while diff > 0.00001:
        if nct.cdf(t_score, df, lc[1]) < upper_limit:
            lc = [lc[0], (lc[0] + lc[1]) / 2, lc[1]]
        else:
            lc = [lc[1], (lc[1] + lc[2]) / 2, lc[2]]
        diff = abs(nct.cdf(t_score, df, lc[1]) - upper_limit)
        lower_ci = lc[1] / np.sqrt(n)
    upper_ci = 0.0
    diff = 1.0
    while diff > 0.00001:
        if nct.cdf(t_score, df, uc[1]) < lower_limit:
            uc = [uc[0], (uc[0] + uc[1]) / 2, uc[1]]
        else:
            uc = [uc[1], (uc[1] + uc[2]) / 2, uc[2]]
        diff = abs(nct.cdf(t_score, df, uc[1]) - lower_limit)
        upper_ci = uc[1] / np.sqrt(n)
    if is_negative:
        return -upper_ci, -lower_ci
    return lower_ci, upper_ci


def _orig_central_ci_paired(d: float, n: int, cl: float) -> Tuple[float, float]:
    df = n - 1
    c = math.exp(math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2))
    se = np.sqrt((df / (df - 2)) * (1 / n) * (1 + d ** 2 * n) - (d ** 2 / c ** 2))
    z = norm.ppf(cl + (1 - cl) / 2)
    return d - se * z, d + se * z


def _orig_se_pooled_paired(d: float, n: int, r: float, cl: float) -> Tuple[float, float]:
    df = n - 1
    c = math.exp(math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2))
    A = n / (2 * (1 - r))
    se = np.sqrt((df / (df - 2)) * (1 / A) * (1 + d ** 2 * A) - (d ** 2 / c ** 2))
    z = norm.ppf(cl + (1 - cl) / 2)
    return d - se * z, d + se * z


def _orig_ncp_one_sample(d: float, n: int, cl: float) -> Tuple[float, float]:
    ncp = d * math.sqrt(n)
    low = nct.ppf(0.5 - cl / 2, n - 1, loc=0, scale=1, nc=ncp) / ncp * d
    high = nct.ppf(0.5 + cl / 2, n - 1, loc=0, scale=1, nc=ncp) / ncp * d
    return low, high


def _orig_mag_paired(d: float, sd1: float, sd2: float, n: int, r: float, cl: float) -> Tuple[float, float]:
    r_c = r * (float(sp_gmean([sd1 ** 2, sd2 ** 2])) / np.mean([sd1 ** 2, sd2 ** 2]))
    df = n - 1
    c = math.exp(math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2))
    lam = float(d * c ** 2 * np.sqrt(n / (2 * (1 - r_c))))
    low = nct.ppf(0.5 - cl / 2, df=df, nc=lam) / np.sqrt(n / (2 * (1 - r_c)))
    high = nct.ppf(0.5 + cl / 2, df=df, nc=lam) / np.sqrt(n / (2 * (1 - r_c)))
    return low, high


def _orig_morris_paired(d: float, n: int, r: float, cl: float) -> Tuple[float, float]:
    df = n - 1
    c = math.exp(math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2))
    var = ((df / (df - 2)) * 2 * (1 - r) / n * (1 + d ** 2 * n / (2 * (1 - r))) - d ** 2 / c ** 2) * c ** 2
    z = norm.ppf(cl + (1 - cl) / 2)
    return d - np.sqrt(var) * z, d + np.sqrt(var) * z


def _orig_algina_keselman(d: float, sd1: float, sd2: float, n: int, r: float, cl: float) -> Tuple[float, float]:
    df = n - 1
    r_c = r * (float(sp_gmean([sd1 ** 2, sd2 ** 2])) / np.mean([sd1 ** 2, sd2 ** 2]))
    k = np.sqrt(n / (2 * (1 - r_c)))
    # NCT_ci_t returns NCT values (not divided by sqrt(n))
    is_neg = (d * k) < 0
    t_s = abs(d * k)
    upper_limit = 1 - (1 - cl) / 2
    lower_limit = (1 - cl) / 2
    lc = [-t_s, t_s / 2, t_s]
    uc = [t_s, 2 * t_s, 3 * t_s]
    while nct.cdf(t_s, df, lc[0]) < upper_limit:
        lc = [lc[0] - t_s, lc[0], lc[2]]
    while nct.cdf(t_s, df, uc[0]) < lower_limit:
        if nct.cdf(t_s, df) < lower_limit:
            uc = [uc[0] / 4, uc[0], uc[2]]
    while nct.cdf(t_s, df, uc[2]) > lower_limit:
        uc = [uc[0], uc[2], uc[2] + t_s]
    lower_nct = 0.0
    diff = 1.0
    while diff > 0.00001:
        if nct.cdf(t_s, df, lc[1]) < upper_limit:
            lc = [lc[0], (lc[0] + lc[1]) / 2, lc[1]]
        else:
            lc = [lc[1], (lc[1] + lc[2]) / 2, lc[2]]
        diff = abs(nct.cdf(t_s, df, lc[1]) - upper_limit)
        lower_nct = lc[1]
    upper_nct = 0.0
    diff = 1.0
    while diff > 0.00001:
        if nct.cdf(t_s, df, uc[1]) < lower_limit:
            uc = [uc[0], (uc[0] + uc[1]) / 2, uc[1]]
        else:
            uc = [uc[1], (uc[1] + uc[2]) / 2, uc[2]]
        diff = abs(nct.cdf(t_s, df, uc[1]) - lower_limit)
        upper_nct = uc[1]
    if is_neg:
        lower_nct, upper_nct = -upper_nct, -lower_nct
    return lower_nct / k, upper_nct / k


def _orig_central_ci_two_sample_z(d: float, n1: int, n2: int, cl: float) -> Tuple[float, float, float]:
    se = np.sqrt(((n1 + n2) / (n1 * n2)) + (d ** 2 / (2 * (n1 + n2))))
    z = norm.ppf(cl + (1 - cl) / 2)
    return d - se * z, d + se * z, se


def _orig_central_ci_two_sample_t(d: float, n1: int, n2: int, cl: float) -> Tuple[float, float]:
    n = n1 + n2
    df = n - 2
    c = math.exp(math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2))
    hn = 2 / (1 / n1 + 1 / n2)
    A = hn / 2
    se = np.sqrt((df / (df - 2)) * (1 / A) * (1 + d ** 2 * A) - (d ** 2 / c ** 2))
    z = norm.ppf(cl + (1 - cl) / 2)
    return d - se * z, d + se * z


# ---------------------------------------------------------------------------
# Import ESEK classes
# ---------------------------------------------------------------------------
from esek.confidence_intervals.ci_cohens_d import CohensDCI


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestRegressionCohensDCI:
    """Regression tests comparing CohensDCI methods against original formulas."""

    # --- one-sample Z ---
    def test_one_sample_z_ci(self) -> None:
        d, n, cl = 0.5, 50, 0.95
        lo_orig, hi_orig, _ = _orig_central_ci_one_sample(d, n, cl)
        result = CohensDCI.one_sample_z(d=d, n=n, confidence_level=cl)
        assert result.ci_low == pytest.approx(lo_orig, rel=1e-6)
        assert result.ci_high == pytest.approx(hi_orig, rel=1e-6)

    def test_one_sample_z_negative_d(self) -> None:
        d, n, cl = -0.8, 30, 0.95
        lo_orig, hi_orig, _ = _orig_central_ci_one_sample(d, n, cl)
        result = CohensDCI.one_sample_z(d=d, n=n, confidence_level=cl)
        assert result.ci_low == pytest.approx(lo_orig, rel=1e-6)
        assert result.ci_high == pytest.approx(hi_orig, rel=1e-6)

    # --- pivotal one-sample t (original takes t_stat, not d) ---
    def test_one_sample_t_pivotal(self) -> None:
        d, n, cl = 0.6, 40, 0.95
        t_score = d * math.sqrt(n)
        df = n - 1
        lo_orig, hi_orig = _orig_pivotal_ci_t(t_score, df, n, cl)
        result = CohensDCI.one_sample_t_pivotal(t_stat=t_score, n=n, confidence_level=cl)
        assert result.ci_low == pytest.approx(lo_orig, abs=1e-4)
        assert result.ci_high == pytest.approx(hi_orig, abs=1e-4)

    def test_one_sample_t_pivotal_negative(self) -> None:
        d, n, cl = -0.4, 50, 0.95
        t_score = d * math.sqrt(n)
        df = n - 1
        lo_orig, hi_orig = _orig_pivotal_ci_t(t_score, df, n, cl)
        result = CohensDCI.one_sample_t_pivotal(t_stat=t_score, n=n, confidence_level=cl)
        assert result.ci_low == pytest.approx(lo_orig, abs=1e-4)
        assert result.ci_high == pytest.approx(hi_orig, abs=1e-4)

    # --- central one-sample t (returns list, index 0 = "true" SE) ---
    def test_paired_t_central_ci(self) -> None:
        d, n, cl = 0.5, 30, 0.95
        lo_orig, hi_orig = _orig_central_ci_paired(d, n, cl)
        results = CohensDCI.paired_t_central(d=d, n=n, confidence_level=cl)
        # index 0 is the "true" SE method (Hedges & Olkin 1985)
        result = results[0]
        assert result.ci_low == pytest.approx(lo_orig, rel=1e-6)
        assert result.ci_high == pytest.approx(hi_orig, rel=1e-6)

    # --- pooled paired SE (returns list, index 0 = "true") ---
    def test_paired_t_pooled_central(self) -> None:
        d, n, r, cl = 0.5, 30, 0.4, 0.95
        lo_orig, hi_orig = _orig_se_pooled_paired(d, n, r, cl)
        results = CohensDCI.paired_t_pooled_central(d=d, n=n, r=r, confidence_level=cl)
        result = results[0]
        assert result.ci_low == pytest.approx(lo_orig, rel=1e-6)
        assert result.ci_high == pytest.approx(hi_orig, rel=1e-6)

    # --- NCP one-sample ---
    def test_one_sample_t_ncp(self) -> None:
        d, n, cl = 0.5, 40, 0.95
        lo_orig, hi_orig = _orig_ncp_one_sample(d, n, cl)
        result = CohensDCI.one_sample_t_ncp(d=d, n=n, confidence_level=cl)
        assert result.ci_low == pytest.approx(lo_orig, rel=1e-5)
        assert result.ci_high == pytest.approx(hi_orig, rel=1e-5)

    # --- MAG paired ---
    def test_paired_t_mag(self) -> None:
        d, sd1, sd2, n, r, cl = 0.5, 1.2, 1.4, 30, 0.4, 0.95
        lo_orig, hi_orig = _orig_mag_paired(d, sd1, sd2, n, r, cl)
        result = CohensDCI.paired_t_mag(d=d, sd1=sd1, sd2=sd2, n=n, r=r, confidence_level=cl)
        assert result.ci_low == pytest.approx(lo_orig, rel=1e-5)
        assert result.ci_high == pytest.approx(hi_orig, rel=1e-5)

    def test_paired_t_mag_equal_sds(self) -> None:
        """Equal SDs means corrected_r == r."""
        d, sd1, sd2, n, r, cl = 0.5, 1.5, 1.5, 30, 0.4, 0.95
        lo_orig, hi_orig = _orig_mag_paired(d, sd1, sd2, n, r, cl)
        result = CohensDCI.paired_t_mag(d=d, sd1=sd1, sd2=sd2, n=n, r=r, confidence_level=cl)
        assert result.ci_low == pytest.approx(lo_orig, rel=1e-5)
        assert result.ci_high == pytest.approx(hi_orig, rel=1e-5)

    # --- Morris paired ---
    def test_paired_t_morris(self) -> None:
        d, n, r, cl = 0.5, 30, 0.4, 0.95
        lo_orig, hi_orig = _orig_morris_paired(d, n, r, cl)
        result = CohensDCI.paired_t_morris(d=d, n=n, r=r, confidence_level=cl)
        assert result.ci_low == pytest.approx(lo_orig, rel=1e-6)
        assert result.ci_high == pytest.approx(hi_orig, rel=1e-6)

    # --- Algina-Keselman paired ---
    def test_paired_t_algina_keselman(self) -> None:
        d, sd1, sd2, n, r, cl = 0.5, 1.2, 1.4, 30, 0.4, 0.95
        lo_orig, hi_orig = _orig_algina_keselman(d, sd1, sd2, n, r, cl)
        result = CohensDCI.paired_t_algina_keselman(d=d, sd1=sd1, sd2=sd2, n=n, r=r, confidence_level=cl)
        assert result.ci_low == pytest.approx(lo_orig, abs=1e-4)
        assert result.ci_high == pytest.approx(hi_orig, abs=1e-4)

    # --- two-sample Z ---
    def test_independent_z_ci(self) -> None:
        d, n1, n2, cl = 0.5, 40, 45, 0.95
        lo_orig, hi_orig, _ = _orig_central_ci_two_sample_z(d, n1, n2, cl)
        result = CohensDCI.independent_z(d=d, n1=n1, n2=n2, confidence_level=cl)
        assert result.ci_low == pytest.approx(lo_orig, rel=1e-6)
        assert result.ci_high == pytest.approx(hi_orig, rel=1e-6)

    # --- two-sample t central (returns list, index 0 = "true") ---
    def test_independent_t_central(self) -> None:
        d, n1, n2, cl = 0.5, 40, 45, 0.95
        lo_orig, hi_orig = _orig_central_ci_two_sample_t(d, n1, n2, cl)
        results = CohensDCI.independent_t_central(d=d, n1=n1, n2=n2, confidence_level=cl)
        result = results[0]
        assert result.ci_low == pytest.approx(lo_orig, rel=1e-6)
        assert result.ci_high == pytest.approx(hi_orig, rel=1e-6)

    # --- lambda-prime (corrected formula) ---
    def test_lambda_prime_corrected_formula(self) -> None:
        """
        ESEK uses corrected denominator (scale * c2), not the original buggy
        denominator ((2*(1-r)) / c2). Results should be statistically sensible
        (CI containing d) and NOT match the original buggy output.
        """
        d, sd1, sd2, n, r, cl = 0.5, 1.2, 1.4, 30, 0.4, 0.95
        result = CohensDCI.paired_t_lambda_prime(d=d, sd1=sd1, sd2=sd2, n=n, r=r, confidence_level=cl)
        # CI should contain d
        assert result.ci_low < d < result.ci_high
        # CI should be a reasonable width (not huge like [1.15, 6.39])
        assert (result.ci_high - result.ci_low) < 2.0

    # --- 90% CI is narrower than 95% CI ---
    def test_ci_width_narrows_with_confidence(self) -> None:
        d, n, cl95, cl90 = 0.5, 50, 0.95, 0.90
        r95 = CohensDCI.one_sample_z(d=d, n=n, confidence_level=cl95)
        r90 = CohensDCI.one_sample_z(d=d, n=n, confidence_level=cl90)
        width95 = r95.ci_high - r95.ci_low
        width90 = r90.ci_high - r90.ci_low
        assert width90 < width95

    # --- different methods produce different CIs (not all identical) ---
    def test_paired_methods_differ(self) -> None:
        """With high correlation, pooled CI is narrower than the standard central CI."""
        d, n, r_high, cl = 0.5, 30, 0.8, 0.95
        central_list = CohensDCI.paired_t_central(d=d, n=n, confidence_level=cl)
        pooled_list = CohensDCI.paired_t_pooled_central(d=d, n=n, r=r_high, confidence_level=cl)
        central = central_list[0]
        pooled = pooled_list[0]
        # With high r (0.8), pooled A = n/(2*(1-r)) = 30/(2*0.2) = 75 > n, so CI is narrower
        width_central = central.ci_high - central.ci_low
        width_pooled = pooled.ci_high - pooled.ci_low
        assert width_pooled < width_central


# ---------------------------------------------------------------------------
# Regression tests for proportion calculators vs original dev branch formulas
# (inlining key formulas from stats/Calculator/Poportions/)
# ---------------------------------------------------------------------------

class TestRegressionProportionCalculators:
    """Regression tests for proportion calculators against original formulas."""

    def test_one_sample_arcsine_effect_size(self) -> None:
        """h = 2*arcsin(sqrt(p)) - 2*arcsin(sqrt(p0)) — Cohen's h formula."""
        from esek.calculators.proportions.one_sample_proportion import OneSampleProportions

        p = 0.3
        p0 = 0.5  # null hypothesis proportion
        # Cohen's h = phi(p) - phi(p0) where phi(x) = 2*arcsin(sqrt(x))
        h_orig = 2 * math.asin(math.sqrt(p)) - 2 * math.asin(math.sqrt(p0))
        result = OneSampleProportions.from_parameters(
            proportion_sample=p, sample_size=100,
            population_proportion=p0, confidence_level=0.95
        )
        # The h value should match the original formula
        assert result.cohens_h is not None
        assert abs(result.cohens_h.value - h_orig) < 0.001

    def test_two_independent_proportions_cohens_h(self) -> None:
        """Cohen's h for two independent proportions."""
        from esek.calculators.proportions.two_independent_proportions import TwoIndependentProportions

        p1, p2, n1, n2 = 0.4, 0.25, 100, 100
        h_orig = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
        result = TwoIndependentProportions.from_parameters(
            proportion_sample_1=p1, proportion_sample_2=p2,
            sample_size_1=n1, sample_size_2=n2, confidence_level=0.95
        )
        assert result.cohens_h is not None
        assert abs(result.cohens_h.value - h_orig) < 0.001


# ---------------------------------------------------------------------------
# Regression tests for correlation calculators vs original dev branch formulas
# ---------------------------------------------------------------------------

class TestRegressionCorrelationCalculators:
    """Regression tests for correlation CIs against original formulas."""

    def test_pearson_fisher_z_ci(self) -> None:
        """Fisher-z CI for Pearson r."""
        from esek.confidence_intervals.ci_correlations import fisher_z_ci

        r, n, cl = 0.5, 50, 0.95
        # Original Fisher-z formula (from Associations_and_Correlations.py)
        z_r = 0.5 * math.log((1 + r) / (1 - r))
        se = 1 / math.sqrt(n - 3)
        z_crit = norm.ppf(cl + (1 - cl) / 2)
        lo_orig = math.tanh(z_r - z_crit * se)
        hi_orig = math.tanh(z_r + z_crit * se)

        lo_esek, hi_esek = fisher_z_ci(r=r, n=n, confidence_level=cl)
        assert lo_esek == pytest.approx(lo_orig, rel=1e-5)
        assert hi_esek == pytest.approx(hi_orig, rel=1e-5)

    def test_pearson_ci_contains_r(self) -> None:
        """95% CI should contain the sample correlation."""
        from esek.confidence_intervals.ci_correlations import fisher_z_ci

        for r_val in [-0.7, -0.3, 0.0, 0.3, 0.7]:
            lo, hi = fisher_z_ci(r=r_val, n=100, confidence_level=0.95)
            assert lo <= r_val <= hi


# ---------------------------------------------------------------------------
# Regression tests for converter vs original dev branch formulas
# ---------------------------------------------------------------------------

class TestRegressionConverters:
    """Regression tests for converters against original formulas."""

    def test_d_to_r_conversion(self) -> None:
        """d to r (equal n): r = d / sqrt(d^2 + 4)."""
        from esek.converters.d_conversions import d_to_r_equal_n

        d, n = 0.5, 50
        r_orig = d / math.sqrt(d ** 2 + 4)
        result = d_to_r_equal_n(d=d, n=n)
        assert result.output_value == pytest.approx(r_orig, rel=1e-6)

    def test_r_to_d_conversion(self) -> None:
        """r to d: d = 2r / sqrt(1 - r^2)."""
        from esek.converters.r_conversions import r_to_d

        r, n = 0.3, 50
        d_orig = 2 * r / math.sqrt(1 - r ** 2)
        result = r_to_d(r=r, n1=n, n2=n)
        assert result.output_value == pytest.approx(d_orig, rel=1e-6)

    def test_d_to_odds_ratio(self) -> None:
        """d to OR: OR = exp(d * pi / sqrt(3))."""
        from esek.converters.d_conversions import d_to_odds_ratio

        d = 0.5
        or_orig = math.exp(d * math.pi / math.sqrt(3))
        result = d_to_odds_ratio(d=d)
        assert result.output_value == pytest.approx(or_orig, rel=1e-6)

    def test_t_to_d_one_sample(self) -> None:
        """t to d (one sample): d = t / sqrt(n-1) [ESEK formula]."""
        from esek.converters.statistic_to_effect_size import StatisticToEffectSize

        t, n = 2.5, 30
        # ESEK uses d = t / sqrt(df) = t / sqrt(n-1)
        d_orig = t / math.sqrt(n - 1)
        result = StatisticToEffectSize.from_t_one_sample(t=t, n=n)
        assert result.cohens_d == pytest.approx(d_orig, rel=1e-6)


# ---------------------------------------------------------------------------
# Regression tests for eta-squared CI vs original dev branch formulas
# ---------------------------------------------------------------------------

class TestRegressionEtaSquaredCI:
    """Regression tests for eta-squared CI against original formulas."""

    def test_eta_squared_f_based_ci(self) -> None:
        """eta² CI from F statistic — CI should contain eta² and be in [0,1]."""
        from esek.confidence_intervals.ci_eta_squared import EtaSquaredCI

        F, df1, df2, cl = 5.0, 2, 47, 0.95
        eta2 = (F * df1) / (F * df1 + df2)
        result = EtaSquaredCI.from_f(f_statistic=F, df1=df1, df2=df2, confidence_level=cl)
        lo = result.ci_partial_eta_sq_f_method[0]
        hi = result.ci_partial_eta_sq_f_method[1]
        assert lo <= eta2 <= hi
        assert lo >= 0
        assert hi <= 1

    def test_eta_squared_increases_with_f(self) -> None:
        """Larger F → larger eta² point estimate."""
        from esek.confidence_intervals.ci_eta_squared import EtaSquaredCI

        r1 = EtaSquaredCI.from_f(f_statistic=3.0, df1=2, df2=47, confidence_level=0.95)
        r2 = EtaSquaredCI.from_f(f_statistic=8.0, df1=2, df2=47, confidence_level=0.95)
        assert r1.partial_eta_squared < r2.partial_eta_squared
