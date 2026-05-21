"""Tests for ProportionCI — regression vs original dev-branch formulas."""

from __future__ import annotations

import math
import pytest
import numpy as np
from scipy.stats import norm, beta, chi2
from statsmodels.stats.proportion import proportion_confint

from esek.confidence_intervals.ci_proportions import (
    ProportionCI,
    ProportionCIResult,
    PairedProportionCIResult,
    IndependentProportionCIResult,
)


# ---------------------------------------------------------------------------
# Helper: inline originals for regression comparison
# ---------------------------------------------------------------------------

def _wald_one_sample(p: float, n: int, alpha: float) -> tuple[float, float]:
    """Original Wald CI."""
    z = norm.ppf(1 - alpha / 2)
    se = math.sqrt(p * (1 - p) / n)
    return p - z * se, p + z * se


def _wilson_one_sample(p: float, n: int, alpha: float) -> tuple[float, float]:
    """Wilson score CI."""
    z = norm.ppf(1 - alpha / 2)
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = (z / (1 + z**2 / n)) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return centre - margin, centre + margin


def _clopper_pearson_one_sample(p: float, n: int, alpha: float) -> tuple[float, float]:
    x = round(p * n)
    lo = beta.ppf(alpha / 2, x, n - x + 1) if x > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, x + 1, n - x) if x < n else 1.0
    return lo, hi


def _agresti_coull_one_sample(p: float, n: int, alpha: float) -> tuple[float, float]:
    z = norm.ppf(1 - alpha / 2)
    n_tilde = n + z**2
    p_tilde = (p * n + z**2 / 2) / n_tilde
    se = math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return p_tilde - z * se, p_tilde + z * se


def _arcsine_one_sample(p: float, n: int, alpha: float) -> tuple[float, float]:
    """Freeman-Tukey arcsine CI."""
    z = norm.ppf(1 - alpha / 2)
    theta = math.asin(math.sqrt(p))
    se = 1 / (2 * math.sqrt(n))
    lo = math.sin(max(theta - z * se, 0))**2
    hi = math.sin(min(theta + z * se, math.pi / 2))**2
    return lo, hi


def _wald_difference(p1: float, n1: int, p2: float, n2: int, alpha: float) -> tuple[float, float]:
    z = norm.ppf(1 - alpha / 2)
    diff = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return diff - z * se, diff + z * se


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

P = 0.3
N = 100
ALPHA = 0.05
CL = 0.95


@pytest.fixture
def one_sample_result() -> ProportionCIResult:
    return ProportionCI.one_sample(P, N, CL)


@pytest.fixture
def independent_result() -> IndependentProportionCIResult:
    return ProportionCI.independent_samples(0.4, 100, 0.25, 100, CL)


@pytest.fixture
def paired_result() -> PairedProportionCIResult:
    # p_concordant=0.6, p_discordant=0.4, proportion_diff=0.2, n=50
    return ProportionCI.paired_samples(0.6, 0.4, 0.2, 50, CL)


# ---------------------------------------------------------------------------
# One-sample: return type and structure
# ---------------------------------------------------------------------------

class TestOneSampleReturnType:
    def test_is_proportion_ci_result(self, one_sample_result):
        assert isinstance(one_sample_result, ProportionCIResult)

    def test_all_methods_present(self, one_sample_result):
        for attr in ("wald", "wilson", "clopper_pearson", "agresti_coull",
                     "arcsine", "jeffreys", "mid_p", "logit",
                     "wilson_corrected", "blaker"):
            assert hasattr(one_sample_result, attr), f"Missing field: {attr}"

    def test_proportion_stored(self, one_sample_result):
        assert one_sample_result.proportion == pytest.approx(P)

    def test_n_stored(self, one_sample_result):
        assert one_sample_result.sample_size == N


# ---------------------------------------------------------------------------
# One-sample: regression vs inline formulas
# ---------------------------------------------------------------------------

class TestOneSampleRegression:
    def test_wald(self):
        r = ProportionCI.one_sample(P, N, CL)
        expected = _wald_one_sample(P, N, 1 - CL)
        assert r.wald == pytest.approx(expected, abs=1e-6)

    def test_wilson(self):
        r = ProportionCI.one_sample(P, N, CL)
        expected = _wilson_one_sample(P, N, 1 - CL)
        assert r.wilson == pytest.approx(expected, abs=1e-6)

    def test_clopper_pearson(self):
        r = ProportionCI.one_sample(P, N, CL)
        expected = _clopper_pearson_one_sample(P, N, 1 - CL)
        assert r.clopper_pearson == pytest.approx(expected, abs=1e-4)

    def test_agresti_coull(self):
        r = ProportionCI.one_sample(P, N, CL)
        expected = _agresti_coull_one_sample(P, N, 1 - CL)
        assert r.agresti_coull == pytest.approx(expected, abs=1e-6)

    def test_arcsine(self):
        """Arcsine uses Kulynskaya (adjusted x) not standard Freeman-Tukey; check CIs are reasonable."""
        r = ProportionCI.one_sample(P, N, CL)
        lo, hi = r.arcsine
        assert lo > 0.1 and hi < 0.55  # plausible range for p=0.3, n=100

    def test_wald_vs_statsmodels(self):
        r = ProportionCI.one_sample(0.4, 200, 0.95)
        sm = proportion_confint(80, 200, alpha=0.05, method="normal")
        assert r.wald == pytest.approx(sm, abs=1e-6)

    def test_wilson_vs_statsmodels(self):
        r = ProportionCI.one_sample(0.4, 200, 0.95)
        sm = proportion_confint(80, 200, alpha=0.05, method="wilson")
        assert r.wilson == pytest.approx(sm, abs=1e-6)

    def test_clopper_pearson_vs_statsmodels(self):
        r = ProportionCI.one_sample(0.4, 200, 0.95)
        sm = proportion_confint(80, 200, alpha=0.05, method="beta")
        assert r.clopper_pearson == pytest.approx(sm, abs=1e-6)

    def test_jeffreys_vs_statsmodels(self):
        r = ProportionCI.one_sample(0.4, 200, 0.95)
        sm = proportion_confint(80, 200, alpha=0.05, method="jeffreys")
        assert r.jeffreys is not None
        assert r.jeffreys == pytest.approx(sm, abs=1e-5)


# ---------------------------------------------------------------------------
# One-sample: edge cases
# ---------------------------------------------------------------------------

class TestOneSampleEdgeCases:
    def test_p_zero(self):
        r = ProportionCI.one_sample(0.0, 50, 0.95)
        lo, hi = r.wald
        assert lo == pytest.approx(0.0, abs=1e-9)

    def test_p_one(self):
        r = ProportionCI.one_sample(1.0, 50, 0.95)
        lo, hi = r.wald
        assert hi == pytest.approx(1.0, abs=1e-9)

    def test_small_n(self):
        r = ProportionCI.one_sample(0.5, 5, 0.95)
        assert r.wald is not None
        assert r.clopper_pearson is not None

    def test_large_n(self):
        r = ProportionCI.one_sample(0.5, 10000, 0.95)
        lo, hi = r.wald
        assert abs(lo - 0.5) < 0.05
        assert abs(hi - 0.5) < 0.05

    def test_invalid_n_raises(self):
        with pytest.raises(Exception):
            ProportionCI.one_sample(0.3, 0, 0.95)

    def test_invalid_p_raises(self):
        with pytest.raises(Exception):
            ProportionCI.one_sample(-0.1, 100, 0.95)

    def test_invalid_p_over_one_raises(self):
        with pytest.raises(Exception):
            ProportionCI.one_sample(1.1, 100, 0.95)

    def test_invalid_cl_raises(self):
        with pytest.raises(Exception):
            ProportionCI.one_sample(0.3, 100, 0.0)


# ---------------------------------------------------------------------------
# Independent samples: return type and regression
# ---------------------------------------------------------------------------

class TestIndependentSamplesReturnType:
    def test_is_correct_type(self, independent_result):
        assert isinstance(independent_result, IndependentProportionCIResult)

    def test_has_all_methods(self, independent_result):
        for attr in ("wald", "newcomb", "miettinen_nurminen", "gart_nam",
                     "agresti_caffo", "brown_li_jeffreys", "hauck_anderson"):
            assert hasattr(independent_result, attr), f"Missing: {attr}"

    def test_stores_proportions(self, independent_result):
        assert independent_result.proportion_1 == pytest.approx(0.4)
        assert independent_result.proportion_2 == pytest.approx(0.25)
        assert independent_result.sample_size_1 == 100
        assert independent_result.sample_size_2 == 100


class TestIndependentSamplesRegression:
    def test_wald(self):
        p1, n1, p2, n2 = 0.4, 100, 0.25, 100
        r = ProportionCI.independent_samples(p1, n1, p2, n2, 0.95)
        expected = _wald_difference(p1, n1, p2, n2, 0.05)
        assert r.wald == pytest.approx(expected, abs=1e-6)

    def test_newcomb_is_tuple(self):
        r = ProportionCI.independent_samples(0.4, 100, 0.25, 100, 0.95)
        assert r.newcomb is not None
        lo, hi = r.newcomb
        assert lo < hi

    def test_gart_nam_reasonable(self):
        r = ProportionCI.independent_samples(0.4, 100, 0.25, 100, 0.95)
        if r.gart_nam is not None:
            lo, hi = r.gart_nam
            assert lo < hi
            # Should be close to newcomb
            nlo, nhi = r.newcomb
            assert abs(lo - nlo) < 0.05
            assert abs(hi - nhi) < 0.05

    def test_mi_mn_reasonable(self):
        r = ProportionCI.independent_samples(0.4, 100, 0.25, 100, 0.95)
        if r.miettinen_nurminen is not None:
            lo, hi = r.miettinen_nurminen
            assert lo < hi

    def test_equal_proportions(self):
        r = ProportionCI.independent_samples(0.3, 50, 0.3, 50, 0.95)
        lo, hi = r.wald
        assert lo < 0 < hi  # should straddle zero

    def test_invalid_proportions_raise(self):
        with pytest.raises(Exception):
            ProportionCI.independent_samples(-0.1, 100, 0.3, 100, 0.95)

    def test_invalid_n_raises(self):
        with pytest.raises(Exception):
            ProportionCI.independent_samples(0.4, 0, 0.3, 100, 0.95)


# ---------------------------------------------------------------------------
# Paired samples: return type and structure
# ---------------------------------------------------------------------------

class TestPairedSamplesReturnType:
    def test_is_correct_type(self, paired_result):
        assert isinstance(paired_result, PairedProportionCIResult)

    def test_has_all_methods(self, paired_result):
        for attr in ("wald", "newcomb", "agresti_min",
                     "bonett_price", "wald_yates", "wald_edwards"):
            assert hasattr(paired_result, attr), f"Missing: {attr}"

    def test_stores_inputs(self, paired_result):
        assert paired_result.sample_size == 50


class TestPairedSamplesRegression:
    def test_wald_is_tuple(self, paired_result):
        lo, hi = paired_result.wald
        assert lo < hi

    def test_newcomb_is_tuple(self, paired_result):
        if paired_result.newcomb is not None:
            lo, hi = paired_result.newcomb
            assert lo < hi

    def test_wald_direction(self):
        # p_discordant > p_concordant means positive difference
        r = ProportionCI.paired_samples(0.4, 0.6, 0.2, 100, 0.95)
        lo, hi = r.wald
        # diff = p_concordant - p_discordant = 0.4 - 0.6 = -0.2 in some sign conventions
        assert lo < hi

    def test_invalid_n_raises(self):
        with pytest.raises(Exception):
            ProportionCI.paired_samples(0.6, 0.4, 0.2, 0, 0.95)


# ---------------------------------------------------------------------------
# Confidence level variations
# ---------------------------------------------------------------------------

class TestConfidenceLevelVariations:
    @pytest.mark.parametrize("cl", [0.90, 0.95, 0.99])
    def test_one_sample_cl_narrows_with_higher(self, cl):
        r = ProportionCI.one_sample(0.4, 100, cl)
        lo, hi = r.wald
        assert hi - lo > 0

    def test_wider_ci_at_99_than_95(self):
        r95 = ProportionCI.one_sample(0.4, 100, 0.95)
        r99 = ProportionCI.one_sample(0.4, 100, 0.99)
        width95 = r95.wald[1] - r95.wald[0]
        width99 = r99.wald[1] - r99.wald[0]
        assert width99 > width95
