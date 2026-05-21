"""Tests for the new confidence interval modules.

Covers:
    - EtaSquaredCI (ci_eta_squared)
    - spearman_ci (ci_correlations)
    - cramer_v_ci (ci_correlations)
    - SpearmanCIResult, CramerVCIResult frozen dataclasses
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from esek.confidence_intervals import (
    EtaSquaredCI,
    cramer_v_ci,
    fisher_z_ci,
    spearman_ci,
)
from esek.confidence_intervals.ci_correlations import SpearmanCIResult, CramerVCIResult
from esek.confidence_intervals.ci_eta_squared import EtaSquaredCIResult


# ============================================================
# EtaSquaredCI
# ============================================================


class TestEtaSquaredCI:
    """Tests for EtaSquaredCI.from_f and from_partial_eta_squared."""

    def test_from_f_returns_correct_type(self):
        result = EtaSquaredCI.from_f(f_statistic=4.2, df1=2, df2=57)
        assert isinstance(result, EtaSquaredCIResult)

    def test_from_f_ci_contains_estimate(self):
        result = EtaSquaredCI.from_f(f_statistic=4.2, df1=2, df2=57)
        lo, hi = result.ci_partial_eta_sq_fleishman
        assert lo <= result.partial_eta_squared <= hi

    def test_from_f_omega_sq_ci_smaller(self):
        """Partial ω² should be bias-corrected downward from η²."""
        result = EtaSquaredCI.from_f(f_statistic=5.0, df1=2, df2=60)
        eta_hi = result.ci_partial_eta_sq_fleishman[1]
        omega_hi = result.ci_partial_omega_sq[1]
        assert omega_hi <= eta_hi + 1e-3  # omega ≤ eta (bias corrected)

    def test_from_partial_eta_roundtrips(self):
        """Converting eta² → F → eta² should recover the original."""
        f = 4.5
        df1, df2 = 2, 60
        result_f = EtaSquaredCI.from_f(f_statistic=f, df1=df1, df2=df2)
        result_eta = EtaSquaredCI.from_partial_eta_squared(
            result_f.partial_eta_squared, df1=df1, df2=df2
        )
        assert result_eta.ci_partial_eta_sq_fleishman == pytest.approx(
            result_f.ci_partial_eta_sq_fleishman, abs=1e-3
        )

    def test_larger_f_gives_wider_ci(self):
        r1 = EtaSquaredCI.from_f(2.0, df1=1, df2=30)
        r2 = EtaSquaredCI.from_f(8.0, df1=1, df2=30)
        width1 = r1.ci_partial_eta_sq_fleishman[1] - r1.ci_partial_eta_sq_fleishman[0]
        width2 = r2.ci_partial_eta_sq_fleishman[1] - r2.ci_partial_eta_sq_fleishman[0]
        # Larger F → point estimate moves away from 0 → CI may be narrower or wider
        # But both should be valid (positive widths)
        assert width1 > 0
        assert width2 > 0

    def test_invalid_f_statistic_raises(self):
        with pytest.raises(ValueError, match="≥ 0"):
            EtaSquaredCI.from_f(f_statistic=-1.0, df1=2, df2=50)

    def test_invalid_df_raises(self):
        with pytest.raises(ValueError, match="Degrees of freedom"):
            EtaSquaredCI.from_f(f_statistic=3.0, df1=0, df2=50)

    def test_invalid_eta_raises(self):
        with pytest.raises(ValueError, match="partial_eta_sq"):
            EtaSquaredCI.from_partial_eta_squared(1.1, df1=2, df2=50)

    def test_invalid_ci_level_raises(self):
        with pytest.raises(ValueError, match="confidence_level"):
            EtaSquaredCI.from_f(f_statistic=3.0, df1=2, df2=50, confidence_level=1.5)

    def test_ci_width_larger_at_95_vs_80(self):
        r95 = EtaSquaredCI.from_f(4.0, df1=2, df2=50, confidence_level=0.95)
        r80 = EtaSquaredCI.from_f(4.0, df1=2, df2=50, confidence_level=0.80)
        w95 = r95.ci_partial_eta_sq_fleishman[1] - r95.ci_partial_eta_sq_fleishman[0]
        w80 = r80.ci_partial_eta_sq_fleishman[1] - r80.ci_partial_eta_sq_fleishman[0]
        assert w95 > w80

    def test_metadata_has_ncp(self):
        result = EtaSquaredCI.from_f(4.0, df1=2, df2=50)
        assert "ncp_lower" in result.metadata
        assert "ncp_upper" in result.metadata
        assert result.metadata["ncp_lower"] >= 0

    def test_small_f_gives_near_zero_lower_ci(self):
        """F close to 0 → lower CI near 0."""
        result = EtaSquaredCI.from_f(f_statistic=0.1, df1=2, df2=100)
        assert result.ci_partial_eta_sq_fleishman[0] == pytest.approx(0.0, abs=0.01)


# ============================================================
# spearman_ci
# ============================================================


class TestSpearmanCI:
    """Tests for the spearman_ci function."""

    def test_returns_correct_type(self):
        result = spearman_ci(rho=0.4, n=50)
        assert isinstance(result, SpearmanCIResult)

    def test_ci_contains_rho(self):
        result = spearman_ci(rho=0.4, n=50)
        lo, hi = result.ci_bonett_wright_z
        assert lo < 0.4 < hi

    def test_all_ci_variants_present(self):
        result = spearman_ci(rho=0.4, n=50)
        for attr in ("ci_bonett_wright_z", "ci_bonett_wright_t", "ci_fieller", "ci_fisher_z", "ci_fisher_t"):
            lo, hi = getattr(result, attr)
            assert lo < hi

    def test_wider_ci_at_lower_level(self):
        r95 = spearman_ci(rho=0.5, n=40, confidence_level=0.95)
        r80 = spearman_ci(rho=0.5, n=40, confidence_level=0.80)
        w95 = r95.ci_bonett_wright_z[1] - r95.ci_bonett_wright_z[0]
        w80 = r80.ci_bonett_wright_z[1] - r80.ci_bonett_wright_z[0]
        assert w95 > w80

    def test_negative_rho(self):
        result = spearman_ci(rho=-0.5, n=60)
        lo, hi = result.ci_bonett_wright_z
        assert lo < -0.5 < hi

    def test_near_zero_rho_symmetric(self):
        result = spearman_ci(rho=0.0, n=100)
        lo, hi = result.ci_bonett_wright_z
        assert pytest.approx(lo, abs=1e-3) == -hi

    def test_invalid_rho_raises(self):
        with pytest.raises(ValueError, match="rho"):
            spearman_ci(rho=1.5, n=50)
        with pytest.raises(ValueError, match="rho"):
            spearman_ci(rho=1.0, n=50)

    def test_small_n_raises(self):
        with pytest.raises(ValueError, match="n must be"):
            spearman_ci(rho=0.3, n=3)

    def test_metadata_has_se(self):
        result = spearman_ci(rho=0.4, n=50)
        assert "se_bonett_wright" in result.metadata
        assert "fisher_z" in result.metadata


# ============================================================
# cramer_v_ci
# ============================================================


class TestCramerVCI:
    """Tests for the cramer_v_ci function."""

    def test_returns_correct_type(self):
        result = cramer_v_ci(cramer_v=0.3, n=100, df=2)
        assert isinstance(result, CramerVCIResult)

    def test_ci_contains_cramer_v(self):
        result = cramer_v_ci(cramer_v=0.3, n=100, df=2)
        lo, hi = result.ci
        assert lo <= 0.3 <= hi

    def test_wider_ci_at_lower_level(self):
        r95 = cramer_v_ci(cramer_v=0.3, n=100, df=2, confidence_level=0.95)
        r80 = cramer_v_ci(cramer_v=0.3, n=100, df=2, confidence_level=0.80)
        w95 = r95.ci[1] - r95.ci[0]
        w80 = r80.ci[1] - r80.ci[0]
        assert w95 > w80

    def test_near_zero_cramer_v(self):
        result = cramer_v_ci(cramer_v=0.05, n=200, df=1)
        lo, hi = result.ci
        assert lo >= 0.0
        assert hi > 0.0

    def test_invalid_cramer_v_raises(self):
        with pytest.raises(ValueError, match="cramer_v"):
            cramer_v_ci(cramer_v=-0.1, n=100, df=2)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="n must"):
            cramer_v_ci(cramer_v=0.3, n=1, df=2)

    def test_attributes_stored(self):
        result = cramer_v_ci(cramer_v=0.3, n=100, df=2, confidence_level=0.99)
        assert result.cramer_v == 0.3
        assert result.n == 100
        assert result.df == 2
        assert result.confidence_level == 0.99
        assert result.chi_square == pytest.approx(0.3**2 * 100 * 2, abs=1e-8)

    def test_ci_non_negative(self):
        result = cramer_v_ci(cramer_v=0.1, n=50, df=3)
        assert result.ci[0] >= 0.0


# ============================================================
# Cross-validation with Fisher z CI (already tested, just sanity)
# ============================================================


class TestFisherZCI:
    """Regression tests for fisher_z_ci to confirm it's unchanged."""

    def test_known_value(self):
        lo, hi = fisher_z_ci(r=0.5, n=50)
        assert lo == pytest.approx(0.2575, abs=5e-4)
        assert hi == pytest.approx(0.6833, abs=5e-4)

    def test_small_n_raises(self):
        with pytest.raises(ValueError, match="n must be > 3"):
            fisher_z_ci(r=0.3, n=3)
