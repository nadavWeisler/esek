"""Interval-by-interval correlation calculator — Pearson r and related measures.

Migrated from ``stats/Calculator/AssociationCorrelations/IntervalRatioCorrelation.py``
in the ``dev`` branch.  Reformatted as a class with typed result objects and proper
input validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.special as special
from scipy.stats import bootstrap, f as f_dist, ncf, norm, pearsonr, t


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PearsonResult:
    """Result of a Pearson correlation analysis.

    Attributes:
        r: Pearson correlation coefficient.
        r_squared: Coefficient of determination (r²).
        t_statistic: t-statistic for the test of H₀: r = 0.
        degrees_of_freedom: df = n − 2.
        p_value: Two-tailed p-value.
        n: Sample size.
        confidence_level: Nominal CI level (0–1).
        ci_fisher: Fisher z-transformed CI (lower, upper). The most standard CI.
        ci_bonett: Bonett (2008) CI (lower, upper).
        ci_bootstrap: Percentile bootstrap CI (lower, upper).
        ci_ncp: Non-central F–based CI for η (lower, upper).
        fisher_z: Fisher z-transformation of r.
        standard_errors: Dict of approximated SEs (Fisher, Bonett, etc.).
        metadata: Additional derived quantities.
    """

    r: float
    r_squared: float
    t_statistic: float
    degrees_of_freedom: int
    p_value: float
    n: int
    confidence_level: float
    ci_fisher: tuple[float, float]
    ci_bonett: tuple[float, float]
    ci_bootstrap: tuple[float, float]
    ci_ncp: tuple[float, float]
    fisher_z: float
    standard_errors: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helper — Non-central F confidence interval
# ---------------------------------------------------------------------------


def _ncp_f_ci(
    f_statistic: float,
    df1: int,
    df2: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Compute the non-central-F CI for a given F-statistic.

    Uses bisection search on the non-central F CDF, matching the algorithm
    in the original ``Non_Central_CI_F`` function.

    References:
        Steiger (2004); Smithson (2001).
    """
    alpha = 1.0 - confidence_level
    upper_tail = 1.0 - alpha / 2  # e.g. 0.975 for 95% CI
    lower_tail = alpha / 2

    tol = 1e-7

    # --- lower bound of NCP ---
    # Find ncp_lo such that P(F(df1,df2,ncp_lo) >= f_stat) = upper_tail
    lo_bounds = [1e-6, f_statistic / 2.0, f_statistic]
    if ncf.cdf(f_statistic, df1, df2, lo_bounds[0]) < upper_tail:
        ncp_lo = 0.0
    else:
        # Expand upper bracket until CDF drops below target
        while ncf.cdf(f_statistic, df1, df2, lo_bounds[2]) > upper_tail:
            lo_bounds = [lo_bounds[0], lo_bounds[2], lo_bounds[2] + f_statistic]
        diff = 1.0
        while diff > tol:
            mid = (lo_bounds[0] + lo_bounds[1]) / 2.0
            if ncf.cdf(f_statistic, df1, df2, mid) < upper_tail:
                lo_bounds = [lo_bounds[0], mid, lo_bounds[1]]
            else:
                lo_bounds = [mid, (mid + lo_bounds[2]) / 2.0, lo_bounds[2]]
            diff = abs(ncf.cdf(f_statistic, df1, df2, lo_bounds[1]) - upper_tail)
        ncp_lo = lo_bounds[1]

    # --- upper bound of NCP ---
    hi_bounds = [f_statistic, 2.0 * f_statistic, 3.0 * f_statistic]
    while ncf.cdf(f_statistic, df1, df2, hi_bounds[0]) < lower_tail:
        hi_bounds = [hi_bounds[0] / 4.0, hi_bounds[0], hi_bounds[2]]
    while ncf.cdf(f_statistic, df1, df2, hi_bounds[2]) > lower_tail:
        hi_bounds = [hi_bounds[0], hi_bounds[2], hi_bounds[2] + f_statistic]
    diff = 1.0
    while diff > 1e-5:
        mid = (hi_bounds[0] + hi_bounds[1]) / 2.0
        if ncf.cdf(f_statistic, df1, df2, mid) < lower_tail:
            hi_bounds = [hi_bounds[0], mid, hi_bounds[1]]
        else:
            hi_bounds = [mid, (mid + hi_bounds[2]) / 2.0, hi_bounds[2]]
        diff = abs(ncf.cdf(f_statistic, df1, df2, hi_bounds[1]) - lower_tail)
    ncp_hi = hi_bounds[1]

    return ncp_lo, ncp_hi


# ---------------------------------------------------------------------------
# Main calculator class
# ---------------------------------------------------------------------------


class PearsonCorrelation:
    """Calculate Pearson r and related measures for interval/ratio data.

    The class exposes a single static method, :meth:`from_data`, which
    accepts two numeric arrays and an optional confidence level.

    Example::

        import numpy as np
        from esek.calculators.correlations import PearsonCorrelation

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 60)
        y = 0.5 * x + rng.normal(0, 1, 60)

        result = PearsonCorrelation.from_data(x, y, confidence_level=0.95)
        print(result.r, result.ci_fisher)
    """

    @staticmethod
    def from_data(
        x: np.ndarray,
        y: np.ndarray,
        confidence_level: float = 0.95,
        bootstrap_n_resamples: int = 1000,
        bootstrap_random_state: int | None = None,
    ) -> PearsonResult:
        """Calculate Pearson r with confidence intervals.

        Parameters:
            x: First numeric variable (array-like, length ≥ 4).
            y: Second numeric variable (same length as *x*).
            confidence_level: Nominal CI level (default 0.95).
            bootstrap_n_resamples: Number of bootstrap resamples for the
                percentile CI (default 1000).
            bootstrap_random_state: Random seed for reproducibility.

        Returns:
            :class:`PearsonResult` with r, CIs, SEs, and derived statistics.

        Raises:
            ValueError: If inputs are invalid (unequal lengths, n < 4, etc.).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1-D arrays.")
        if len(x) != len(y):
            raise ValueError(f"x and y must have the same length ({len(x)} ≠ {len(y)}).")
        n = len(x)
        if n < 4:
            raise ValueError(f"Sample size must be at least 4 (got {n}).")
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in (0, 1) (got {confidence_level}).")

        # Core calculation
        r, p_value = pearsonr(x, y)
        df = n - 2
        r_sq = r**2

        # Regression-based t-statistic
        mu_x, mu_y = np.mean(x), np.mean(y)
        sd_x, sd_y = np.std(x, ddof=1), np.std(y, ddof=1)
        slope = (sd_y / sd_x) * r
        intercept = mu_y - slope * mu_x
        predicted = slope * x + intercept
        ss_total = np.sum((y - mu_y) ** 2)
        ss_res = np.sum((y - predicted) ** 2)
        ss_x = np.sum((x - mu_x) ** 2)
        se_slope = math.sqrt((1.0 / df) * (ss_res / ss_x))
        t_stat = slope / se_slope if se_slope > 0 else math.nan

        z_crit = norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)

        # --- Standard error approximations ---
        ses = PearsonCorrelation._standard_errors(r, r_sq, n)

        # --- Confidence intervals ---
        # 1. Fisher z-based CI (Fisher 1921) — most standard
        fisher_z = 0.5 * math.log((1.0 + r) / (1.0 - r)) if abs(r) < 1.0 else math.copysign(math.inf, r)
        se_zr = 1.0 / math.sqrt(n - 3) if n > 3 else math.nan
        z_lo = fisher_z - z_crit * se_zr
        z_hi = fisher_z + z_crit * se_zr
        ci_fisher = (math.tanh(z_lo), math.tanh(z_hi))

        # 2. Bonett (2008) — uses atanh; guard against r = ±1
        bonett_se = ses.get("bonett", (1.0 - r_sq) / math.sqrt(n - 3))
        if abs(r) < 1.0:
            ci_bonett = (
                math.tanh(math.atanh(r) - z_crit * bonett_se),
                math.tanh(math.atanh(r) + z_crit * bonett_se),
            )
        else:
            ci_bonett = (math.copysign(1.0, r), math.copysign(1.0, r))

        # 3. Bootstrap percentile CI
        rng = np.random.default_rng(bootstrap_random_state)
        boot_res = bootstrap(
            (x, y),
            lambda a, b: pearsonr(a, b)[0],
            n_resamples=bootstrap_n_resamples,
            vectorized=False,
            paired=True,
            random_state=rng,
            confidence_level=confidence_level,
        )
        ci_boot = (float(boot_res.confidence_interval.low), float(boot_res.confidence_interval.high))

        # 4. Non-central F CI (for eta ≈ r when df1=1)
        ci_ncp = PearsonCorrelation._ncp_ci_for_r(t_stat, df, confidence_level)

        # --- Additional derived effect sizes ---
        metadata = {
            "slope": round(slope, 6),
            "intercept": round(intercept, 6),
            "se_slope": round(se_slope, 6),
            "approximated_r_hedges_olkin": r + (r * (1.0 - r_sq)) / (2.0 * (n - 3)) if n > 3 else None,
            "common_language_effect_size_dunlap": math.asin(r) / math.pi + 0.5,
            "counter_null_effect_size": math.sqrt(4.0 * r_sq / (1.0 + 3.0 * r_sq)) if r_sq < 1.0 else 1.0,
            "ci_olivoto": PearsonCorrelation._olivoto_ci(r, n),
            "ci_olkin_finn": PearsonCorrelation._olkin_finn_ci(r_sq, n, z_crit),
        }

        return PearsonResult(
            r=float(r),
            r_squared=float(r_sq),
            t_statistic=float(t_stat),
            degrees_of_freedom=int(df),
            p_value=float(p_value),
            n=n,
            confidence_level=confidence_level,
            ci_fisher=ci_fisher,
            ci_bonett=ci_bonett,
            ci_bootstrap=ci_boot,
            ci_ncp=ci_ncp,
            fisher_z=float(fisher_z),
            standard_errors=ses,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _standard_errors(r: float, r_sq: float, n: int) -> dict[str, float]:
        """Return a dict of approximated standard errors for r.

        References: Gnambs (2023) for labels.
        """
        ses: dict[str, float] = {}
        if n < 4:
            return ses
        ses["fisher_1896"] = (1.0 - r_sq) / math.sqrt(n * (1.0 + r_sq))
        ses["fisher_filon_1898"] = (1.0 - r_sq) / math.sqrt(n)
        ses["soper_large"] = (1.0 - r_sq) / math.sqrt(n - 1)
        ses["soper_1913"] = ((1.0 - r_sq) / math.sqrt(n)) * (1.0 + (1.0 + 5.5 * r_sq) / (2.0 * n))
        ses["hotelling_1953"] = (
            ((1.0 - r_sq) / math.sqrt(n - 1))
            * (
                1.0
                + (11.0 * r_sq) / (4.0 * (n - 1))
                + (-192.0 * r_sq + 479.0 * r_sq**2) / (32.0 * (n - 1) ** 2)
            )
        )
        ses["bonett"] = (1.0 - r_sq) / math.sqrt(n - 3) if n > 3 else math.nan
        ses["regression"] = math.sqrt((1.0 - r_sq) / (n - 2)) if n > 2 else math.nan
        # Ghosh and Hedges involve hyp2f1; compute safely
        try:
            ses["ghosh"] = float(
                math.sqrt(
                    max(
                        0.0,
                        1.0
                        - ((n - 2) * (1.0 - r_sq) / (n - 1))
                        * float(special.hyp2f1(1, 1, (n + 1) / 2.0, r_sq))
                        - (
                            (2.0 / (n - 1))
                            * (math.gamma(n / 2.0) / math.gamma((n - 1) / 2.0)) ** 2
                            * math.sqrt(abs(r_sq))
                            * float(special.hyp2f1(0.5, 0.5, (n + 1) / 2.0, r_sq))
                        )
                        ** 2,
                    )
                )
            )
        except Exception:
            ses["ghosh"] = math.nan
        return ses

    @staticmethod
    def _ncp_ci_for_r(
        t_stat: float,
        df: int,
        confidence_level: float,
    ) -> tuple[float, float]:
        """Compute the non-central F–based CI for Pearson r via η."""
        if math.isnan(t_stat) or df < 1:
            return (math.nan, math.nan)
        f_stat = t_stat**2  # F = t² when df1 = 1
        ncp_lo, ncp_hi = _ncp_f_ci(f_stat, 1, df, confidence_level)
        n = df + 2
        ci_lo = math.sqrt(ncp_lo / (ncp_lo + n - 2)) if ncp_lo > 0 else 0.0
        ci_hi = math.sqrt(ncp_hi / (ncp_hi + n - 2))
        return (ci_lo, ci_hi)

    @staticmethod
    def _olivoto_ci(r: float, n: int) -> tuple[float, float]:
        """Olivoto et al. (2018) half-width CI."""
        hw = 0.45304 ** abs(r) * 2.25152 * n**-0.50089
        return (r - hw, r + hw)

    @staticmethod
    def _olkin_finn_ci(
        r_sq: float,
        n: int,
        z_crit: float,
    ) -> tuple[float, float]:
        """Olkin & Finn (1995) CI for r via R²."""
        se_r2 = math.sqrt((4.0 * r_sq * (1.0 - r_sq) ** 2 * (n - 2) ** 2) / ((n**2 - 1) * (n + 3)))
        lo = math.sqrt(max(0.0, r_sq - z_crit * se_r2))
        hi = math.sqrt(min(1.0, r_sq + z_crit * se_r2))
        return (lo, hi)
