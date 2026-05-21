"""Multiple R² calculator with 20 adjusted R² estimators and CIs.

Migrated and refactored from:
``stats/Calculator/AssociationCorrelations/MultipleCorrelation/Multiple_R_Square.py``
in the ``dev`` branch.

Provides:
- Multiple R² from data (via sklearn OLS)
- 20 adjusted/cross-validity R² estimators
- CI for R² (Wishart/Olkin-Finn and Fisher transformation methods)
- Non-central F–based CI for R²

Statistical assumptions:
    - Ordinary least squares with p predictor columns named ``x1, x2, ...``
      and one outcome column named ``y`` (or ``predicted``).
    - All estimators assume the linear model is correctly specified.
    - Adjusted R² estimators correct for different sources of bias; they may
      differ substantially in small samples.
    - References are provided per estimator; consult Yin & Fan (2001) for
      a comprehensive comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.special as special
import scipy.optimize as opt
from scipy.stats import f as f_dist, norm, ncf


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultipleRSquaredResult:
    """Result of a multiple R² analysis.

    Attributes:
        r_squared: Raw (sample) R².
        f_statistic: Overall F-statistic for the model.
        df1: Numerator df (number of predictors).
        df2: Denominator df (n − p − 1).
        p_value: p-value for overall F-test.
        n: Sample size.
        n_predictors: Number of predictors (p).
        confidence_level: Nominal CI level.
        ci_wishart: Wishart (1931) / Olkin-Finn (1995) CI (lower, upper).
        ci_fisher: Fisher z-transformation CI (lower, upper).
        ci_ncp: Non-central F CI (lower, upper).
        adjusted_estimators: Dict of 20 adjusted/cross-validity R² estimates.
        metadata: Additional quantities.
    """

    r_squared: float
    f_statistic: float
    df1: int
    df2: int
    p_value: float
    n: int
    n_predictors: int
    confidence_level: float
    ci_wishart: tuple[float, float]
    ci_fisher: tuple[float, float]
    ci_ncp: tuple[float, float]
    adjusted_estimators: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal NCP-F helper (reuse the same algorithm as other CI modules)
# ---------------------------------------------------------------------------


def _ncp_f_ci(
    f_statistic: float,
    df1: int,
    df2: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Non-central F NCP CI via bisection."""
    alpha = 1.0 - confidence_level
    upper_tail = 1.0 - alpha / 2
    lower_tail = alpha / 2
    tol_lo, tol_hi = 1e-7, 1e-5

    lo = [1e-3, f_statistic / 2.0, f_statistic]
    if ncf.cdf(f_statistic, df1, df2, lo[0]) < upper_tail:
        ncp_lo = 0.0
    else:
        while ncf.cdf(f_statistic, df1, df2, lo[2]) > upper_tail:
            lo = [lo[0], lo[2], lo[2] + f_statistic]
        diff = 1.0
        while diff > tol_lo:
            if ncf.cdf(f_statistic, df1, df2, lo[1]) < upper_tail:
                lo = [lo[0], (lo[0] + lo[1]) / 2.0, lo[1]]
            else:
                lo = [lo[1], (lo[1] + lo[2]) / 2.0, lo[2]]
            diff = abs(ncf.cdf(f_statistic, df1, df2, lo[1]) - upper_tail)
        ncp_lo = lo[1]

    hi = [f_statistic, 2.0 * f_statistic, 3.0 * f_statistic]
    while ncf.cdf(f_statistic, df1, df2, hi[0]) < lower_tail:
        hi = [hi[0] / 4.0, hi[0], hi[2]]
    while ncf.cdf(f_statistic, df1, df2, hi[2]) > lower_tail:
        hi = [hi[0], hi[2], hi[2] + f_statistic]
    diff = 1.0
    while diff > tol_hi:
        if ncf.cdf(f_statistic, df1, df2, hi[1]) < lower_tail:
            hi = [hi[0], (hi[0] + hi[1]) / 2.0, hi[1]]
        else:
            hi = [hi[1], (hi[1] + hi[2]) / 2.0, hi[2]]
        diff = abs(ncf.cdf(f_statistic, df1, df2, hi[1]) - lower_tail)
    ncp_hi = hi[1]

    return ncp_lo, ncp_hi


# ---------------------------------------------------------------------------
# Adjusted R² estimator functions
# ---------------------------------------------------------------------------


def compute_adjusted_r_squared(
    r_squared: float,
    n: int,
    p: int,
) -> dict[str, float]:
    """Compute 20 adjusted and cross-validity R² estimators.

    Parameters:
        r_squared: Observed R² (0 ≤ r² ≤ 1).
        n: Sample size.
        p: Number of predictors.

    Returns:
        Dict mapping estimator name → adjusted R² value.

    References:
        Yin & Fan (2001) for a comprehensive simulation comparison.
        Smith (1928/9), Ezekiel (1930), Wherry (1931), Olkin & Pratt (1958),
        Pratt (1964), Herzberg (1969), Claudy (1978), Cattin (1980),
        Alf & Graf (2002), Lord/Uhl & Eisenberg (1950/1970),
        Lord-Nicholson (1960), Darlington-Stein (1967/1960), Burket (1964),
        Brown (1975), Rozeboom (1978, 1981), Claudy-I (1978).
    """
    df_total = n - 1
    df_residual = n - p - 1
    df = n - p

    if df_residual <= 0 or df <= 0:
        return {}

    q = 1.0 - r_squared
    term1 = (n - 3) * q / df_residual

    estimators: dict[str, float] = {}

    # --- Adjusted R² estimators ---
    estimators["smith_1929"] = 1.0 - (n / df) * q
    ezekiel = 1.0 - (df_total / df_residual) * q
    estimators["ezekiel_1930"] = ezekiel
    estimators["wherry_1931"] = 1.0 - (df_total / df) * q

    try:
        op = 1.0 - term1 * float(special.hyp2f1(1, 1, (df + 1) / 2.0, q))
        estimators["olkin_pratt_1958"] = op
        estimators["pratt_1964"] = 1.0 - (((n - 3) * q) / df_residual) * (1.0 + (2.0 * q) / (df - 2.3))
        estimators["herzberg_1969"] = 1.0 - (((n - 3) * q) / df_residual) * (1.0 + (2.0 * q) / (df + 1))
        estimators["claudy_1978"] = 1.0 - (((n - 4) * q) / df_residual) * (1.0 + (2.0 * q) / (df + 1))
        estimators["cattin_1980"] = 1.0 - term1 * ((1.0 + (2.0 * q) / df_residual) + ((8.0 * q**2) / (df_residual * (df + 3))))
    except Exception:
        pass

    # Alf & Graf (2002) MLE
    try:
        result = opt.minimize_scalar(
            lambda rho: -(1.0 - rho) ** (n / 2.0) * float(special.hyp2f1(0.5 * n, 0.5 * n, 0.5 * p, rho * r_squared)),
            bounds=(0.0, 1.0),
            method="bounded",
        )
        estimators["alf_graf_2002_mle"] = float(result.x)
    except Exception:
        pass

    # --- Cross-validity / shrinkage estimators ---
    estimators["lord_1950"] = 1.0 - (n + p + 1) / (n - p - 1) * q
    estimators["lord_nicholson_1960"] = 1.0 - ((n + p + 1) / n) * (df_total / df_residual) * q
    if df > 2:
        estimators["darlington_stein_1967"] = 1.0 - ((n + 1) / n) * (df_total / df_residual) * ((n - 2) / (df - 2)) * q
    denom_burket = math.sqrt(max(r_squared, 1e-12)) * df
    estimators["burket_1964"] = (n * r_squared - p) / denom_burket if denom_burket > 0 else math.nan
    denom_brown_large = (n - 2 * p - 2) * ezekiel + p
    estimators["brown_large_1975"] = (((df - 3) * ezekiel**2 + ezekiel) / denom_brown_large) if denom_brown_large != 0 else math.nan
    if "olkin_pratt_1958" in estimators:
        op = estimators["olkin_pratt_1958"]
        denom_brown_small = (n - 2 * p - 2) * op + p
        estimators["brown_small_1975"] = (((df - 3) * op**2 + op) / denom_brown_small) if denom_brown_small != 0 else math.nan
        estimators["rozeboom2_small_1981"] = op * (1.0 + (p / (df - 2)) * ((1.0 - op) / max(op, 1e-12)))**-1 if df > 2 else math.nan
        estimators["claudy1_small_1978"] = 2.0 * op - r_squared
    estimators["rozeboom_1978"] = 1.0 - ((n + p) / df) * q
    estimators["rozeboom2_large_1981"] = ezekiel * (1.0 + (p / (df - 2)) * ((1.0 - ezekiel) / max(ezekiel, 1e-12)))**-1 if df > 2 else math.nan
    estimators["claudy1_large_1978"] = 2.0 * ezekiel - r_squared

    return {k: round(float(v), 6) for k, v in estimators.items() if not math.isnan(v)}


# ---------------------------------------------------------------------------
# Public calculator
# ---------------------------------------------------------------------------


class MultipleRSquared:
    """Calculate multiple R² and adjusted R² from data or summary statistics.

    Example::

        import pandas as pd
        from esek.calculators.correlations import MultipleRSquared

        df = pd.DataFrame({"x1": [...], "x2": [...], "y": [...]})
        result = MultipleRSquared.from_data(df, outcome_col="y")
        print(result.r_squared, result.adjusted_estimators["ezekiel_1930"])
    """

    @staticmethod
    def from_data(
        data,  # pandas DataFrame
        outcome_col: str = "y",
        confidence_level: float = 0.95,
    ) -> MultipleRSquaredResult:
        """Fit OLS and compute R² with CIs and adjusted estimates.

        Parameters:
            data: pandas DataFrame with predictor columns and one outcome column.
                  The outcome column is specified by ``outcome_col``.
                  All other columns are used as predictors.
            outcome_col: Name of the outcome/dependent variable column.
            confidence_level: Nominal CI level (default 0.95).

        Returns:
            :class:`MultipleRSquaredResult`.

        Raises:
            ImportError: If sklearn is not installed.
            ValueError: For invalid inputs.
        """
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError as e:
            raise ImportError("scikit-learn is required for MultipleRSquared.from_data.") from e

        import pandas as pd  # noqa: PLC0415

        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame.")
        if outcome_col not in data.columns:
            raise ValueError(f"outcome_col '{outcome_col}' not found in DataFrame.")
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in (0, 1) (got {confidence_level}).")

        y = data[outcome_col].to_numpy(dtype=float)
        predictor_cols = [c for c in data.columns if c != outcome_col]
        if not predictor_cols:
            raise ValueError("DataFrame must have at least one predictor column.")

        X = data[predictor_cols].to_numpy(dtype=float)
        n, p = X.shape

        if n < p + 2:
            raise ValueError(f"Sample size ({n}) must be > number of predictors + 1 ({p + 1}).")

        r_sq = LinearRegression().fit(X, y).score(X, y)
        return MultipleRSquared.from_r_squared(
            r_squared=r_sq,
            n=n,
            p=p,
            confidence_level=confidence_level,
        )

    @staticmethod
    def from_r_squared(
        r_squared: float,
        n: int,
        p: int,
        confidence_level: float = 0.95,
    ) -> MultipleRSquaredResult:
        """Compute CIs and adjusted estimates from a known R² value.

        Parameters:
            r_squared: Observed R² (0 ≤ r² ≤ 1).
            n: Sample size.
            p: Number of predictors.
            confidence_level: Nominal CI level.

        Returns:
            :class:`MultipleRSquaredResult`.
        """
        if not (0.0 <= r_squared <= 1.0):
            raise ValueError(f"r_squared must be in [0, 1] (got {r_squared}).")
        if n < p + 2:
            raise ValueError(f"n ({n}) must be > p + 1 ({p + 1}).")
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in (0, 1) (got {confidence_level}).")

        df1 = p
        df2 = n - p - 1
        q = 1.0 - r_squared
        f_stat = (r_squared / p) / (q / df2) if (q * df2 > 0) else math.nan
        p_value = float(f_dist.sf(f_stat, df1, df2)) if not math.isnan(f_stat) else math.nan
        z_crit = norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)

        # CI 1 — Wishart (1931) / Olkin & Finn (1995) formula
        ase_wishart = math.sqrt((4.0 * r_squared * q**2 * df2**2) / ((n**2 - 1) * (n + 3)))
        ci_wishart = (
            max(0.0, r_squared - z_crit * ase_wishart),
            min(1.0, r_squared + z_crit * ase_wishart),
        )

        # CI 2 — Fisher z-transform (Algina, 1999 approximation for R²)
        if 0.0 < r_squared < 1.0:
            zr2 = math.log((1.0 + math.sqrt(r_squared)) / (1.0 - math.sqrt(r_squared)))
            ase_algina = math.sqrt(4.0 / n)
            z_lo = zr2 - z_crit * ase_algina
            z_hi = zr2 + z_crit * ase_algina
            lo_fisher = max(0.0, ((math.exp(z_lo) - 1) / (math.exp(z_lo) + 1))**2)
            hi_fisher = min(1.0, ((math.exp(z_hi) - 1) / (math.exp(z_hi) + 1))**2)
            ci_fisher = (lo_fisher, hi_fisher)
        elif r_squared >= 1.0:
            ci_fisher = (1.0, 1.0)
        else:
            ci_fisher = (0.0, 0.0)

        # CI 3 — Non-central F
        if not math.isnan(f_stat) and math.isfinite(f_stat) and f_stat > 0:
            ncp_lo, ncp_hi = _ncp_f_ci(f_stat, df1, df2, confidence_level)
            total_df = df1 + df2 + 1
            ci_ncp = (
                math.sqrt(max(0.0, ncp_lo / (ncp_lo + total_df))),
                math.sqrt(min(1.0, ncp_hi / (ncp_hi + total_df))),
            )
        elif r_squared >= 1.0:
            ci_ncp = (1.0, 1.0)
        else:
            ci_ncp = (0.0, 1.0)

        # Adjusted estimators
        adjusted = compute_adjusted_r_squared(r_squared, n, p)

        return MultipleRSquaredResult(
            r_squared=round(float(r_squared), 6),
            f_statistic=round(float(f_stat), 6) if not math.isnan(f_stat) else math.nan,
            df1=int(df1),
            df2=int(df2),
            p_value=round(float(p_value), 6) if not math.isnan(p_value) else math.nan,
            n=int(n),
            n_predictors=int(p),
            confidence_level=confidence_level,
            ci_wishart=(round(ci_wishart[0], 6), round(ci_wishart[1], 6)),
            ci_fisher=(round(ci_fisher[0], 6), round(ci_fisher[1], 6)),
            ci_ncp=(round(ci_ncp[0], 6), round(ci_ncp[1], 6)),
            adjusted_estimators=adjusted,
        )
