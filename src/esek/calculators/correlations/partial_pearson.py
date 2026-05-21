"""Partial and semi-partial Pearson correlation calculator.

Migrated and refactored from:
``stats/Calculator/AssociationCorrelations/Partial_Pearson.py``
in the ``dev`` branch.

Uses ``pingouin.partial_corr`` for the core computation and wraps it with
typed result objects and cleaner validation.

Statistical assumptions:
    - Assumes a multivariate normal distribution for the CI approximation.
    - CI uses Fisher z-transform with SE = 1/√(n − 3) — an approximation;
      simulation studies suggest this is reasonable for moderate n.
    - Partial correlation controls for ALL covariate columns.
    - Semi-partial correlation controls for covariates on the X side only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartialCorrelationResult:
    """Result of a partial or semi-partial Pearson correlation.

    Attributes:
        partial_r: Partial Pearson r (controlling for all covariates).
        partial_r_ci: CI for partial r (lower, upper).
        partial_r_p_value: p-value for partial r.
        semi_partial_r: Semi-partial r (covariate control on X only).
        semi_partial_r_ci: CI for semi-partial r (lower, upper).
        semi_partial_r_p_value: p-value for semi-partial r.
        n: Sample size.
        n_covariates: Number of covariate columns controlled for.
        confidence_level: Nominal CI level.
        metadata: Additional quantities.

    Notes:
        The CI approximation (Fisher z-transform) may be inaccurate for very
        small samples or extreme partial correlations.
    """

    partial_r: float
    partial_r_ci: tuple[float, float]
    partial_r_p_value: float
    semi_partial_r: float
    semi_partial_r_ci: tuple[float, float]
    semi_partial_r_p_value: float
    n: int
    n_covariates: int
    confidence_level: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class PartialPearsonCorrelation:
    """Partial and semi-partial Pearson correlation.

    The DataFrame must contain exactly two columns named
    ``"independent_variable"`` and ``"dependent_variable"`` plus any number
    of covariate columns.  All non-IV/DV columns are treated as covariates.

    Example::

        import pandas as pd
        from esek.calculators.correlations import PartialPearsonCorrelation

        df = pd.DataFrame({
            "independent_variable": x,
            "dependent_variable": y,
            "covariate1": z1,
        })
        result = PartialPearsonCorrelation.from_data(df)
        print(result.partial_r, result.partial_r_ci)
    """

    @staticmethod
    def from_data(
        data,  # pandas DataFrame
        confidence_level: float = 0.95,
    ) -> PartialCorrelationResult:
        """Compute partial and semi-partial Pearson correlations.

        Parameters:
            data: pandas DataFrame with columns ``"independent_variable"``,
                  ``"dependent_variable"``, and any covariate columns.
            confidence_level: Nominal CI level (default 0.95).

        Returns:
            :class:`PartialCorrelationResult`.

        Raises:
            ImportError: If pingouin is not installed.
            ValueError: For missing required columns or invalid CI level.
        """
        try:
            import pingouin as pg  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("pingouin is required for PartialPearsonCorrelation.from_data.") from e

        import pandas as pd  # noqa: PLC0415

        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame.")
        required = {"independent_variable", "dependent_variable"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}.")
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in (0, 1) (got {confidence_level}).")

        covariate_cols = [c for c in data.columns if c not in required]
        n = len(data)

        if n < 5:
            raise ValueError(f"Sample size must be ≥ 5 for partial correlation (got {n}).")

        # Partial correlation (control on both sides)
        partial_out = pg.partial_corr(
            data,
            x="independent_variable",
            y="dependent_variable",
            covar=covariate_cols if covariate_cols else None,
        )
        partial_r = float(partial_out["r"].iloc[0])
        partial_p = float(partial_out["p_val"].iloc[0])

        # Semi-partial correlation (control on X side only)
        semi_out = pg.partial_corr(
            data,
            x="independent_variable",
            y="dependent_variable",
            x_covar=covariate_cols if covariate_cols else None,
        )
        semi_r = float(semi_out["r"].iloc[0])
        semi_p = float(semi_out["p_val"].iloc[0])

        z_crit = norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)
        se = 1.0 / math.sqrt(max(n - 3, 1))

        def _fisher_ci(r: float) -> tuple[float, float]:
            safe_r = max(min(r, 0.999999), -0.999999)
            zr = math.atanh(safe_r)
            return (
                max(math.tanh(zr - z_crit * se), -1.0),
                min(math.tanh(zr + z_crit * se), 1.0),
            )

        return PartialCorrelationResult(
            partial_r=round(partial_r, 6),
            partial_r_ci=tuple(round(v, 6) for v in _fisher_ci(partial_r)),  # type: ignore[arg-type]
            partial_r_p_value=round(partial_p, 6),
            semi_partial_r=round(semi_r, 6),
            semi_partial_r_ci=tuple(round(v, 6) for v in _fisher_ci(semi_r)),  # type: ignore[arg-type]
            semi_partial_r_p_value=round(semi_p, 6),
            n=int(n),
            n_covariates=len(covariate_cols),
            confidence_level=confidence_level,
            metadata={"covariate_columns": covariate_cols},
        )
