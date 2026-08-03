"""Cohen's kappa (unweighted and weighted) for two raters.

Migrated from ``stats/Calculator/MeasureAgreements/CohensKappa.ipynb`` and
``Weighted_CohensKappa.ipynb`` on the ``dev`` branch.

References
----------
- Cohen (1960) A coefficient of agreement for nominal scales
- Fleiss, Cohen & Everitt (1969) Large sample standard errors of kappa
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from ...core import InvalidInputError, StatisticalComputationError
from ...core.validation import validate_confidence_level, validate_contingency_table

WeightType = Literal["unweighted", "linear", "quadratic"]


@dataclass(frozen=True)
class CohensKappaResult:
    """Result of a Cohen's kappa analysis."""

    kappa: float
    observed_agreement: float
    expected_agreement: float
    standard_error: float
    standard_error_h0: float
    z_statistic: float
    p_value: float
    z_statistic_h0: float
    p_value_h0: float
    confidence_level: float
    ci: tuple[float, float]
    ci_h0: tuple[float, float]
    weight_type: str
    n: int
    n_categories: int


class CohensKappa:
    """Cohen's kappa for a square contingency table (two raters)."""

    @staticmethod
    def from_table(
        table: np.ndarray | list[list[float]],
        confidence_level: float = 0.95,
        weight_type: WeightType = "unweighted",
    ) -> CohensKappaResult:
        """Compute Cohen's kappa from a square contingency table.

        Parameters
        ----------
        table:
            Square contingency table of rating counts.
        confidence_level:
            Confidence level in ``(0, 1)``.
        weight_type:
            ``"unweighted"``, ``"linear"`` (equal-spacing), or ``"quadratic"``
            (Fleiss-Cohen weights).
        """
        validate_contingency_table(table, name="table")
        validate_confidence_level(confidence_level)
        arr = np.asarray(table, dtype=float)
        if arr.shape[0] != arr.shape[1]:
            raise InvalidInputError(
                f"'table' must be square, got shape {arr.shape}."
            )
        if weight_type not in ("unweighted", "linear", "quadratic"):
            raise InvalidInputError(
                "weight_type must be 'unweighted', 'linear', or 'quadratic'."
            )

        n = float(np.sum(arr))
        if n <= 0:
            raise InvalidInputError("'table' must contain a positive total count.")

        k = arr.shape[0]
        weights = _agreement_weights(k, weight_type)
        row_props = np.sum(arr, axis=1) / n
        col_props = np.sum(arr, axis=0) / n
        p_o = float(np.sum(weights * arr) / n)
        p_e = float(np.sum(weights * np.outer(row_props, col_props)))
        if abs(1.0 - p_e) < 1e-15:
            raise StatisticalComputationError(
                "Expected agreement is 1; Cohen's kappa is undefined."
            )

        kappa = (p_o - p_e) / (1.0 - p_e)
        se, se_h0 = _kappa_standard_errors(arr, weights, kappa, p_e, n)
        z_crit = float(stats.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))

        z_stat = kappa / se if se > 0 else float("nan")
        z_h0 = kappa / se_h0 if se_h0 > 0 else float("nan")
        p_value = _z_p_value(z_stat)
        p_h0 = _z_p_value(z_h0)

        return CohensKappaResult(
            kappa=float(kappa),
            observed_agreement=p_o,
            expected_agreement=p_e,
            standard_error=float(se),
            standard_error_h0=float(se_h0),
            z_statistic=float(z_stat),
            p_value=p_value,
            z_statistic_h0=float(z_h0),
            p_value_h0=p_h0,
            confidence_level=float(confidence_level),
            ci=(float(kappa - se * z_crit), float(kappa + se * z_crit)),
            ci_h0=(float(kappa - se_h0 * z_crit), float(kappa + se_h0 * z_crit)),
            weight_type=weight_type,
            n=int(n),
            n_categories=k,
        )


def _agreement_weights(k: int, weight_type: WeightType) -> np.ndarray:
    if weight_type == "unweighted":
        return np.eye(k, dtype=float)
    idx = np.arange(1, k + 1, dtype=float)
    diffs = np.abs(np.subtract.outer(idx, idx)) / (k - 1)
    if weight_type == "linear":
        return 1.0 - diffs
    return 1.0 - diffs**2


def _kappa_standard_errors(
    table: np.ndarray,
    weights: np.ndarray,
    kappa: float,
    p_e: float,
    n: float,
) -> tuple[float, float]:
    """Fleiss–Cohen–Everitt (1969) SE and H0 SE."""
    probs = table / n
    row_props = np.sum(probs, axis=1)
    col_props = np.sum(probs, axis=0)

    # Weighted SE (Fleiss, Cohen & Everitt)
    w_row = weights @ col_props
    w_col = weights.T @ row_props
    variance_matrix = weights - np.add.outer(w_col, w_row) * (1.0 - kappa)
    # For unweighted this reduces to the classic diagonal-only form.
    term = np.sum(probs * variance_matrix**2) - (kappa - p_e * (1.0 - kappa)) ** 2
    se = math_sqrt(term / ((1.0 - p_e) ** 2) / n)

    # SE under H0
    outer = np.outer(row_props, col_props)
    w_rows_h0 = weights @ col_props
    w_cols_h0 = weights.T @ row_props
    weighted_var = (weights - np.add.outer(w_rows_h0, w_cols_h0)) ** 2
    # Use category marginals for unweighted H0 formula parity with notebook
    if np.allclose(weights, np.eye(weights.shape[0])):
        agreement = np.eye(weights.shape[0])
        outer_t = np.outer(col_props, row_props).T
        outer_add = np.add.outer(row_props, col_props)
        variance_h0 = outer_t * (agreement - outer_add) ** 2
        se_h0 = math_sqrt(
            (np.sum(variance_h0) - p_e**2) / (n * (1.0 - p_e) ** 2)
        )
    else:
        se_h0 = math_sqrt(
            (np.sum(outer * weighted_var) - p_e**2) / (n * (1.0 - p_e) ** 2)
        )
    return float(se), float(se_h0)


def math_sqrt(value: float) -> float:
    if value < 0:
        value = 0.0
    return float(np.sqrt(value))


def _z_p_value(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return min(float(stats.norm.sf(abs(z)) * 2.0), 0.99999)
