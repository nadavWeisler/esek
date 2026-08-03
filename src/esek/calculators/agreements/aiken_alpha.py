"""Aiken's alpha for square contingency tables.

Migrated from ``stats/Calculator/MeasureAgreements/AickensAlpha.ipynb`` on the
``dev`` branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ...core import InvalidInputError, StatisticalComputationError
from ...core.validation import validate_confidence_level, validate_contingency_table


@dataclass(frozen=True)
class AikenAlphaResult:
    """Result of an Aiken's alpha analysis."""

    alpha: float
    standard_error: float
    confidence_level: float
    ci: tuple[float, float]
    n: float
    n_categories: int


class AikenAlpha:
    """Aiken's alpha via iterative marginal MLE."""

    @staticmethod
    def from_table(
        table: np.ndarray | list[list[float]] | list[float],
        confidence_level: float = 0.95,
        tol: float = 1e-5,
        max_iter: int = 500,
    ) -> AikenAlphaResult:
        """Compute Aiken's alpha from a square contingency table.

        Parameters
        ----------
        table:
            Square contingency table, or a flat length-``k²`` vector that is
            reshaped to ``k × k``.
        confidence_level:
            Confidence level in ``(0, 1)``.
        tol:
            Convergence tolerance for the iterative update.
        max_iter:
            Maximum number of iterations.
        """
        validate_confidence_level(confidence_level)
        arr = np.asarray(table, dtype=float)
        if arr.ndim == 1:
            side = int(np.sqrt(arr.size))
            if side * side != arr.size:
                raise InvalidInputError(
                    "Flat contingency input must have a perfect-square length."
                )
            arr = arr.reshape(side, side)
        validate_contingency_table(arr, name="table")
        if arr.shape[0] != arr.shape[1]:
            raise InvalidInputError(f"'table' must be square, got shape {arr.shape}.")

        k = arr.shape[0]
        weights = np.eye(k, dtype=float)
        y = arr + 1.0 / (k**2)
        n = float(np.sum(y))
        weighted_sum = float(np.sum(y * weights))
        p_o = weighted_sum / n
        row_sums = np.sum(y, axis=1)
        col_sums = np.sum(y, axis=0)
        p_rows = row_sums / n
        p_cols = col_sums / n
        p_e = float(np.sum(np.outer(p_rows, p_cols) * weights))
        if abs(1.0 - p_e) < 1e-15:
            raise StatisticalComputationError(
                "Expected agreement is 1; Aiken's alpha is undefined."
            )
        alpha = (p_o - p_e) / (1.0 - p_e)

        for _ in range(max_iter):
            previous = alpha
            pr_den = n * (1.0 - alpha + alpha * (weights @ p_cols) / p_e)
            p_rows = row_sums / pr_den
            p_rows[0] = 1.0 - np.sum(p_rows[1:])
            pc_den = n * (1.0 - alpha + alpha * (p_rows @ weights) / p_e)
            p_cols = col_sums / pc_den
            p_cols[0] = 1.0 - np.sum(p_cols[1:])
            p_e = float(np.sum(np.outer(p_rows, p_cols) * weights))
            if abs(1.0 - p_e) < 1e-15:
                raise StatisticalComputationError(
                    "Expected agreement became 1 during Aiken iteration."
                )
            alpha = (p_o - p_e) / (1.0 - p_e)
            if abs(alpha - previous) <= tol:
                break
        else:
            raise StatisticalComputationError(
                f"Aiken's alpha did not converge within {max_iter} iterations."
            )

        se = _aiken_standard_error(
            y=y,
            weights=weights,
            alpha=float(alpha),
            p_e=p_e,
            p_rows=p_rows,
            p_cols=p_cols,
            row_sums=row_sums,
            col_sums=col_sums,
            weighted_sum=weighted_sum,
            n=n,
        )
        z_crit = float(stats.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))
        return AikenAlphaResult(
            alpha=float(alpha),
            standard_error=float(se),
            confidence_level=float(confidence_level),
            ci=(float(alpha - z_crit * se), float(alpha + z_crit * se)),
            n=n,
            n_categories=k,
        )


def _aiken_standard_error(
    *,
    y: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    p_e: float,
    p_rows: np.ndarray,
    p_cols: np.ndarray,
    row_sums: np.ndarray,
    col_sums: np.ndarray,
    weighted_sum: float,
    n: float,
) -> float:
    diff_rows = p_rows[1:] - p_rows[0]
    diff_cols = p_cols[1:] - p_cols[0]
    v = alpha / p_e / ((1.0 - alpha) * p_e + alpha)
    t = 1.0 / alpha - 1.0
    d_alpha = -n * (1.0 - p_e) / (1.0 - alpha) / ((1.0 - alpha) * p_e + alpha)
    d_rows = -weighted_sum * diff_rows * ((n / weighted_sum) ** 2)
    d_cols = -weighted_sum * diff_cols * ((n / weighted_sum) ** 2)

    rows_matrix = (
        -np.sum(y[:, 0]) / (p_cols[0] ** 2)
        + weighted_sum * (2.0 * p_e * t + 1.0) * v**2 * np.outer(diff_rows, diff_rows)
    )
    rows_term = (
        -col_sums[1:] / (p_cols[1:] ** 2)
        - col_sums[0] / (p_cols[0] ** 2)
        + weighted_sum * (2.0 * p_e * t + 1.0) * v**2 * (diff_rows**2)
    )
    np.fill_diagonal(rows_matrix, rows_term)

    cols_matrix = (
        -np.sum(y[0, :]) / (p_rows[0] ** 2)
        + weighted_sum * (2.0 * p_e * t + 1.0) * v**2 * np.outer(diff_cols, diff_cols)
    )
    cols_term = (
        -row_sums[1:] / (p_rows[1:] ** 2)
        - row_sums[0] / (p_rows[0] ** 2)
        + weighted_sum * (2.0 * p_e * t + 1.0) * v**2 * (diff_cols**2)
    )
    np.fill_diagonal(cols_matrix, cols_term)

    cross = (
        -weighted_sum * v
        + weighted_sum
        * (2.0 * p_e * t + 1.0)
        * v**2
        * np.outer(diff_cols, diff_rows).T
    )
    cross_term = (
        -2.0 * weighted_sum * v
        + weighted_sum * (2.0 * p_e * t + 1.0) * v**2 * diff_rows * diff_cols
    )
    np.fill_diagonal(cross, cross_term)

    top = np.concatenate(([d_alpha], d_rows, d_cols))
    center = np.column_stack((d_rows.reshape(-1, 1), rows_matrix, cross))
    bottom = np.column_stack((d_cols.reshape(-1, 1), cross.T, cols_matrix))
    hessian = np.vstack((top, center, bottom))
    try:
        cov = np.linalg.inv(-hessian)
    except np.linalg.LinAlgError as exc:
        raise StatisticalComputationError(
            "Aiken alpha Hessian is singular; SE is undefined."
        ) from exc
    var = float(cov[0, 0])
    if var < 0:
        raise StatisticalComputationError(
            "Aiken alpha variance estimate is negative."
        )
    return float(np.sqrt(var))
