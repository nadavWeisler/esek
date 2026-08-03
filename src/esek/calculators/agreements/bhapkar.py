"""Bhapkar test of marginal homogeneity.

Migrated from ``stats/Calculator/MeasureAgreements/Bhapkar.ipynb`` on the
``dev`` branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from ...core import InvalidInputError, StatisticalComputationError
from ...core.validation import validate_contingency_table


@dataclass(frozen=True)
class BhapkarResult:
    """Result of a Bhapkar marginal-homogeneity test."""

    chi_square: float
    degrees_of_freedom: int
    p_value: float
    n: int
    n_categories: int


class BhapkarTest:
    """Bhapkar test for square contingency tables."""

    @staticmethod
    def from_table(table: np.ndarray | list[list[float]]) -> BhapkarResult:
        """Run the Bhapkar test of marginal homogeneity.

        Parameters
        ----------
        table:
            Square contingency table of paired categorical ratings.
        """
        validate_contingency_table(table, name="table")
        arr = np.asarray(table, dtype=float)
        if arr.shape[0] != arr.shape[1]:
            raise InvalidInputError(f"'table' must be square, got shape {arr.shape}.")

        n = float(np.sum(arr))
        if n <= 0:
            raise InvalidInputError("'table' must contain a positive total count.")

        k = arr.shape[0]
        row_sums = np.sum(arr, axis=1)[:-1]
        col_sums = np.sum(arr, axis=0)[:-1]
        d = row_sums - col_sums
        d_matrix = np.tile(d, (k - 1, 1)).T

        diag = np.zeros((k - 1, k - 1), dtype=float)
        np.fill_diagonal(diag, row_sums + col_sums)
        weights = arr[:-1, :-1]
        cov = diag - weights.T - weights - (d_matrix * d_matrix.T) / n
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError as exc:
            raise StatisticalComputationError(
                "Bhapkar covariance matrix is singular."
            ) from exc

        chi_sq = float(abs((d_matrix @ d_matrix.T @ inv)[0, 0]))
        df = k - 1
        p_value = float(1.0 - chi2.cdf(chi_sq, df))
        return BhapkarResult(
            chi_square=chi_sq,
            degrees_of_freedom=df,
            p_value=p_value,
            n=int(n),
            n_categories=k,
        )
