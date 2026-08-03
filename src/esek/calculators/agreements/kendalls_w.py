"""Kendall's coefficient of concordance W.

Migrated from ``stats/Calculator/MeasureAgreements/KendallsW.ipynb`` on the
``dev`` branch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ...core import InvalidInputError
from ...core.validation import validate_non_empty


@dataclass(frozen=True)
class KendallsWResult:
    """Result of a Kendall's W analysis."""

    w: float
    w_tie_corrected: float
    mean_spearman: float
    chi_square: float
    degrees_of_freedom: int
    p_value: float
    n_subjects: int
    n_raters: int


class KendallsW:
    """Kendall's W for subject × rater rating matrices."""

    @staticmethod
    def from_data(
        data: np.ndarray | Sequence[Sequence[float]],
    ) -> KendallsWResult:
        """Compute Kendall's W from a subjects × raters matrix.

        Parameters
        ----------
        data:
            Matrix with subjects in rows and raters (or ranking sources) in
            columns.  Values are ranked within each column.
        """
        arr = np.asarray(data, dtype=float)
        validate_non_empty(arr, name="data")
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            raise InvalidInputError(
                "'data' must be a 2-D matrix with at least 2 subjects and 2 raters."
            )
        if np.any(~np.isfinite(arr)):
            raise InvalidInputError("'data' must contain only finite values.")

        n_subjects, n_raters = arr.shape
        ranked = _rank_columns(arr)
        rank_sums = np.sum(ranked, axis=1)
        ss = float(np.sum(rank_sums**2) - (np.sum(rank_sums) ** 2) / n_subjects)
        denom_raw = (n_raters**2) * (n_subjects**3 - n_subjects) / 12.0
        w = ss / denom_raw

        ties = _tie_correction(ranked)
        denom_corr = (
            (n_raters**2) * (n_subjects**3 - n_subjects) - n_raters * ties
        ) / 12.0
        if denom_corr <= 0:
            raise InvalidInputError("Tie correction leaves a non-positive denominator.")
        w_corr = ss / denom_corr
        mean_spearman = (n_raters * w_corr - 1.0) / (n_raters - 1.0)
        chi_sq = n_raters * (n_subjects - 1) * w_corr
        df = n_subjects - 1
        p_value = float(stats.chi2.sf(chi_sq, df))

        return KendallsWResult(
            w=float(w),
            w_tie_corrected=float(w_corr),
            mean_spearman=float(mean_spearman),
            chi_square=float(chi_sq),
            degrees_of_freedom=df,
            p_value=p_value,
            n_subjects=n_subjects,
            n_raters=n_raters,
        )


def _rank_columns(arr: np.ndarray) -> np.ndarray:
    ranked = np.empty_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        ranked[:, j] = stats.rankdata(arr[:, j], method="average")
    return ranked


def _tie_correction(ranked: np.ndarray) -> float:
    total = 0.0
    for j in range(ranked.shape[1]):
        _, counts = np.unique(ranked[:, j], return_counts=True)
        total += float(np.sum(counts**3 - counts))
    return total
