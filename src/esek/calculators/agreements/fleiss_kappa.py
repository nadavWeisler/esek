"""Fleiss' kappa and Randolph's kappa for multiple raters.

Migrated from ``stats/Calculator/MeasureAgreements/Kappa_Fleiss.ipynb`` on the
``dev`` branch.  The notebook hardcoded demo data; this port uses the caller
matrix and derives the number of raters from each row sum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ...core import InvalidInputError, StatisticalComputationError
from ...core.validation import validate_confidence_level, validate_non_empty


@dataclass(frozen=True)
class FleissKappaResult:
    """Result of a Fleiss / Randolph kappa analysis."""

    fleiss_kappa: float
    randolph_kappa: float
    standard_error: float
    z_statistic: float
    p_value: float
    confidence_level: float
    ci: tuple[float, float]
    n_subjects: int
    n_raters: int
    n_categories: int


class FleissKappa:
    """Fleiss' kappa for subject × category rating-count matrices."""

    @staticmethod
    def from_counts(
        counts: np.ndarray | list[list[float]],
        confidence_level: float = 0.95,
    ) -> FleissKappaResult:
        """Compute Fleiss' and Randolph's kappa.

        Parameters
        ----------
        counts:
            Matrix with subjects in rows and categories in columns.  Each entry
            is the number of raters assigning that category to the subject.
            Every row must sum to the same number of raters.
        confidence_level:
            Confidence level in ``(0, 1)``.
        """
        validate_confidence_level(confidence_level)
        arr = np.asarray(counts, dtype=float)
        validate_non_empty(arr, name="counts")
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            raise InvalidInputError(
                "'counts' must be a 2-D matrix with at least 2 subjects and 2 categories."
            )
        if np.any(arr < 0):
            raise InvalidInputError("'counts' must contain only non-negative values.")

        row_sums = np.sum(arr, axis=1)
        n_raters = float(row_sums[0])
        if n_raters < 2:
            raise InvalidInputError("Each subject must be rated by at least 2 raters.")
        if not np.allclose(row_sums, n_raters):
            raise InvalidInputError(
                "Every subject row must sum to the same number of raters."
            )

        n_subjects = arr.shape[0]
        n_categories = arr.shape[1]
        total_ratings = n_subjects * n_raters

        # Observed agreement (Fleiss)
        p_bar = (
            np.sum(arr**2) - n_subjects * n_raters
        ) / (n_subjects * n_raters * (n_raters - 1))
        category_props = np.sum(arr, axis=0) / total_ratings
        p_e = float(np.sum(category_props**2))
        p_e3 = float(np.sum(category_props**3))
        p_e_randolph = 1.0 / n_categories

        if abs(1.0 - p_e) < 1e-15:
            raise StatisticalComputationError(
                "Expected agreement is 1; Fleiss' kappa is undefined."
            )

        fleiss = (p_bar - p_e) / (1.0 - p_e)
        randolph = (p_bar - p_e_randolph) / (1.0 - p_e_randolph)

        var_term1 = 2.0 / (n_subjects * n_raters * (n_raters - 1))
        var_term2 = (
            p_e
            - (2.0 * n_raters - 3.0) * p_e**2
            + 2.0 * (n_raters - 1.0) * p_e3
        )
        var_term3 = (1.0 - p_e) ** 2
        se = float(np.sqrt(max(var_term1 * (var_term2 / var_term3), 0.0)))

        z_stat = fleiss / se if se > 0 else float("nan")
        p_value = (
            min(float(stats.norm.sf(abs(z_stat)) * 2.0), 0.99999)
            if np.isfinite(z_stat)
            else float("nan")
        )
        z_crit = float(stats.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))

        return FleissKappaResult(
            fleiss_kappa=float(fleiss),
            randolph_kappa=float(randolph),
            standard_error=se,
            z_statistic=float(z_stat),
            p_value=p_value,
            confidence_level=float(confidence_level),
            ci=(float(fleiss - se * z_crit), float(fleiss + se * z_crit)),
            n_subjects=n_subjects,
            n_raters=int(n_raters),
            n_categories=n_categories,
        )
