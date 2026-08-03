"""Gwet's AC agreement family.

Migrated from ``stats/Calculator/MeasureAgreements/Gwet.ipynb`` on the ``dev``
branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ...core import InvalidInputError, StatisticalComputationError
from ...core.validation import validate_confidence_level
from ._weights import (
    WeightMethod,
    build_weight_matrix,
    coerce_ratings_matrix,
    subjects_by_category_counts,
)


@dataclass(frozen=True)
class GwetACResult:
    """Result of a Gwet AC analysis."""

    ac: float
    standard_error: float
    t_statistic: float
    p_value: float
    confidence_level: float
    ci: tuple[float, float]
    weight_method: str
    n_subjects: int
    n_raters: int
    n_categories: int


class GwetAC:
    """Gwet's AC1 / weighted AC for subjects × raters ratings."""

    @staticmethod
    def from_data(
        data: np.ndarray | list,
        weight_method: WeightMethod = "unweighted",
        confidence_level: float = 0.95,
    ) -> GwetACResult:
        """Compute Gwet's AC.

        Parameters
        ----------
        data:
            Subjects × raters matrix.  Missing ratings may be ``None``, ``""``,
            or ``NaN``.
        weight_method:
            Agreement weighting scheme.
        confidence_level:
            Confidence level in ``(0, 1)``.
        """
        validate_confidence_level(confidence_level)
        ratings, categories = coerce_ratings_matrix(data)
        n_subjects, n_raters = ratings.shape
        n_categories = categories.size
        weights = build_weight_matrix(categories, weight_method)
        counts = subjects_by_category_counts(ratings, categories)
        weighted_counts = counts @ weights

        raters_per_subject = np.sum(counts, axis=1)
        usable = raters_per_subject >= 2
        if not np.any(usable):
            raise InvalidInputError(
                "At least one subject must have two or more non-missing ratings."
            )

        q = np.sum(counts * (weighted_counts - 1.0), axis=1)
        p_o = float(
            np.sum(q[usable] / (raters_per_subject[usable] * (raters_per_subject[usable] - 1.0)))
            / np.sum(usable)
        )

        rater_matrix = np.tile(raters_per_subject, (n_categories, 1)).T
        # Avoid divide-by-zero for fully missing subjects
        safe = np.where(rater_matrix > 0, counts / rater_matrix, 0.0)
        expected_vec = np.mean(safe, axis=0)
        p_e = float(
            np.sum(weights)
            * np.sum(expected_vec * (1.0 - expected_vec))
            / (n_categories * (n_categories - 1))
        )
        if abs(1.0 - p_e) < 1e-15:
            raise StatisticalComputationError(
                "Expected agreement is 1; Gwet AC is undefined."
            )

        ac = (p_o - p_e) / (1.0 - p_e)

        denom = raters_per_subject * (raters_per_subject - 1.0)
        denom = np.where(denom == 0, np.nan, denom)
        var_po = q / denom
        expected_matrix = np.tile(expected_vec, (n_subjects, 1))
        var_pe = np.sum(
            (np.sum(weights) / (n_categories * (n_categories - 1)))
            * (counts * (1.0 - expected_matrix))
            / np.where(rater_matrix > 0, rater_matrix, np.nan),
            axis=1,
        )
        var_ac1 = (
            (n_subjects / np.sum(usable))
            * (var_po - (p_e * usable.astype(float)))
            / (1.0 - p_e)
        )
        variance_vec = var_ac1 - 2.0 * (1.0 - ac) * (var_pe - p_e) / (1.0 - p_e)
        finite = np.isfinite(variance_vec)
        se = float(
            np.sqrt(
                (1.0 / (n_subjects * (n_subjects - 1)))
                * np.sum((variance_vec[finite] - ac) ** 2)
            )
        )
        if se == 0.0:
            raise StatisticalComputationError("Gwet AC standard error is zero.")

        df = n_subjects - 1
        t_stat = ac / se
        p_value = min(float(stats.t.sf(abs(t_stat), df) * 2.0), 0.99999)
        t_crit = float(stats.t.ppf(1.0 - (1.0 - confidence_level) / 2.0, df))

        return GwetACResult(
            ac=float(ac),
            standard_error=se,
            t_statistic=float(t_stat),
            p_value=p_value,
            confidence_level=float(confidence_level),
            ci=(float(ac - se * t_crit), float(ac + se * t_crit)),
            weight_method=weight_method,
            n_subjects=n_subjects,
            n_raters=n_raters,
            n_categories=n_categories,
        )
