"""Krippendorff's alpha.

Migrated from ``stats/Calculator/MeasureAgreements/KrippendorfFinal.ipynb`` on
the ``dev`` branch.
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
class KrippendorffAlphaResult:
    """Result of a Krippendorff α analysis."""

    alpha: float
    alpha_prime: float
    standard_error: float
    t_statistic: float
    p_value: float
    confidence_level: float
    ci: tuple[float, float]
    weight_method: str
    n_subjects: int
    n_raters: int
    n_categories: int


class KrippendorffAlpha:
    """Krippendorff's alpha for subjects × raters ratings."""

    @staticmethod
    def from_data(
        data: np.ndarray | list,
        weight_method: WeightMethod = "unweighted",
        confidence_level: float = 0.95,
    ) -> KrippendorffAlphaResult:
        """Compute Krippendorff's α.

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
        if np.sum(usable) < 2:
            raise InvalidInputError(
                "At least two subjects must have two or more non-missing ratings."
            )

        counts_u = counts[usable]
        raters_u = raters_per_subject[usable]
        mean_raters = float(np.mean(raters_u))
        q = np.sum(counts_u * (weighted_counts[usable] - 1.0), axis=1)
        n_u = int(np.sum(usable))
        epsilon = 1.0 / float(np.sum(raters_u))

        p_o_raw = float(np.sum(q / (mean_raters * (raters_u - 1.0))) / n_u)
        p_o = (1.0 - epsilon) * p_o_raw + epsilon
        expected_vec = np.sum(counts_u / mean_raters, axis=0) / n_u
        p_e = float(np.sum(weights * np.outer(expected_vec, expected_vec)))
        if abs(1.0 - p_e) < 1e-15:
            raise StatisticalComputationError(
                "Expected agreement is 1; Krippendorff alpha is undefined."
            )

        alpha = (p_o - p_e) / (1.0 - p_e)
        alpha_prime = (p_o_raw - p_e) / (1.0 - p_e)

        term1 = q / (mean_raters * (raters_u - 1.0)) - p_o * (
            raters_u - mean_raters
        ) / mean_raters
        term2 = (term1 - p_e) / (1.0 - p_e)
        expected_matrix = np.tile(expected_vec, (n_categories, 1))
        weighted_expected = (
            np.sum(expected_matrix * weights, axis=1)
            + np.sum(expected_matrix * weights.T, axis=1)
        ) / 2.0
        expected_final = np.tile(weighted_expected, (n_u, 1))
        term3 = (
            np.sum(counts_u * expected_final, axis=1) / mean_raters
            - p_e * (raters_u - mean_raters) / mean_raters
        )
        term4 = term2 - 2.0 * (1.0 - alpha_prime) * (term3 - p_e) / (1.0 - p_e)
        se = float(
            np.sqrt((1.0 / (n_u * (n_u - 1))) * np.sum((term4 - alpha_prime) ** 2))
        )
        if se == 0.0:
            raise StatisticalComputationError(
                "Krippendorff alpha standard error is zero."
            )

        df = n_subjects - 1
        t_stat = alpha / se
        p_value = min(float(stats.t.sf(abs(t_stat), df) * 2.0), 0.99999)
        t_crit = float(stats.t.ppf(1.0 - (1.0 - confidence_level) / 2.0, df))

        return KrippendorffAlphaResult(
            alpha=float(alpha),
            alpha_prime=float(alpha_prime),
            standard_error=se,
            t_statistic=float(t_stat),
            p_value=p_value,
            confidence_level=float(confidence_level),
            ci=(float(alpha - se * t_crit), float(alpha + se * t_crit)),
            weight_method=weight_method,
            n_subjects=n_subjects,
            n_raters=n_raters,
            n_categories=n_categories,
        )
