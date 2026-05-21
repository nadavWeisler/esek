"""Differences between categorical association measures.

This module migrates the legacy calculator for comparing Goodman-Kruskal Lambda
and Goodman-Kruskal Tau across two independent contingency tables or two pairs
of categorical columns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
from scipy.stats import norm

from ...core import InvalidInputError, StatisticalComputationError
from ...core.validation import (
    validate_confidence_level,
    validate_contingency_table,
    validate_groups_equal_length,
    validate_non_empty,
    validate_proportion,
    validate_sample_size,
)


@dataclass(frozen=True)
class ConfidenceInterval:
    """A simple immutable confidence interval."""

    lower: float
    upper: float


@dataclass(frozen=True)
class AssociationEstimate:
    """One association estimate with inference details."""

    value: float
    standard_error: float
    statistic: float
    p_value: float
    confidence_interval: ConfidenceInterval


@dataclass(frozen=True)
class LambdaTauResult:
    """Lambda and Tau results for a single contingency table."""

    sample_size: int
    confidence_level: float
    contingency_table: tuple[tuple[int, ...], ...]
    lambda_rows: AssociationEstimate
    lambda_columns: AssociationEstimate
    lambda_symmetric: AssociationEstimate
    tau_rows: AssociationEstimate
    tau_columns: AssociationEstimate
    tau_symmetric: AssociationEstimate


@dataclass(frozen=True)
class DifferenceEstimate:
    """Difference test between two association estimates."""

    difference: float
    standard_error: float
    z_statistic: float
    p_value: float


@dataclass(frozen=True)
class CategoricalDifferenceResult:
    """Comparison result for two categorical-association structures."""

    confidence_level: float
    group1: LambdaTauResult
    group2: LambdaTauResult
    lambda_rows_difference: DifferenceEstimate
    lambda_columns_difference: DifferenceEstimate
    lambda_symmetric_difference: DifferenceEstimate
    tau_rows_difference: DifferenceEstimate
    tau_columns_difference: DifferenceEstimate
    tau_symmetric_difference: DifferenceEstimate


class CategoricalAssociationDifference:
    """Compare categorical association measures across two samples."""

    @staticmethod
    def from_data(
        col1: Sequence[object],
        col2: Sequence[object],
        col3: Sequence[object],
        col4: Sequence[object],
        confidence_level: float = 0.95,
    ) -> CategoricalDifferenceResult:
        """Compare Lambda and Tau from two pairs of categorical columns."""
        validate_confidence_level(confidence_level)
        table1 = _columns_to_contingency(col1, col2)
        table2 = _columns_to_contingency(col3, col4)
        return CategoricalAssociationDifference.from_contingency_tables(
            table1,
            table2,
            confidence_level,
        )

    @staticmethod
    def from_contingency_tables(
        table1: Sequence[Sequence[int]],
        table2: Sequence[Sequence[int]],
        confidence_level: float = 0.95,
    ) -> CategoricalDifferenceResult:
        """Compare Lambda and Tau from two contingency tables."""
        validate_confidence_level(confidence_level)
        group1 = _analyze_table(table1, confidence_level)
        group2 = _analyze_table(table2, confidence_level)

        return CategoricalDifferenceResult(
            confidence_level=float(confidence_level),
            group1=group1,
            group2=group2,
            lambda_rows_difference=_difference(group1.lambda_rows, group2.lambda_rows),
            lambda_columns_difference=_difference(
                group1.lambda_columns,
                group2.lambda_columns,
            ),
            lambda_symmetric_difference=_difference(
                group1.lambda_symmetric,
                group2.lambda_symmetric,
            ),
            tau_rows_difference=_difference(group1.tau_rows, group2.tau_rows),
            tau_columns_difference=_difference(group1.tau_columns, group2.tau_columns),
            tau_symmetric_difference=_difference(
                group1.tau_symmetric,
                group2.tau_symmetric,
            ),
        )


def _analyze_table(
    table: Sequence[Sequence[int]],
    confidence_level: float,
) -> LambdaTauResult:
    """Compute Lambda and Tau summaries for one contingency table."""
    contingency = _validated_contingency(table)
    lambda_rows, lambda_columns, lambda_symmetric = _goodman_kruskal_lambda(
        contingency,
        confidence_level,
    )
    tau_rows, tau_columns, tau_symmetric = _goodman_kruskal_tau(
        contingency,
        confidence_level,
    )

    return LambdaTauResult(
        sample_size=int(contingency.sum()),
        confidence_level=float(confidence_level),
        contingency_table=_table_to_tuple(contingency),
        lambda_rows=lambda_rows,
        lambda_columns=lambda_columns,
        lambda_symmetric=lambda_symmetric,
        tau_rows=tau_rows,
        tau_columns=tau_columns,
        tau_symmetric=tau_symmetric,
    )


def _goodman_kruskal_lambda(
    matrix: np.ndarray,
    confidence_level: float,
) -> tuple[AssociationEstimate, AssociationEstimate, AssociationEstimate]:
    """Compute Goodman-Kruskal Lambda using Hartwig's tie-aware formulas.

    When ties yield multiple admissible standard errors, the implementation uses
    the maximum candidate to keep the result deterministic and conservative.
    """
    matrix = _validated_contingency(matrix)
    n = float(matrix.sum())
    csum = np.sum(matrix, axis=0)
    rsum = np.sum(matrix, axis=1)

    nrc = float(np.sum(np.max(matrix, axis=1)))
    nkc = float(np.sum(np.max(matrix, axis=0)))
    nrm = float(np.max(rsum))
    nkm = float(np.max(csum))
    um = nrm + nkm
    uc = nrc + nkc

    lambda_row = _safe_ratio(nrc - nkm, n - nkm)
    lambda_col = _safe_ratio(nkc - nrm, n - nrm)
    lambda_symmetric = _safe_ratio(nrc + nkc - nrm - nkm, (2.0 * n) - nrm - nkm)

    rows_with_largest_rsum = np.where(rsum == nrm)[0]
    cols_with_largest_csum = np.where(csum == nkm)[0]

    largest_rows_vector = np.asarray(
        [float(np.max(matrix[row_idx, :])) for row_idx in rows_with_largest_rsum],
        dtype=float,
    )
    largest_cols_vector = np.asarray(
        [float(np.max(matrix[:, col_idx])) for col_idx in cols_with_largest_csum],
        dtype=float,
    )

    nks_nrs = list(product(largest_rows_vector, largest_cols_vector))
    nk_tag = [float(item[0]) for item in nks_nrs]
    nr_tag = [float(item[1]) for item in nks_nrs]

    n_tags = [
        float(matrix[row_idx, col_idx])
        for row_idx, col_idx in product(rows_with_largest_rsum, cols_with_largest_csum)
    ]

    sum_of_highest_values_rows = []
    for row_idx in rows_with_largest_rsum:
        values = [
            float(matrix[row_idx, col_idx])
            for col_idx in range(matrix.shape[1])
            if matrix[row_idx, col_idx] == np.max(matrix[:, col_idx])
        ]
        sum_of_highest_values_rows.append(sum(values))

    sum_of_highest_values_cols = []
    for col_idx in cols_with_largest_csum:
        values = [
            float(matrix[row_idx, col_idx])
            for row_idx in range(matrix.shape[0])
            if matrix[row_idx, col_idx] == np.max(matrix[row_idx, :])
        ]
        sum_of_highest_values_cols.append(sum(values))

    skcr_srck = list(product(sum_of_highest_values_rows, sum_of_highest_values_cols))
    skcr = [float(item[0]) for item in skcr_srck]
    srck = [float(item[1]) for item in skcr_srck]

    srk = float(
        np.sum(
            [
                matrix[row_idx, col_idx]
                for row_idx in range(matrix.shape[0])
                for col_idx in range(matrix.shape[1])
                if matrix[row_idx, col_idx] == np.max(matrix[row_idx, :])
                and matrix[row_idx, col_idx] == np.max(matrix[:, col_idx])
            ]
        )
    )

    utag = [skcr_i + srck_i + nr_tag_i + nk_tag_i for skcr_i, srck_i, nr_tag_i, nk_tag_i in zip(skcr, srck, nr_tag, nk_tag)]

    se_rows_candidates = [
        0.0
        if lambda_row == 0.0
        else _sqrt_nonnegative(
            ((n - nrc) * (nrc + nkm - (2.0 * srck_value))) / ((n - nkm) ** 3)
        )
        for srck_value in srck
    ]
    se_cols_candidates = [
        0.0
        if lambda_col == 0.0
        else _sqrt_nonnegative(
            ((n - nkc) * (nkc + nrm - (2.0 * skcr_value))) / ((n - nrm) ** 3)
        )
        for skcr_value in skcr
    ]
    se_sym_candidates = [
        0.0
        if lambda_symmetric == 0.0
        else _sqrt_nonnegative(
            (
                ((2.0 * n) - um) * ((2.0 * n) - uc) * (um + uc + (4.0 * n) - (2.0 * utag_value))
                - (2.0 * (((2.0 * n) - um) ** 2) * (n - srk))
                - (2.0 * (((2.0 * n) - uc) ** 2) * (n - ntag_value))
            )
            / (((2.0 * n) - um) ** 4)
        )
        for utag_value, ntag_value in zip(utag, n_tags)
    ]

    return (
        _association_from_candidates(lambda_row, se_rows_candidates, confidence_level),
        _association_from_candidates(lambda_col, se_cols_candidates, confidence_level),
        _association_from_candidates(
            lambda_symmetric,
            se_sym_candidates,
            confidence_level,
        ),
    )


def _goodman_kruskal_tau(
    matrix: np.ndarray,
    confidence_level: float,
) -> tuple[AssociationEstimate, AssociationEstimate, AssociationEstimate]:
    """Compute Goodman-Kruskal Tau using the legacy asymptotic formulas."""
    matrix = _validated_contingency(matrix)
    sample_size = float(matrix.sum())
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)

    conditional_errors_columns = (sample_size**2) - float(np.sum(col_sums**2))
    conditional_errors_rows = (sample_size**2) - float(np.sum(row_sums**2))
    mean_rows = _safe_ratio(conditional_errors_rows, sample_size**2)
    mean_columns = _safe_ratio(conditional_errors_columns, sample_size**2)

    unconditional_error_rows = (sample_size**2) - (
        sample_size * float(np.sum((matrix[:, np.newaxis] ** 2) / col_sums[np.newaxis]))
    )
    tau_rows = 1.0 - _safe_ratio(unconditional_error_rows, conditional_errors_rows)
    v_rows = _safe_ratio(unconditional_error_rows, sample_size**2)

    ase_rows = 0.0
    if mean_rows > 0.0:
        ase_rows = _sqrt_nonnegative(
            np.sum(
                (
                    matrix
                    * (
                        -2.0 * v_rows * (row_sums[:, np.newaxis] / sample_size)
                        + mean_rows
                        * (
                            (2.0 * matrix / col_sums)
                            - np.sum((matrix / col_sums) ** 2, axis=0)
                        )
                        - ((mean_rows * (v_rows + 1.0)) - (2.0 * v_rows))
                    )
                    ** 2
                )
                / ((sample_size**2) * (mean_rows**4))
            )
        )

    unconditional_error_columns = (sample_size**2) - (
        sample_size * float(np.sum((matrix**2) / row_sums[:, np.newaxis]))
    )
    tau_columns = 1.0 - _safe_ratio(unconditional_error_columns, conditional_errors_columns)
    v_columns = _safe_ratio(unconditional_error_columns, sample_size**2)

    ase_columns_sum = 0.0
    if mean_columns > 0.0:
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                term = (
                    matrix[row_idx, col_idx]
                    * (
                        -2.0 * v_columns * (col_sums[col_idx] / sample_size)
                        + mean_columns
                        * (
                            (2.0 * matrix[row_idx, col_idx] / row_sums[row_idx])
                            - np.sum((matrix[row_idx, :] / row_sums[row_idx]) ** 2)
                        )
                        - ((mean_columns * (v_columns + 1.0)) - (2.0 * v_columns))
                    )
                    ** 2
                    / ((sample_size**2) * (mean_columns**4))
                )
                ase_columns_sum += float(term)
    ase_columns = _sqrt_nonnegative(ase_columns_sum)

    alpha = _safe_ratio(
        (sample_size**2) - float(np.sum(row_sums**2)),
        (2.0 * (sample_size**2)) - float(np.sum(row_sums**2)) - float(np.sum(col_sums**2)),
    )
    tau_symmetric = (tau_rows * alpha) + ((1.0 - alpha) * tau_columns)
    ase_symmetric = (ase_rows * alpha) + ((1.0 - alpha) * ase_columns)

    return (
        _association_from_standard_error(tau_rows, ase_rows, confidence_level),
        _association_from_standard_error(tau_columns, ase_columns, confidence_level),
        _association_from_standard_error(tau_symmetric, ase_symmetric, confidence_level),
    )


def _columns_to_contingency(
    x: Sequence[object],
    y: Sequence[object],
) -> np.ndarray:
    """Convert two categorical columns into a contingency table."""
    validate_non_empty(x, name="x")
    validate_non_empty(y, name="y")
    validate_groups_equal_length(x, y, name1="x", name2="y")

    x_array = np.asarray(list(x), dtype=object)
    y_array = np.asarray(list(y), dtype=object)

    mask = np.array([_is_present(xi) and _is_present(yi) for xi, yi in zip(x_array, y_array)])
    filtered_x = x_array[mask]
    filtered_y = y_array[mask]
    validate_non_empty(filtered_x, name="filtered_x")
    validate_non_empty(filtered_y, name="filtered_y")
    validate_groups_equal_length(filtered_x, filtered_y, name1="filtered_x", name2="filtered_y")

    x_categories, x_numeric = np.unique(filtered_x, return_inverse=True)
    y_categories, y_numeric = np.unique(filtered_y, return_inverse=True)
    table = np.zeros((len(x_categories), len(y_categories)), dtype=int)
    for row_idx, col_idx in zip(x_numeric, y_numeric):
        table[row_idx, col_idx] += 1

    validate_contingency_table(table, name="table")
    return table.astype(float)


def _validated_contingency(table: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    """Validate and normalize a contingency table."""
    validate_contingency_table(table, name="table")
    matrix = np.asarray(table, dtype=float)
    matrix = matrix[np.sum(matrix, axis=1) > 0.0, :]
    matrix = matrix[:, np.sum(matrix, axis=0) > 0.0]
    validate_contingency_table(matrix, name="table")
    total = float(matrix.sum())
    if total <= 0.0:
        raise InvalidInputError("'table' must contain a positive total count.")
    validate_sample_size(int(total), name="table_total")
    return matrix


def _association_from_candidates(
    value: float,
    se_candidates: Sequence[float],
    confidence_level: float,
) -> AssociationEstimate:
    """Create a deterministic association estimate from multiple SE candidates."""
    se = max((float(candidate) for candidate in se_candidates), default=0.0)
    return _association_from_standard_error(value, se, confidence_level)


def _association_from_standard_error(
    value: float,
    standard_error: float,
    confidence_level: float,
) -> AssociationEstimate:
    """Create an association estimate from a single standard error."""
    value_clipped = min(max(float(value), 0.0), 1.0)
    se = max(float(standard_error), 0.0)
    z_critical = float(norm.ppf(1.0 - ((1.0 - confidence_level) / 2.0)))
    ci = ConfidenceInterval(
        lower=max(0.0, value_clipped - (z_critical * se)),
        upper=min(1.0, value_clipped + (z_critical * se)),
    )
    validate_proportion(ci.lower, name="lower")
    validate_proportion(ci.upper, name="upper")

    if se == 0.0:
        statistic = math.inf if value_clipped != 0.0 else 0.0
        p_value = 0.0 if value_clipped != 0.0 else 1.0
    else:
        statistic = value_clipped / se
        p_value = float(norm.sf(abs(statistic)) * 2.0)

    return AssociationEstimate(
        value=value_clipped,
        standard_error=se,
        statistic=statistic,
        p_value=p_value,
        confidence_interval=ci,
    )


def _difference(first: AssociationEstimate, second: AssociationEstimate) -> DifferenceEstimate:
    """Compute a z-test for the difference between two association estimates."""
    difference = first.value - second.value
    standard_error = math.sqrt((first.standard_error**2) + (second.standard_error**2))
    if standard_error == 0.0:
        z_statistic = math.inf if difference != 0.0 else 0.0
        p_value = 0.0 if difference != 0.0 else 1.0
    else:
        z_statistic = difference / standard_error
        p_value = float(norm.sf(abs(z_statistic)) * 2.0)

    return DifferenceEstimate(
        difference=float(difference),
        standard_error=float(standard_error),
        z_statistic=float(z_statistic),
        p_value=float(p_value),
    )


def _table_to_tuple(matrix: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Convert a contingency table to an immutable integer tuple representation."""
    return tuple(tuple(int(value) for value in row) for row in matrix.astype(int))


def _is_present(value: object) -> bool:
    """Return whether a categorical cell should be included."""
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a safe ratio, defaulting to 0 when the denominator is zero."""
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def _sqrt_nonnegative(value: float) -> float:
    """Safely take the square root of a numerically non-negative quantity."""
    numeric_value = float(value)
    if numeric_value < 0.0:
        if numeric_value > -1e-12:
            numeric_value = 0.0
        else:
            raise StatisticalComputationError(
                f"Encountered a negative variance term: {numeric_value}."
            )
    return math.sqrt(numeric_value)
