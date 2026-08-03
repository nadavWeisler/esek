"""Shared weight matrices for Gwet AC and Krippendorff α."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ...core import InvalidInputError

WeightMethod = Literal[
    "unweighted",
    "linear",
    "quadratic",
    "ordinal",
    "bipolar",
    "circular",
    "radical",
    "ratio",
]


def build_weight_matrix(
    values: np.ndarray,
    method: WeightMethod,
) -> np.ndarray:
    """Build an agreement weight matrix for ordered category values."""
    values = np.asarray(values, dtype=float)
    k = values.size
    if k < 2:
        raise InvalidInputError("At least two distinct rating values are required.")
    max_v = float(np.max(values))
    min_v = float(np.min(values))
    span = max_v - min_v
    if method == "unweighted":
        return np.eye(k, dtype=float)
    if method == "linear":
        if span == 0:
            raise InvalidInputError("Linear weights require a non-zero value range.")
        return 1.0 - np.abs(np.subtract.outer(values, values)) / span
    if method == "quadratic":
        if span == 0:
            raise InvalidInputError("Quadratic weights require a non-zero value range.")
        return 1.0 - (np.subtract.outer(values, values) ** 2) / (span**2)
    if method == "ordinal":
        ordinal = (
            (np.maximum.outer(values, values) - np.minimum.outer(values, values) + 1)
            * (np.maximum.outer(values, values) - np.minimum.outer(values, values))
            / 2.0
        )
        return 1.0 - ordinal / np.max(ordinal)
    if method == "bipolar":
        neq = np.not_equal.outer(np.arange(1, k + 1), np.arange(1, k + 1))
        sq = np.subtract.outer(values, values) ** 2
        add = np.add.outer(values, values)
        raw = np.where(
            neq,
            sq / ((add - 2.0 * min_v) * (2.0 * max_v - add)),
            0.0,
        )
        return 1.0 - raw / np.max(raw)
    if method == "circular":
        raw = np.sin(np.pi * (np.subtract.outer(values, values) / (span + 1.0))) ** 2
        return 1.0 - raw / np.max(raw)
    if method == "radical":
        if span == 0:
            raise InvalidInputError("Radical weights require a non-zero value range.")
        return 1.0 - np.sqrt(np.abs(np.subtract.outer(values, values))) / np.sqrt(span)
    if method == "ratio":
        if max_v + min_v == 0:
            raise InvalidInputError("Ratio weights require a non-zero value sum.")
        raw = (np.subtract.outer(values, values) / np.add.outer(values, values)) ** 2
        scale = ((max_v - min_v) / (max_v + min_v)) ** 2
        return 1.0 - raw / scale
    raise InvalidInputError(f"Unknown weights method: {method!r}.")


def subjects_by_category_counts(
    ratings: np.ndarray,
    categories: np.ndarray,
) -> np.ndarray:
    """Convert subjects × raters ratings to subjects × category counts.

    Missing values are encoded as ``np.nan``.
    """
    n_subjects = ratings.shape[0]
    counts = np.zeros((n_subjects, categories.size), dtype=float)
    for i in range(n_subjects):
        row = ratings[i]
        valid = row[~np.isnan(row)]
        for j, value in enumerate(categories):
            counts[i, j] = np.sum(valid == value)
    return counts


def coerce_ratings_matrix(
    data: np.ndarray | list,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce a subjects × raters matrix, treating ``''`` / ``None`` as missing.

    Returns
    -------
    ratings:
        Float matrix with NaN for missing entries.
    categories:
        Sorted unique observed rating values.
    """
    arr = np.array(data, dtype=object)
    if arr.ndim != 2:
        raise InvalidInputError("'data' must be a 2-D subjects × raters matrix.")
    ratings = np.empty(arr.shape, dtype=float)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
                ratings[i, j] = np.nan
            else:
                ratings[i, j] = float(value)
    valid = ratings[~np.isnan(ratings)]
    if valid.size == 0:
        raise InvalidInputError("'data' contains no observed ratings.")
    categories = np.unique(valid)
    return ratings, categories
