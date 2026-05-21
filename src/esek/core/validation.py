"""Input validation helpers for the ESEK library.

All validators raise ``InvalidInputError`` on invalid input and return
``None`` on success (so they can be chained like assertions).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .exceptions import InvalidInputError


def validate_sample_size(n: Any, name: str = "n") -> None:
    """Raise ``InvalidInputError`` if *n* is not a positive integer.

    Parameters
    ----------
    n:
        The sample size to validate.
    name:
        Variable name used in the error message.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
        raise InvalidInputError(
            f"'{name}' must be an integer, got {type(n).__name__}."
        )
    if n < 1:
        raise InvalidInputError(f"'{name}' must be >= 1, got {n}.")


def validate_confidence_level(confidence_level: Any, name: str = "confidence_level") -> None:
    """Raise ``InvalidInputError`` if *confidence_level* is not in (0, 1).

    Parameters
    ----------
    confidence_level:
        The confidence level (e.g. 0.95 for 95 %).
    name:
        Variable name used in the error message.
    """
    try:
        cl = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(
            f"'{name}' must be a float in (0, 1), got {confidence_level!r}."
        ) from exc
    if not (0.0 < cl < 1.0):
        raise InvalidInputError(
            f"'{name}' must be strictly between 0 and 1, got {cl}."
        )


def validate_standard_deviation(sd: Any, name: str = "standard_deviation") -> None:
    """Raise ``InvalidInputError`` if *sd* is not a positive finite number.

    Parameters
    ----------
    sd:
        The standard deviation to validate.
    name:
        Variable name used in the error message.
    """
    try:
        sd_f = float(sd)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(
            f"'{name}' must be a positive number, got {sd!r}."
        ) from exc
    if not math.isfinite(sd_f):
        raise InvalidInputError(f"'{name}' must be finite, got {sd_f}.")
    if sd_f <= 0.0:
        raise InvalidInputError(f"'{name}' must be > 0, got {sd_f}.")


def validate_not_nan(value: Any, name: str = "value") -> None:
    """Raise ``InvalidInputError`` if *value* is NaN.

    Parameters
    ----------
    value:
        The value to check.
    name:
        Variable name used in the error message.
    """
    try:
        if math.isnan(float(value)):
            raise InvalidInputError(f"'{name}' must not be NaN.")
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(
            f"'{name}' must be numeric, got {type(value).__name__}."
        ) from exc


def validate_proportion(p: Any, name: str = "proportion") -> None:
    """Raise ``InvalidInputError`` if *p* is not in [0, 1].

    Parameters
    ----------
    p:
        The proportion value to validate.
    name:
        Variable name used in the error message.
    """
    try:
        p_f = float(p)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(
            f"'{name}' must be a float in [0, 1], got {p!r}."
        ) from exc
    if not (0.0 <= p_f <= 1.0):
        raise InvalidInputError(
            f"'{name}' must be in [0, 1], got {p_f}."
        )


def validate_groups_equal_length(
    group1: Sequence[Any],
    group2: Sequence[Any],
    name1: str = "group1",
    name2: str = "group2",
) -> None:
    """Raise ``InvalidInputError`` if *group1* and *group2* differ in length.

    Parameters
    ----------
    group1, group2:
        The two sequences to compare.
    name1, name2:
        Variable names used in the error message.
    """
    if len(group1) != len(group2):
        raise InvalidInputError(
            f"'{name1}' (length {len(group1)}) and '{name2}' (length {len(group2)}) "
            "must have the same number of observations for a paired design."
        )


def validate_positive(value: Any, name: str = "value") -> None:
    """Raise ``InvalidInputError`` if *value* is not strictly positive.

    Parameters
    ----------
    value:
        The numeric value to validate.
    name:
        Variable name used in the error message.
    """
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(
            f"'{name}' must be a positive number, got {value!r}."
        ) from exc
    if not math.isfinite(v) or v <= 0.0:
        raise InvalidInputError(f"'{name}' must be > 0, got {v}.")


def validate_non_empty(data: Sequence[Any], name: str = "data") -> None:
    """Raise ``InvalidInputError`` if *data* is empty.

    Parameters
    ----------
    data:
        The sequence to check.
    name:
        Variable name used in the error message.
    """
    if len(data) == 0:
        raise InvalidInputError(f"'{name}' must not be empty.")


def validate_contingency_table(table: Any, name: str = "table") -> None:
    """Raise ``InvalidInputError`` if *table* is not a valid 2-D contingency table.

    A valid table must be array-like with at least 2 rows and 2 columns,
    all non-negative integer values.

    Parameters
    ----------
    table:
        The contingency table to validate.
    name:
        Variable name used in the error message.
    """
    try:
        arr = np.asarray(table, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(
            f"'{name}' must be array-like with numeric values."
        ) from exc
    if arr.ndim != 2:
        raise InvalidInputError(
            f"'{name}' must be 2-dimensional, got {arr.ndim} dimensions."
        )
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise InvalidInputError(
            f"'{name}' must have at least 2 rows and 2 columns, got shape {arr.shape}."
        )
    if np.any(arr < 0):
        raise InvalidInputError(
            f"'{name}' must contain only non-negative values."
        )
