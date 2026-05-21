"""Core module: exceptions, validation, and type aliases for ESEK."""

from .exceptions import EsekError, InvalidInputError, StatisticalComputationError
from .validation import (
    validate_sample_size,
    validate_confidence_level,
    validate_standard_deviation,
    validate_not_nan,
    validate_proportion,
    validate_groups_equal_length,
    validate_positive,
)

__all__ = [
    "EsekError",
    "InvalidInputError",
    "StatisticalComputationError",
    "validate_sample_size",
    "validate_confidence_level",
    "validate_standard_deviation",
    "validate_not_nan",
    "validate_proportion",
    "validate_groups_equal_length",
    "validate_positive",
]
