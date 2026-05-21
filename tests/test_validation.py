"""Tests for the core validation module."""

import pytest
from esek.core.exceptions import EsekError, InvalidInputError, StatisticalComputationError
from esek.core.validation import (
    validate_sample_size,
    validate_confidence_level,
    validate_standard_deviation,
    validate_not_nan,
    validate_proportion,
    validate_groups_equal_length,
    validate_positive,
    validate_non_empty,
    validate_contingency_table,
)


# ---------------------------------------------------------------------------
# validate_sample_size
# ---------------------------------------------------------------------------

class TestValidateSampleSize:
    def test_valid_integer(self):
        validate_sample_size(10)  # should not raise

    def test_valid_one(self):
        validate_sample_size(1)

    def test_zero_raises(self):
        with pytest.raises(InvalidInputError, match="n.*must be >= 1"):
            validate_sample_size(0)

    def test_negative_raises(self):
        with pytest.raises(InvalidInputError):
            validate_sample_size(-5)

    def test_float_raises(self):
        with pytest.raises(InvalidInputError):
            validate_sample_size(10.5)

    def test_string_raises(self):
        with pytest.raises(InvalidInputError):
            validate_sample_size("10")

    def test_bool_raises(self):
        with pytest.raises(InvalidInputError):
            validate_sample_size(True)

    def test_custom_name_in_message(self):
        with pytest.raises(InvalidInputError, match="n1"):
            validate_sample_size(0, "n1")


# ---------------------------------------------------------------------------
# validate_confidence_level
# ---------------------------------------------------------------------------

class TestValidateConfidenceLevel:
    def test_valid_95(self):
        validate_confidence_level(0.95)

    def test_valid_99(self):
        validate_confidence_level(0.99)

    def test_zero_raises(self):
        with pytest.raises(InvalidInputError):
            validate_confidence_level(0.0)

    def test_one_raises(self):
        with pytest.raises(InvalidInputError):
            validate_confidence_level(1.0)

    def test_negative_raises(self):
        with pytest.raises(InvalidInputError):
            validate_confidence_level(-0.5)

    def test_greater_than_one_raises(self):
        with pytest.raises(InvalidInputError):
            validate_confidence_level(1.5)

    def test_string_raises(self):
        with pytest.raises(InvalidInputError):
            validate_confidence_level("high")


# ---------------------------------------------------------------------------
# validate_standard_deviation
# ---------------------------------------------------------------------------

class TestValidateStandardDeviation:
    def test_valid(self):
        validate_standard_deviation(2.5)

    def test_zero_raises(self):
        with pytest.raises(InvalidInputError):
            validate_standard_deviation(0.0)

    def test_negative_raises(self):
        with pytest.raises(InvalidInputError):
            validate_standard_deviation(-1.0)

    def test_nan_raises(self):
        import math
        with pytest.raises(InvalidInputError):
            validate_standard_deviation(math.nan)


# ---------------------------------------------------------------------------
# validate_not_nan
# ---------------------------------------------------------------------------

class TestValidateNotNan:
    def test_valid_float(self):
        validate_not_nan(3.14)

    def test_nan_raises(self):
        import math
        with pytest.raises(InvalidInputError, match="NaN"):
            validate_not_nan(math.nan)

    def test_string_raises(self):
        with pytest.raises(InvalidInputError):
            validate_not_nan("x")


# ---------------------------------------------------------------------------
# validate_proportion
# ---------------------------------------------------------------------------

class TestValidateProportion:
    def test_valid_zero(self):
        validate_proportion(0.0)

    def test_valid_one(self):
        validate_proportion(1.0)

    def test_valid_half(self):
        validate_proportion(0.5)

    def test_negative_raises(self):
        with pytest.raises(InvalidInputError):
            validate_proportion(-0.1)

    def test_greater_than_one_raises(self):
        with pytest.raises(InvalidInputError):
            validate_proportion(1.01)


# ---------------------------------------------------------------------------
# validate_groups_equal_length
# ---------------------------------------------------------------------------

class TestValidateGroupsEqualLength:
    def test_equal_lengths(self):
        validate_groups_equal_length([1, 2, 3], [4, 5, 6])

    def test_unequal_raises(self):
        with pytest.raises(InvalidInputError, match="same number"):
            validate_groups_equal_length([1, 2], [1, 2, 3])


# ---------------------------------------------------------------------------
# validate_contingency_table
# ---------------------------------------------------------------------------

class TestValidateContingencyTable:
    def test_valid_2x2(self):
        validate_contingency_table([[10, 5], [3, 8]])

    def test_1d_raises(self):
        with pytest.raises(InvalidInputError, match="2-dimensional"):
            validate_contingency_table([1, 2, 3])

    def test_1x2_raises(self):
        with pytest.raises(InvalidInputError, match="at least 2 rows"):
            validate_contingency_table([[1, 2]])

    def test_negative_raises(self):
        with pytest.raises(InvalidInputError, match="non-negative"):
            validate_contingency_table([[10, -1], [3, 8]])


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    def test_invalid_input_is_esek_error(self):
        assert issubclass(InvalidInputError, EsekError)

    def test_computation_error_is_esek_error(self):
        assert issubclass(StatisticalComputationError, EsekError)

    def test_esek_error_is_exception(self):
        assert issubclass(EsekError, Exception)
