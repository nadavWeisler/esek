"""Custom exceptions for the ESEK library."""


class EsekError(Exception):
    """Base exception for all ESEK errors."""


class InvalidInputError(EsekError):
    """Raised when a function receives invalid input arguments.

    Examples include negative sample sizes, invalid confidence levels,
    non-positive standard deviations, or mismatched array lengths.
    """


class StatisticalComputationError(EsekError):
    """Raised when a statistical computation fails or is undefined.

    Examples include division by zero in pooled SD (when all variances are
    zero), NCP search convergence failure, or numerically unstable results.
    """


class NotImplementedForMethodError(EsekError):
    """Raised when a method type is not implemented for a given test."""

    def __init__(self, method_type: str, test_type: str) -> None:
        super().__init__(
            f"'{method_type}' is not implemented for '{test_type}'."
        )
