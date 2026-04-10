"""
Mathematical and statistical utility functions for the ESEK package.

This module provides general-purpose math and statistics helpers, including
adaptive numerical integration, Winsorized statistics, and dataclass utilities.
"""

from dataclasses import asdict, is_dataclass
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from .interfaces import MethodType


def not_implemented(method_type: MethodType, stats_test_type: str):
    """
    Decorator to mark a class method as not implemented.

    Args:
        method_type (MethodType): Type of the method (e.g., 'from_score', 'from_parameters', 'from_data').
        stats_test_type (str): The statistical test type.

    Returns:
        function: A decorated function that always raises NotImplementedError.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            raise NotImplementedError(
                f"{method_type} method is not implemented for {stats_test_type}"
            )

        return wrapper

    return decorator


def convert_results_to_dict(dataclass_instance: Any) -> dict:
    """
    Converts a dataclass instance to a dictionary.

    Args:
        dataclass_instance (dataclass): An instance of a dataclass.

    Returns:
        dict: A dictionary representation of the dataclass instance.
    """
    if not (
        is_dataclass(dataclass_instance) and not isinstance(dataclass_instance, type)
    ):
        raise TypeError(
            f"Expected a dataclass instance, got: {type(dataclass_instance)}"
        )

    return asdict(dataclass_instance)


def density(x: float) -> float:
    """
    Density function for the normal distribution.

    Args:
        x (float): The input value.

    Returns:
        float: The density of the normal distribution at x.
    """

    return float(np.array(x) ** 2 * stats.norm.pdf(np.array(x)))


def area_under_function(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    limit: int = 10,
    eps: float = 1e-5,
) -> float:
    """Recursively compute the area under a function using adaptive Simpson's rule."""

    def simpson_recursive(
        f: Callable[[float], float],
        a: float,
        b: float,
        fa: float,
        fb: float,
        fm: float,
        depth: int,
    ) -> float:

        mid = (a + b) / 2
        h = b - a
        whole = (fa + 4 * fm + fb) * h / 6
        lm = (a + mid) / 2
        rm = (mid + b) / 2
        flm = f(lm)
        frm = f(rm)
        left = (fa + 4 * flm + fm) * (h / 2) / 6
        right = (fm + 4 * frm + fb) * (h / 2) / 6

        if abs(left + right - whole) < eps or depth == 0:
            return left + right

        return simpson_recursive(f, a, mid, fa, fm, flm, depth - 1) + simpson_recursive(
            f, mid, b, fm, fb, frm, depth - 1
        )

    fa = f(a)
    fb = f(b)
    mid = (a + b) / 2
    fm = f(mid)

    return simpson_recursive(f, a, b, fa, fb, fm, limit)


def winsorized_variance(x: list[float] | NDArray, trimming_level=0.2) -> float:
    """
    Compute the Winsorized variance of a sample.

    Parameters
    ----------
    x : list[float] or NDArray
        The input sample data.
    trimming_level : float, optional
        The proportion to Winsorize from each tail (default 0.2).

    Returns
    -------
    float
        The Winsorized variance of the sample.
    """
    y = np.sort(x)
    n = len(x)
    ibot = int(np.floor(trimming_level * n)) + 1
    itop = n - ibot + 1
    xbot = y[ibot - 1]
    xtop = y[itop - 1]
    y = np.where(y <= xbot, xbot, y)
    y = np.where(y >= xtop, xtop, y)
    winvar = np.std(y, ddof=1) ** 2
    return float(winvar)


def winsorized_correlation(x: list[float], y: list[float], trimming_level=0.2) -> dict:
    """
    Compute the Winsorized correlation between two samples.

    Parameters
    ----------
    x : list[float]
        The first sample data.
    y : list[float]
        The second sample data.
    trimming_level : float, optional
        The proportion to Winsorize from each tail (default 0.2).

    Returns
    -------
    dict
        A dictionary containing:
        - ``cor``: Winsorized correlation coefficient.
        - ``cov``: Winsorized covariance.
        - ``p.value``: Two-tailed p-value for the correlation.
        - ``n``: Sample size.
        - ``test_statistic``: The test statistic.
    """
    sample_size = len(x)
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    trimming_size = int(np.floor(trimming_level * sample_size)) + 1
    x_lower = x_sorted[trimming_size - 1]
    x_upper = x_sorted[sample_size - trimming_size]
    y_lower = y_sorted[trimming_size - 1]
    y_upper = y_sorted[sample_size - trimming_size]
    x_winsorized = np.clip(x, x_lower, x_upper)
    y_winsorized = np.clip(y, y_lower, y_upper)
    winsorized_correlation_result = np.corrcoef(x_winsorized, y_winsorized)[0, 1]
    winsorized_covariance = np.cov(x_winsorized, y_winsorized)[0, 1]
    test_statistic = winsorized_correlation_result * np.sqrt(
        (sample_size - 2) / (1 - winsorized_correlation_result**2)
    )
    number_of_trimmed_values = int(np.floor(trimming_level * sample_size))
    p_value = 2 * (
        1
        - stats.t.cdf(
            np.abs(test_statistic), sample_size - 2 * number_of_trimmed_values - 2
        )
    )
    return {
        "cor": winsorized_correlation_result,
        "cov": winsorized_covariance,
        "p.value": p_value,
        "n": sample_size,
        "test_statistic": test_statistic,
    }
