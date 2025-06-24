"""
This module provides utility functions for the Calculator package in the stats project.
"""

from dataclasses import asdict, is_dataclass
from typing import Any
import numpy as np
from scipy.stats import norm

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


def central_ci_from_cohens_d(
    cohens_d: float, sample_size: float, confidence_level: float
) -> tuple[float, float, float]:
    """
    Calculate the confidence intervals and standard error for Cohen's d effect size in a
    one-sample Z-test.

    This function calculates the confidence intervals of the effect size (Cohen's d) for a
    one-sample Z-test or two dependent samples test using the Hedges and Olkin (1985)
    formula to estimate the standard error.

    Parameters
    ----------
    cohens_d : float
        The calculated Cohen's d effect size
    sample_size : float
        The size of the sample
    confidence_level : float
        The confidence level as a decimal (e.g., 0.95 for 95%)

    Returns
    -------
    tuple
        A tuple containing:
        - ci_lower (float): Lower bound of the confidence interval
        - ci_upper (float): Upper bound of the confidence interval
        - standard_error_es (float): Standard error of the effect size

    Notes
    -----
    Since the effect size in the population and its standard deviation are unknown,
    we estimate it based on the sample using the Hedges and Olkin (1985) formula
    to estimate the standard deviation of the effect size.
    """
    Standard_error_es = np.sqrt((1 / sample_size) + ((cohens_d**2 / (2 * sample_size))))
    z_critical_value = norm.ppf(confidence_level + ((1 - confidence_level) / 2))
    ci_lower, ci_upper = (
        cohens_d - Standard_error_es * z_critical_value,
        cohens_d + Standard_error_es * z_critical_value,
    )
    return ci_lower, ci_upper, Standard_error_es
