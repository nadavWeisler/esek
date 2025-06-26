"""
This module provides utility functions for the Calculator package in the stats project.
"""

from dataclasses import asdict, is_dataclass
from typing import Any


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
