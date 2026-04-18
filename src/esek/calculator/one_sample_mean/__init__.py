"""Initialization module for One Sample Mean statistical tests."""

from .one_sample_t import OneSampleTResults, OneSampleTTest
from .one_sample_z import OneSampleZResults, OneSampleZTests

__all__ = [
    "OneSampleTResults",
    "OneSampleTTest",
    "OneSampleZResults",
    "OneSampleZTests",
]
