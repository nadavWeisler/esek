"""ESEK — Effect Size Estimation Kit.

Public API
----------

Calculators::

    from esek.calculators.two_independent_mean.two_independent_t import TwoIndependentTTests

Converters::

    from esek import EffectSizeConverter

    result = EffectSizeConverter.d_to_r(d=0.5, n1=30, n2=30)
    print(result.output_value)

Result objects::

    from esek.results import EffectSizeResult, ConversionResult

Core (validation / exceptions)::

    from esek.core import InvalidInputError, validate_sample_size
"""

from __future__ import annotations

from .converters import EffectSizeConverter, StatisticToEffectSize
from .core.exceptions import (
    EsekError,
    InvalidInputError,
    StatisticalComputationError,
)
from .calculators.contingency_tables import ContingencyTable2x2
from .calculators.correlations import PearsonCorrelation
from .confidence_intervals import CohensDCI

__version__ = "0.2.0"
__author__ = "Nadav Weisler"

__all__ = [
    # Top-level convenience
    "EffectSizeConverter",
    "StatisticToEffectSize",
    "ContingencyTable2x2",
    "PearsonCorrelation",
    "CohensDCI",
    # Exceptions
    "EsekError",
    "InvalidInputError",
    "StatisticalComputationError",
]
