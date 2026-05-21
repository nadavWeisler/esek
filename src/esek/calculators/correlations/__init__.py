"""Correlations calculators — association measures for all measurement levels."""

from .nominal_by_nominal import NominalByNominal
from .nominal_by_interval import NominalByInterval
from .nominal_by_ordinal import NominalByOrdinal
from .ordinal_by_interval import OrdinalByInterval
from .ordinal_by_ordinal import OrdinalByOrdinal
from .ordinal_partial_correlation import OrdinalPartialCorrelation
from .interval_by_interval import PearsonCorrelation
from .multiple_r_squared import MultipleRSquared, MultipleRSquaredResult, compute_adjusted_r_squared
from .partial_pearson import PartialPearsonCorrelation, PartialCorrelationResult
from .correlation_differences import PearsonCorrelationDifference, CorrelationDifferenceResult
from .categorical_differences import CategoricalAssociationDifference, CategoricalDifferenceResult

__all__ = [
    "NominalByNominal",
    "NominalByInterval",
    "NominalByOrdinal",
    "OrdinalByInterval",
    "OrdinalByOrdinal",
    "OrdinalPartialCorrelation",
    "PearsonCorrelation",
    "MultipleRSquared",
    "MultipleRSquaredResult",
    "compute_adjusted_r_squared",
    "PartialPearsonCorrelation",
    "PartialCorrelationResult",
    "PearsonCorrelationDifference",
    "CorrelationDifferenceResult",
    "CategoricalAssociationDifference",
    "CategoricalDifferenceResult",
]
