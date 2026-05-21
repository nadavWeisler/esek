"""Results package: typed, frozen result objects for all ESEK computations."""

from .base import EffectSizeResult, ConversionResult, ConfidenceIntervalResult
from .effect_sizes import (
    CohensDResult,
    HedgesGResult,
    CohensDavResult,
    HedgesGavResult,
    CohensDrmResult,
    HedgesGrmResult,
    GlassDeltaResult,
    BiserialResult,
    RatioOfMeansResult,
    RobustAKPResult,
    RobustExplanatoryResult,
    CLESResult,
    ProbabilityOfSuperiorityResult,
    VarghaDelaneyResult,
    CliffsDeltaResult,
    NonParametricU1Result,
    NonParametricU3Result,
    KraemerAndrewGammaResult,
    WilcoxMusakaQResult,
)
from .proportions import (
    CohenHResult,
    CohenGResult,
    ProportionTestResult,
)
from .descriptive import (
    ConfidenceInterval,
    DescriptiveStatistics,
    SampleStatistics,
    GroupStatistics,
    InferentialStatistics,
    ApproximatedStandardError,
)

__all__ = [
    # Base
    "EffectSizeResult",
    "ConversionResult",
    "ConfidenceIntervalResult",
    # Effect sizes
    "CohensDResult",
    "HedgesGResult",
    "CohensDavResult",
    "HedgesGavResult",
    "CohensDrmResult",
    "HedgesGrmResult",
    "GlassDeltaResult",
    "BiserialResult",
    "RatioOfMeansResult",
    "RobustAKPResult",
    "RobustExplanatoryResult",
    "CLESResult",
    "ProbabilityOfSuperiorityResult",
    "VarghaDelaneyResult",
    "CliffsDeltaResult",
    "NonParametricU1Result",
    "NonParametricU3Result",
    "KraemerAndrewGammaResult",
    "WilcoxMusakaQResult",
    # Proportions
    "CohenHResult",
    "CohenGResult",
    "ProportionTestResult",
    # Descriptive
    "ConfidenceInterval",
    "DescriptiveStatistics",
    "SampleStatistics",
    "GroupStatistics",
    "InferentialStatistics",
    "ApproximatedStandardError",
]
