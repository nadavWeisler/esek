"""Proportions calculators."""

from .one_sample_proportion import OneSampleProportionResults
from .two_independent_proportions import TwoIndependentProportionsResults
from .two_dependent_proportions import TwoDependentProportionsResults, TwoDependentProportions
from .multiple_proportions import MultipleProportions, CochranQResults, GoodnessOfFitResults

__all__ = [
    "OneSampleProportionResults",
    "TwoIndependentProportionsResults",
    "TwoDependentProportionsResults",
    "TwoDependentProportions",
    "MultipleProportions",
    "CochranQResults",
    "GoodnessOfFitResults",
]
