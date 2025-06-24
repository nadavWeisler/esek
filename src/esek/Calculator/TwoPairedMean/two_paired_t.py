"""
TODO: add docstring
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.stats import norm
from ...interfaces import AbstractTest
from ...results import CohenD, HedgesG, CohensDav, HedgesGav, CohensDrm, HedgesGrm
from ...utility import central_ci_from_cohens_d


@dataclass
class TwoPairedTResults:
    """
    TODO: add docstring
    """

    cohens_d: Optional[CohenD] = None
    hedge_g: Optional[HedgesG] = None
    cohens_dav: Optional[CohensDav] = None
    hedge_gav: Optional[HedgesGav] = None
    cohens_drm: Optional[CohensDrm] = None
    hedge_grm: Optional[HedgesGrm] = None
    t_score: Optional[float] = None
    p_value: Optional[float] = None
    degrees_of_freedom: Optional[float] = None
    sample_mean_1: Optional[float] = None
    sample_mean_2: Optional[float] = None
    sample_sd_1: Optional[int] = None
    sample_sd_2: Optional[int] = None
    sample_size_1: Optional[int] = None
    sample_size_2: Optional[int] = None
    mean_difference: Optional[float] = None
    difference_sd: Optional[float] = None
    standard_error: Optional[float] = None
    sample_mean_1: Optional[float] = None
    sample_mean_2: Optional[float] = None 


class TwoPairedTTests(AbstractTest):
    """
    TODO: add docstring
    """

    @staticmethod
    def from_score() -> TwoPairedTResults:
        """
        TODO: add docstring
        """
        pass

    @staticmethod
    def from_parameters() -> TwoPairedTResults:
        """
        TODO: add docstring
        """
        pass

    @staticmethod
    def from_data() -> TwoPairedTResults:
        """
        TODO: add docstring
        """
        pass
