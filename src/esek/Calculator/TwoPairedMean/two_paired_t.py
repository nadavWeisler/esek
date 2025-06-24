"""
TODO: add docstring
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.stats import norm
from ...interfaces import AbstractTest
from ...results import CohenD
from ...utility import central_ci_from_cohens_d


@dataclass
class TwoPairedTResults:
    """
    TODO: add docstring
    """

    pass


class TwoPairedTTests(AbstractTest):
    """
    TODO: add docstring
    """

    @staticmethod
    def from_z_score() -> TwoPairedTResults:
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
