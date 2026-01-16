"""Esek Library"""

__title__ = "esek"
__version__ = "0.1.0"
__author__ = "Esek"
__license__ = "MIT"

from .src.esek import Calculator
from .src.esek import utils

__all__ = [
    "Calculator",
    "utils",
]
