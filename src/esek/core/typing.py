"""Type aliases for ESEK.

Centralises frequently used type constructs so they can be imported
from a single location across the library.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import NDArray

# Anything that can be treated as a 1-D numeric array
ArrayLike = Union[list[float], tuple[float, ...], NDArray[np.floating]]

# A (lower, upper) confidence-interval pair
CITuple = tuple[float, float]

# A numeric scalar (int or float, but not bool)
Numeric = Union[int, float]
