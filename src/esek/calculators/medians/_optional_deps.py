"""Optional / pure-Python helpers for robust median methods."""

from __future__ import annotations

from typing import Any

import numpy as np


def biweight_midvariance(
    data: Any,
    *,
    c: float = 9.0,
    modify_sample_size: bool = False,
) -> float:
    """Biweight midvariance (Beers, Flynn & Gebhardt 1990).

    Pure-NumPy implementation equivalent to ``astropy.stats.biweight_midvariance``
    for the common defaults used by the median calculators.
    """
    arr = np.asarray(data, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return float("nan")

    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad == 0.0:
        # Fall back to mean absolute deviation from the median.
        mad = float(np.mean(np.abs(arr - median)))
    if mad == 0.0:
        return 0.0

    u = (arr - median) / (c * mad)
    mask = np.abs(u) < 1.0
    u = u[mask]
    if u.size == 0:
        return float("nan")

    score = arr[mask] - median
    numerator = float(np.sum((score**2) * (1.0 - u**2) ** 4))
    denominator = float(np.sum((1.0 - u**2) * (1.0 - 5.0 * u**2)))
    if denominator == 0.0:
        return float("nan")

    n_eff = float(n) if not modify_sample_size else float(np.sum(mask))
    return n_eff * numerator / (denominator**2)


def independent_samples_bootstrap(*args: Any, **kwargs: Any) -> Any:
    """Construct an ``arch`` bootstrap object (imported lazily)."""
    from arch.bootstrap import IndependentSamplesBootstrap  # type: ignore[import]

    return IndependentSamplesBootstrap(*args, **kwargs)
