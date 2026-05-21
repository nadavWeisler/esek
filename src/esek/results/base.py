"""Base result objects for ESEK.

All public-facing result objects are immutable frozen dataclasses.
This guarantees that results cannot be accidentally mutated after
construction and enables safe hashing/caching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EffectSizeResult:
    """Generic effect-size result returned by all ESEK calculators.

    Attributes
    ----------
    value:
        The computed effect-size point estimate.
    effect_size_type:
        Human-readable name, e.g. ``"Cohen's d"``.
    method:
        The calculation method/variant used, e.g. ``"from_parameters"``.
    ci_low:
        Lower bound of the confidence interval (``None`` if not computed).
    ci_high:
        Upper bound of the confidence interval (``None`` if not computed).
    standard_error:
        Standard error of the effect-size estimate (``None`` if not computed).
    p_value:
        Two-tailed p-value for the underlying test (``None`` if not computed).
    n:
        Total sample size (``None`` if not applicable).
    confidence_level:
        Confidence level used, e.g. ``0.95`` (``None`` if not computed).
    metadata:
        Dictionary of any extra quantities returned by the specific method
        (e.g. pivotal CI, non-central CI, corrected values).
    """

    value: float
    effect_size_type: str
    method: str
    ci_low: float | None = None
    ci_high: float | None = None
    standard_error: float | None = None
    p_value: float | None = None
    n: int | None = None
    confidence_level: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ci(self) -> tuple[float, float] | None:
        """Return ``(ci_low, ci_high)`` or ``None`` if either bound is absent."""
        if self.ci_low is None or self.ci_high is None:
            return None
        return (self.ci_low, self.ci_high)

    def __str__(self) -> str:
        parts = [f"{self.effect_size_type} = {self.value:.4f}"]
        if self.ci_low is not None and self.ci_high is not None:
            parts.append(f"95% CI [{self.ci_low:.4f}, {self.ci_high:.4f}]")
        if self.standard_error is not None:
            parts.append(f"SE = {self.standard_error:.4f}")
        if self.p_value is not None:
            parts.append(f"p = {self.p_value:.4f}")
        return ", ".join(parts)


@dataclass(frozen=True)
class ConversionResult:
    """Result of converting one effect-size metric to another.

    Attributes
    ----------
    input_type:
        Name of the source metric, e.g. ``"d"``.
    output_type:
        Name of the target metric, e.g. ``"r"``.
    input_value:
        The original effect-size value.
    output_value:
        The converted effect-size value.
    method:
        Description of the conversion formula used.
    metadata:
        Any additional quantities produced during conversion.
    """

    input_type: str
    output_type: str
    input_value: float
    output_value: float
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.input_type} = {self.input_value:.4f} → "
            f"{self.output_type} = {self.output_value:.4f} "
            f"(method: {self.method})"
        )


@dataclass(frozen=True)
class ConfidenceIntervalResult:
    """A standalone confidence interval result.

    Attributes
    ----------
    lower:
        Lower bound of the interval.
    upper:
        Upper bound of the interval.
    confidence_level:
        Nominal confidence level, e.g. ``0.95``.
    method:
        Name of the CI method used (e.g. ``"Wilson"``).
    parameter:
        Name of the quantity being estimated (e.g. ``"proportion"``).
    """

    lower: float
    upper: float
    confidence_level: float
    method: str
    parameter: str = "effect_size"

    @property
    def ci(self) -> tuple[float, float]:
        """Return ``(lower, upper)``."""
        return (self.lower, self.upper)

    def __str__(self) -> str:
        pct = int(self.confidence_level * 100)
        return (
            f"{pct}% CI for {self.parameter} [{self.lower:.4f}, {self.upper:.4f}] "
            f"(method: {self.method})"
        )
