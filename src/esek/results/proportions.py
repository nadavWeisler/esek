"""Result objects for proportion-based effect sizes."""

from __future__ import annotations

from dataclasses import dataclass

from .base import EffectSizeResult


@dataclass(frozen=True)
class CohenHResult(EffectSizeResult):
    """Cohen's h — effect size for two proportions.

    Formula: h = 2·arcsin(√p₁) − 2·arcsin(√p₂)
    """

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Cohen's h")


@dataclass(frozen=True)
class CohenGResult(EffectSizeResult):
    """Cohen's g — deviation of a proportion from 0.5.

    Formula: g = p − 0.5
    """

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Cohen's g")


@dataclass(frozen=True)
class ProportionTestResult(EffectSizeResult):
    """Generic result for a proportion-based test and its effect size."""

    sample_proportion: float | None = None
    population_proportion: float | None = None

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Proportion Effect Size")
