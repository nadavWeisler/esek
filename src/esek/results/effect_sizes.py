"""Typed result objects for effect-size families.

Each class is a frozen dataclass that inherits from ``EffectSizeResult``.
Subclassing is intentionally shallow — we only create a subclass when
it adds new fields or when the distinction matters for type checking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import EffectSizeResult


# ---------------------------------------------------------------------------
# Cohen's d family (standardised mean differences)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CohensDResult(EffectSizeResult):
    """Cohen's d — standardised mean difference, pooled SD denominator."""

    def __post_init__(self) -> None:
        # Enforce effect_size_type when not explicitly supplied
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Cohen's d")


@dataclass(frozen=True)
class HedgesGResult(EffectSizeResult):
    """Hedges' g — bias-corrected Cohen's d."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Hedges' g")


@dataclass(frozen=True)
class CohensDavResult(EffectSizeResult):
    """Cohen's d_av — paired design, average SD denominator."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Cohen's d_av")


@dataclass(frozen=True)
class HedgesGavResult(EffectSizeResult):
    """Hedges' g_av — bias-corrected Cohen's d_av."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Hedges' g_av")


@dataclass(frozen=True)
class CohensDrmResult(EffectSizeResult):
    """Cohen's d_rm — paired design, repeated-measures SD denominator."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Cohen's d_rm")


@dataclass(frozen=True)
class HedgesGrmResult(EffectSizeResult):
    """Hedges' g_rm — bias-corrected Cohen's d_rm."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Hedges' g_rm")


@dataclass(frozen=True)
class GlassDeltaResult(EffectSizeResult):
    """Glass's Δ — standardised mean difference, control-group SD denominator."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Glass's Δ")


# ---------------------------------------------------------------------------
# Correlation-based effect sizes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BiserialResult(EffectSizeResult):
    """Biserial or point-biserial correlation effect size."""

    z_score: float | None = None

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Biserial r")


# ---------------------------------------------------------------------------
# Non-parametric / distribution-free effect sizes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CLESResult(EffectSizeResult):
    """Common Language Effect Size (CLES) / probability of superiority."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "CLES")


@dataclass(frozen=True)
class ProbabilityOfSuperiorityResult(EffectSizeResult):
    """Probability of superiority (PS) — non-parametric."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Probability of Superiority")


@dataclass(frozen=True)
class VarghaDelaneyResult(EffectSizeResult):
    """Vargha & Delaney's A — non-parametric stochastic superiority."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Vargha-Delaney A")


@dataclass(frozen=True)
class CliffsDeltaResult(EffectSizeResult):
    """Cliff's delta — non-parametric rank-based effect size."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Cliff's delta")


@dataclass(frozen=True)
class NonParametricU1Result(EffectSizeResult):
    """Cohen's U1 — proportion of non-overlap (upper tail)."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Cohen's U1")


@dataclass(frozen=True)
class NonParametricU3Result(EffectSizeResult):
    """Cohen's U3 — proportion of distribution B above median of A."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Cohen's U3")


@dataclass(frozen=True)
class KraemerAndrewGammaResult(EffectSizeResult):
    """Kraemer & Andrews' Gamma — non-parametric effect size."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Kraemer-Andrews Gamma")


@dataclass(frozen=True)
class WilcoxMusakaQResult(EffectSizeResult):
    """Wilcox & Musaka's Q — non-parametric effect size."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Wilcox-Musaka Q")


# ---------------------------------------------------------------------------
# Other
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RatioOfMeansResult(EffectSizeResult):
    """Ratio of means (ROM) effect size."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Ratio of Means")


@dataclass(frozen=True)
class RobustAKPResult(EffectSizeResult):
    """Robust AKP effect size (Algina, Keselman & Penfield)."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Robust AKP")


@dataclass(frozen=True)
class RobustExplanatoryResult(EffectSizeResult):
    """Robust Explanatory Measure of Effect Size (Wilcox & Tian)."""

    def __post_init__(self) -> None:
        if not self.effect_size_type:
            object.__setattr__(self, "effect_size_type", "Robust Explanatory")
