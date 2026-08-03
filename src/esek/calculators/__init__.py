"""Calculators package — all statistical calculation methods.

Subpackages are imported lazily where possible so that importing a single
calculator (e.g. agreements) does not pull optional heavy deps such as
astropy/arch used only by median methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .contingency_tables import ContingencyTable2x2
from .stratified_contingency import StratifiedTwoByTwo, StratifiedTwoByTwoResult

if TYPE_CHECKING:
    from . import (
        agreements,
        anova,
        correlations,
        medians,
        one_sample_mean,
        proportions,
        two_independent_mean,
        two_paired_mean,
    )

__all__ = [
    "one_sample_mean",
    "two_paired_mean",
    "two_independent_mean",
    "proportions",
    "correlations",
    "medians",
    "agreements",
    "anova",
    "ContingencyTable2x2",
    "StratifiedTwoByTwo",
    "StratifiedTwoByTwoResult",
]


def __getattr__(name: str):
    if name in {
        "agreements",
        "anova",
        "correlations",
        "medians",
        "one_sample_mean",
        "proportions",
        "two_independent_mean",
        "two_paired_mean",
    }:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
