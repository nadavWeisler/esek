"""Public converter API for ESEK.

Exposes all effect-size conversions via the :class:`EffectSizeConverter`
class so users have one clean entry-point:

    >>> from esek import EffectSizeConverter
    >>> EffectSizeConverter.d_to_r(d=0.5, n1=30, n2=30)
"""

from __future__ import annotations

from ..results.base import ConversionResult
from .d_conversions import (
    d_to_r,
    d_to_r_equal_n,
    d_to_odds_ratio,
    d_to_cohens_f,
    d_to_r_squared,
    d_to_eta_squared,
)
from .r_conversions import (
    r_to_d,
    r_to_d_equal_n,
    r_to_fisher_z,
    fisher_z_to_r,
)
from .odds_ratio_conversions import (
    odds_ratio_to_d,
    log_odds_ratio_to_d,
    odds_ratio_to_r,
)


class EffectSizeConverter:
    """Convenience namespace for all effect-size conversions.

    All methods are static and return a :class:`~esek.results.base.ConversionResult`.
    """

    # ------------------------------------------------------------------
    # d conversions
    # ------------------------------------------------------------------

    @staticmethod
    def d_to_r(d: float, n1: int, n2: int) -> ConversionResult:
        """Convert Cohen's *d* to Pearson *r* (unequal *n* formula)."""
        return d_to_r(d, n1, n2)

    @staticmethod
    def d_to_r_equal_n(d: float, n: int) -> ConversionResult:
        """Convert Cohen's *d* to Pearson *r* assuming equal group sizes."""
        return d_to_r_equal_n(d, n)

    @staticmethod
    def d_to_odds_ratio(d: float) -> ConversionResult:
        """Convert Cohen's *d* to odds ratio (logistic approximation)."""
        return d_to_odds_ratio(d)

    @staticmethod
    def d_to_cohens_f(d: float) -> ConversionResult:
        """Convert Cohen's *d* to Cohen's *f* (two-group ANOVA)."""
        return d_to_cohens_f(d)

    @staticmethod
    def d_to_r_squared(d: float, n1: int, n2: int) -> ConversionResult:
        """Convert Cohen's *d* to *r²*."""
        return d_to_r_squared(d, n1, n2)

    @staticmethod
    def d_to_eta_squared(d: float, n1: int, n2: int) -> ConversionResult:
        """Convert Cohen's *d* to η² (two-group equivalence)."""
        return d_to_eta_squared(d, n1, n2)

    # ------------------------------------------------------------------
    # r conversions
    # ------------------------------------------------------------------

    @staticmethod
    def r_to_d(r: float, n1: int, n2: int) -> ConversionResult:
        """Convert Pearson *r* to Cohen's *d* (unequal *n*)."""
        return r_to_d(r, n1, n2)

    @staticmethod
    def r_to_d_equal_n(r: float) -> ConversionResult:
        """Convert Pearson *r* to Cohen's *d* (equal *n* approximation)."""
        return r_to_d_equal_n(r)

    @staticmethod
    def r_to_fisher_z(r: float) -> ConversionResult:
        """Apply Fisher's *z′* transformation to Pearson *r*."""
        return r_to_fisher_z(r)

    @staticmethod
    def fisher_z_to_r(z: float) -> ConversionResult:
        """Inverse Fisher *z′* transformation back to Pearson *r*."""
        return fisher_z_to_r(z)

    # ------------------------------------------------------------------
    # Odds-ratio conversions
    # ------------------------------------------------------------------

    @staticmethod
    def odds_ratio_to_d(or_: float) -> ConversionResult:
        """Convert an odds ratio to Cohen's *d*."""
        return odds_ratio_to_d(or_)

    @staticmethod
    def log_odds_ratio_to_d(log_or: float) -> ConversionResult:
        """Convert a log odds ratio to Cohen's *d*."""
        return log_odds_ratio_to_d(log_or)

    @staticmethod
    def odds_ratio_to_r(or_: float, n1: int, n2: int) -> ConversionResult:
        """Convert an odds ratio to Pearson *r* (via *d*)."""
        return odds_ratio_to_r(or_, n1, n2)


__all__ = [
    "EffectSizeConverter",
    "d_to_r",
    "d_to_r_equal_n",
    "d_to_odds_ratio",
    "d_to_cohens_f",
    "d_to_r_squared",
    "d_to_eta_squared",
    "r_to_d",
    "r_to_d_equal_n",
    "r_to_fisher_z",
    "fisher_z_to_r",
    "odds_ratio_to_d",
    "log_odds_ratio_to_d",
    "odds_ratio_to_r",
]
