"""Confidence-interval methods for effect sizes.

All public functions return ``(ci_lower, ci_upper)`` tuples. For
structured results use :class:`~esek.results.base.ConfidenceIntervalResult`.
"""

from .ci_mean_difference import (
    central_ci_one_sample,
    central_ci_paired,
    central_ci_two_samples,
    pivotal_ci_one_sample,
    ncp_ci_one_sample,
    multiple_se_ci_two_samples,
)
from .ci_correlations import fisher_z_ci
from .ci_odds_ratio import log_scale_ci

__all__ = [
    "central_ci_one_sample",
    "central_ci_paired",
    "central_ci_two_samples",
    "pivotal_ci_one_sample",
    "ncp_ci_one_sample",
    "multiple_se_ci_two_samples",
    "fisher_z_ci",
    "log_scale_ci",
]
