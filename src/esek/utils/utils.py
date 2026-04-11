"""
Utility functions for the Calculator package.

This module re-exports all utilities from the specialized submodules
:mod:`ci_utils` (confidence interval helpers) and :mod:`math_utils`
(general mathematical and statistical helpers).  Import directly from
those modules when you need only one category of functions.
"""

from .ci_utils import (
    ci_from_cohens_simple,
    compute_fisher_confidence_interval,
    pivotal_ci_t,
    ci_from_cohens_paired,
    ci_from_cohens_d_t_test,
    ci_from_cohens_d_two_samples,
    central_ci_paired,
    calculate_se_pooled,
    ci_t_prime,
    ci_adjusted_lambda_prime,
    ci_mag,
    ci_morris,
    ci_ncp,
)

from .math_utils import (
    not_implemented,
    convert_results_to_dict,
    density,
    area_under_function,
    winsorized_variance,
    winsorized_correlation,
)

__all__ = [
    # CI utilities
    "ci_from_cohens_simple",
    "compute_fisher_confidence_interval",
    "pivotal_ci_t",
    "ci_from_cohens_paired",
    "ci_from_cohens_d_t_test",
    "ci_from_cohens_d_two_samples",
    "central_ci_paired",
    "calculate_se_pooled",
    "ci_t_prime",
    "ci_adjusted_lambda_prime",
    "ci_mag",
    "ci_morris",
    "ci_ncp",
    # Math utilities
    "not_implemented",
    "convert_results_to_dict",
    "density",
    "area_under_function",
    "winsorized_variance",
    "winsorized_correlation",
]
