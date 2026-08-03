"""Confidence-interval methods for effect sizes.

This package exposes both tuple-returning numeric helpers and typed result
APIs such as :class:`.CohensDCI` and :class:`.EtaSquaredCI`.
"""

from .ci_mean_difference import (
    central_ci_one_sample,
    central_ci_paired,
    central_ci_two_samples,
    pivotal_ci_one_sample,
    ncp_ci_one_sample,
    multiple_se_ci_two_samples,
)
from .ci_correlations import (
    fisher_z_ci,
    spearman_ci,
    cramer_v_ci,
    cohens_w_ci,
    contingency_coefficient_ci,
    SpearmanCIResult,
    CramerVCIResult,
    CohensWCIResult,
    ContingencyCoefficientCIResult,
)
from .ci_odds_ratio import log_scale_ci
from .ci_eta_squared import EtaSquaredCI
from .ci_cohens_d import CohensDCI, CohensDCIResult
from .ci_dispersion import mad_ci, sd_ci, MADCIResult, SDCIResult
from .ci_proportions import (
    ProportionCI,
    ProportionCIResult,
    PairedProportionCIResult,
    IndependentProportionCIResult,
)

__all__ = [
    "central_ci_one_sample",
    "central_ci_paired",
    "central_ci_two_samples",
    "pivotal_ci_one_sample",
    "ncp_ci_one_sample",
    "multiple_se_ci_two_samples",
    "fisher_z_ci",
    "spearman_ci",
    "cramer_v_ci",
    "cohens_w_ci",
    "contingency_coefficient_ci",
    "SpearmanCIResult",
    "CramerVCIResult",
    "CohensWCIResult",
    "ContingencyCoefficientCIResult",
    "log_scale_ci",
    "EtaSquaredCI",
    "CohensDCI",
    "CohensDCIResult",
    "mad_ci",
    "sd_ci",
    "MADCIResult",
    "SDCIResult",
    "ProportionCI",
    "ProportionCIResult",
    "PairedProportionCIResult",
    "IndependentProportionCIResult",
]

