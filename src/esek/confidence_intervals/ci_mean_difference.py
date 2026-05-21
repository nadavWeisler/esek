"""Confidence-interval methods for standardised mean differences (d, g).

All functions return ``(ci_lower, ci_upper, standard_error)`` unless
noted.  They are pure numerical helpers; callers are responsible for
wrapping results in typed objects.

References
----------
- Hedges & Olkin (1985) *Statistical Methods for Meta-Analysis*
- Morris (2000) A meta-analytic review of the SE of Cohen's d
- Hunter & Schmidt (2004) *Methods of Meta-Analysis*
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def _bias_correction(df: float) -> float:
    """Bias-correction factor J for Hedges' g (log-gamma form)."""
    return math.exp(
        math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2)
    )


def central_ci_one_sample(
    effect_size: float,
    sample_size: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    """Central (z-based) CI for Cohen's d from a one-sample test.

    Uses the Hedges & Olkin (1985) SE formula.

    Parameters
    ----------
    effect_size:
        Cohen's d.
    sample_size:
        Sample size n.
    confidence_level:
        Desired CI level, e.g. 0.95.

    Returns
    -------
    (ci_lower, ci_upper, standard_error)
    """
    se = float(np.sqrt(1.0 / sample_size + effect_size**2 / (2.0 * sample_size)))
    z = float(stats.norm.ppf(0.5 + confidence_level / 2))
    return effect_size - se * z, effect_size + se * z, se


def central_ci_paired(
    effect_size: float,
    sample_size: int,
    confidence_level: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Central CI for Cohen's d from a paired-samples design.

    Returns SE estimates from multiple formulas (True, Morris, Hedges,
    Hedges-Olkin, MLE, Large-N, Small-N) in addition to the CI.

    Parameters
    ----------
    effect_size:
        Cohen's d (or dav / drm).
    sample_size:
        Number of pairs n.
    confidence_level:
        Desired CI level.

    Returns
    -------
    9-tuple: (ci_lower, ci_upper, se_true, se_morris, se_hedges,
              se_hedges_olkin, se_mle, se_large_n, se_small_n)
    """
    if sample_size <= 3:
        raise ValueError(
            f"sample_size must be > 3 for paired central CI, got {sample_size}."
        )
    df = sample_size - 1
    J = _bias_correction(df)
    se_true = float(
        np.sqrt(
            (df / (df - 2)) * (1.0 / sample_size) * (1 + effect_size**2 * sample_size)
            - effect_size**2 / J**2
        )
    )
    morris_approx = 1 - 3.0 / (4 * (df - 1) - 1)
    se_morris = float(
        np.sqrt(
            (df / (df - 2)) * (1.0 / sample_size) * (1 + effect_size**2 * sample_size)
            - effect_size**2 / morris_approx**2
        )
    )
    se_hedges = float(np.sqrt(1.0 / sample_size + effect_size**2 / (2.0 * df)))
    se_hedges_olkin = float(np.sqrt(1.0 / sample_size + effect_size**2 / (2.0 * sample_size)))
    se_mle = float(np.sqrt(se_hedges * (df + 2) / df))
    se_large_n = float(np.sqrt(1.0 / sample_size * (1 + effect_size**2 / 8)))
    se_small_n = float(np.sqrt(se_large_n * (df + 1) / (df - 1)))
    z = float(stats.norm.ppf(0.5 + confidence_level / 2))
    ci_lower = effect_size - se_true * z
    ci_upper = effect_size + se_true * z
    return ci_lower, ci_upper, se_true, se_morris, se_hedges, se_hedges_olkin, se_mle, se_large_n, se_small_n


def central_ci_two_samples(
    effect_size: float,
    sample_size_1: int,
    sample_size_2: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    """Central (z-based) CI for Cohen's d from a two-independent-samples test.

    Uses the Hedges & Olkin (1985) simple SE formula.

    Parameters
    ----------
    effect_size:
        Cohen's d.
    sample_size_1, sample_size_2:
        Group sample sizes.
    confidence_level:
        Desired CI level.

    Returns
    -------
    (ci_lower, ci_upper, standard_error)
    """
    n = sample_size_1 + sample_size_2
    se = float(np.sqrt(
        (sample_size_1 + sample_size_2) / (sample_size_1 * sample_size_2)
        + effect_size**2 / (2.0 * n)
    ))
    z = float(stats.norm.ppf(0.5 + confidence_level / 2))
    return effect_size - se * z, effect_size + se * z, se


def multiple_se_ci_two_samples(
    effect_size: float,
    sample_size_1: int,
    sample_size_2: int,
    confidence_level: float,
) -> tuple:
    """Central CI with multiple SE approximations for two-independent-samples d.

    Returns all seven SE variants alongside the CI (uses the "true" SE for the
    CI bounds).

    Returns
    -------
    9-tuple: (ci_lower, ci_upper, se_true, se_morris, se_hedges,
              se_hedges_olkin, se_mle, se_large_n, se_small_n)
    """
    n = sample_size_1 + sample_size_2
    df = n - 2
    if df <= 2:
        raise ValueError(f"df must be > 2 for two-sample CI, got df={df}.")
    J = _bias_correction(df)
    n_harm = 2.0 / (1.0 / sample_size_1 + 1.0 / sample_size_2)
    a = n_harm / 2.0
    se_true = float(
        np.sqrt(
            (df / (df - 2)) * (1.0 / a) * (1 + effect_size**2 * a) - effect_size**2 / J**2
        )
    )
    morris_approx = 1 - 3.0 / (4 * (df - 1) - 1)
    se_morris = float(
        np.sqrt(
            (df / (df - 2)) * (1.0 / a) * (1 + effect_size**2 * a)
            - effect_size**2 / morris_approx**2
        )
    )
    se_hedges = float(np.sqrt(1.0 / a + effect_size**2 / (2.0 * df)))
    se_hedges_olkin = float(np.sqrt(1.0 / a + effect_size**2 / (2.0 * n)))
    se_mle = float(np.sqrt(se_hedges * (df + 2) / df))
    se_large_n = float(np.sqrt(1.0 / a * (1 + effect_size**2 / 8)))
    se_small_n = float(np.sqrt(se_large_n * (df + 1) / (df - 1)))
    z = float(stats.norm.ppf(0.5 + confidence_level / 2))
    ci_lower = effect_size - se_true * z
    ci_upper = effect_size + se_true * z
    return ci_lower, ci_upper, se_true, se_morris, se_hedges, se_hedges_olkin, se_mle, se_large_n, se_small_n


def pivotal_ci_one_sample(
    t_score: float,
    sample_size: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Pivotal (non-central t) CI for Cohen's d from a one-sample t-test.

    Parameters
    ----------
    t_score:
        Observed t-statistic.
    sample_size:
        Sample size n.
    confidence_level:
        Desired CI level.

    Returns
    -------
    (ci_lower, ci_upper)
    """
    from ..utils.distribution_helpers import pivotal_ci_t
    df = sample_size - 1
    return pivotal_ci_t(t_score, df, sample_size, confidence_level)


def ncp_ci_one_sample(
    effect_size: float,
    sample_size: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Non-central parameter (NCP) CI for Cohen's d from a one-sample t-test.

    Parameters
    ----------
    effect_size:
        Cohen's d.
    sample_size:
        Sample size n.
    confidence_level:
        Desired CI level.

    Returns
    -------
    (ci_lower, ci_upper)
    """
    from ..utils.distribution_helpers import ci_ncp
    return ci_ncp(effect_size, sample_size, confidence_level)
