"""
Confidence intervals for proportions.

Migrated from: stats/CI_Constructor/4_Proportions/Proportions CI.py (dev branch)

Covers three designs:
1. One-sample proportion (12 CI methods)
2. Paired-samples proportion difference (6 CI methods)
3. Independent-samples proportion difference (12 CI methods, including Gart-Nam)

All methods are pure Python (no R/rpy2 dependency). The Gart-Nam CI, originally
implemented in R, is re-implemented here in Python using the same algorithm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import newton, root_scalar
from scipy.stats import beta, binom, norm
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint

from ..core.exceptions import InvalidInputError
from ..core.validation import (
    validate_confidence_level,
    validate_proportion,
    validate_sample_size,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProportionCIResult:
    """Confidence intervals for a single proportion (one-sample design).

    All interval fields are ``(lower, upper)`` tuples.
    """

    proportion: float
    sample_size: int
    confidence_level: float
    agresti_coull: tuple[float, float] | None = None
    wald: tuple[float, float] | None = None
    wald_corrected: tuple[float, float] | None = None
    wilson: tuple[float, float] | None = None
    wilson_corrected: tuple[float, float] | None = None
    logit: tuple[float, float] | None = None
    jeffreys: tuple[float, float] | None = None
    clopper_pearson: tuple[float, float] | None = None
    arcsine: tuple[float, float] | None = None
    pratt: tuple[float, float] | None = None
    blaker: tuple[float, float] | None = None
    mid_p: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairedProportionCIResult:
    """Confidence intervals for a paired-samples proportion difference.

    All interval fields are ``(lower, upper)`` tuples.
    """

    proportion_1: float
    proportion_2: float
    difference: float
    sample_size: int
    confidence_level: float
    wald: tuple[float, float] | None = None
    wald_edwards: tuple[float, float] | None = None
    wald_yates: tuple[float, float] | None = None
    agresti_min: tuple[float, float] | None = None
    bonett_price: tuple[float, float] | None = None
    newcomb: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IndependentProportionCIResult:
    """Confidence intervals for an independent-samples proportion difference.

    All interval fields are ``(lower, upper)`` tuples.
    """

    proportion_1: float
    proportion_2: float
    sample_size_1: int
    sample_size_2: int
    difference: float
    confidence_level: float
    wald: tuple[float, float] | None = None
    wald_corrected: tuple[float, float] | None = None
    haldane: tuple[float, float] | None = None
    jeffreys_perks: tuple[float, float] | None = None
    miettinen_nurminen: tuple[float, float] | None = None
    mee: tuple[float, float] | None = None
    agresti_caffo: tuple[float, float] | None = None
    wilson: tuple[float, float] | None = None
    wilson_corrected: tuple[float, float] | None = None
    hauck_anderson: tuple[float, float] | None = None
    brown_li_jeffreys: tuple[float, float] | None = None
    gart_nam: tuple[float, float] | None = None
    newcomb: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _clip_proportion(lo: float, hi: float) -> tuple[float, float]:
    """Clip CI bounds to [0, 1]."""
    return (max(0.0, float(lo)), min(1.0, float(hi)))


def _clip_difference(lo: float, hi: float) -> tuple[float, float]:
    """Clip CI bounds to [-1, 1]."""
    return (max(-1.0, float(lo)), min(1.0, float(hi)))


def _blaker_ci(x: float, n: int, conf_level: float = 0.95, tol: float = 1e-5) -> tuple[float, float]:
    """Blaker (2000) exact CI for a binomial proportion.

    Algorithm: start from Clopper-Pearson bounds and tighten until the
    acceptance probability condition is met.
    """

    def acceptance_prob(x: float, n: int, p: float) -> float:
        p1 = 1.0 - binom.cdf(x - 1, n, p)
        p2 = binom.cdf(x, n, p)
        a1 = p1 + binom.cdf(binom.ppf(p1, n, p) - 1, n, p)
        a2 = p2 + 1 - binom.cdf(binom.ppf(1 - p2, n, p), n, p)
        return float(min(a1, a2))

    alpha = 1.0 - conf_level
    lo = float(beta.ppf(alpha / 2, x, n - x + 1))
    hi = float(beta.ppf(1 - alpha / 2, x + 1, n - x))

    while x != 0 and acceptance_prob(x, n, lo + tol) < alpha:
        lo += tol
    while x != n and acceptance_prob(x, n, hi - tol) < alpha:
        hi -= tol

    return _clip_proportion(lo, hi)


def _midp_ci(x: float, n: int, conf_level: float = 0.95) -> tuple[float, float]:
    """Mid-p binomial CI (Lancaster, 1961)."""

    def f_low(pi: float) -> float:
        return 0.5 * binom.pmf(x, n, pi) + binom.cdf(x - 1, n, pi) - (1 + conf_level) / 2

    def f_up(pi: float) -> float:
        return 0.5 * binom.pmf(x, n, pi) + binom.cdf(x - 1, n, pi) - (1 - conf_level) / 2

    lo = newton(f_low, x / n)
    hi = newton(f_up, x / n)
    return _clip_proportion(lo, hi)


def _mn_se_calculate(
    p1: float, n1: int, p2: float, n2: int, delta: float
) -> tuple[float, float]:
    """Constrained MLE standard errors for Miettinen-Nurminen and MEE methods."""
    k = n2 / n1
    a = 1.0 + k
    b = -(1 + k + p1 + k * p2 + delta * (k + 2))
    c = delta ** 2 + delta * (2 * p1 + k + 1) + p1 + k * p2
    d = -p1 * delta * (1 + delta)
    disc = (b / a / 3) ** 3 - b * c / (6 * a * a) + d / a / 2
    v_val = 0.0 if abs(disc) < np.finfo(float).eps else disc
    s = math.sqrt(max(0.0, (b / a / 3) ** 2 - c / a / 3))
    u = (1.0 if v_val >= 0 else -1.0) * s
    cos_arg = float(np.clip(v_val / u ** 3 if u != 0 else 0, -1, 1))
    w_angle = (math.pi + math.acos(cos_arg)) / 3
    p1h = 2 * u * math.cos(w_angle) - b / a / 3
    p2h = p1h - delta
    p1h = max(0.0, min(1.0, p1h))
    p2h = max(0.0, min(1.0, p2h))
    n = n1 + n2
    var_mn = (p1h * (1 - p1h) / n1 + p2h * (1 - p2h) / n2) * n / (n - 1)
    var_mee = p1h * (1 - p1h) / n1 + p2h * (1 - p2h) / n2
    return math.sqrt(max(0, var_mn)), math.sqrt(max(0, var_mee))


def _mn_pval(p1: float, n1: int, p2: float, n2: int, delta: float) -> tuple[float, float]:
    diff = p1 - p2
    se_mn, se_mee = _mn_se_calculate(p1, n1, p2, n2, delta)
    z_mn = (diff - delta) / se_mn if se_mn > 0 else 0.0
    z_mee = (diff - delta) / se_mee if se_mee > 0 else 0.0
    p_mn = 2 * min(norm.cdf(z_mn), 1 - norm.cdf(z_mn))
    p_mee = 2 * min(norm.cdf(z_mee), 1 - norm.cdf(z_mee))
    return float(p_mn), float(p_mee)


def _mn_ci(
    p1: float, n1: int, p2: float, n2: int, confidence_level: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Miettinen-Nurminen and MEE CIs via root-finding."""
    alpha = 1 - confidence_level
    diff = p1 - p2

    def root_mn(delta: float) -> float:
        return _mn_pval(p1, n1, p2, n2, delta)[0] - alpha

    def root_mee(delta: float) -> float:
        return _mn_pval(p1, n1, p2, n2, delta)[1] - alpha

    mn_lo = root_scalar(root_mn, bracket=[-1, diff]).root
    mn_hi = root_scalar(root_mn, bracket=[diff, 0.999999]).root
    mee_lo = root_scalar(root_mee, bracket=[-1, diff]).root
    mee_hi = root_scalar(root_mee, bracket=[diff, 0.999999]).root

    return _clip_difference(mn_lo, mn_hi), _clip_difference(mee_lo, mee_hi)


def _gart_nam_score(p1: float, n1: int, p2: float, n2: int, theta: float) -> float:
    """Gart-Nam (1988) score statistic for proportion difference theta.

    Pure-Python re-implementation of the R scoretheta function used in the
    original dev-branch CI_Constructor.py.
    """
    prop_diff = (p1 - p2) - theta
    n = n1 + n2
    a = (n1 + 2 * n2) * theta - n - (p1 * n1 + p2 * n2)
    b = (a / n / 3) ** 3 - a * ((n2 * theta - n - 2 * p2 * n2) * theta + (p1 * n1 + p2 * n2)) / (6 * n * n) + (p2 * n2 * theta * (1 - theta)) / n / 2
    c = math.copysign(1, b) * math.sqrt(max(0.0, (a / n / 3) ** 2 - ((n2 * theta - n - 2 * p2 * n2) * theta + (p1 * n1 + p2 * n2)) / n / 3))
    cos_arg = float(np.clip(b / c ** 3 if c != 0 else 0, -1, 1))
    p2d = max(0.0, min(1.0, 2 * c * math.cos((math.pi + math.acos(cos_arg)) / 3) - a / n / 3))
    p1d = max(0.0, min(1.0, p2d + theta))
    variance = max(0.0, p1d * (1 - p1d) / n1 + p2d * (1 - p2d) / n2)
    if variance == 0:
        return 0.0
    sc_term = (p1d * (1 - p1d) * (1 - 2 * p1d) / (n1 ** 2) - p2d * (1 - p2d) * (1 - 2 * p2d) / (n2 ** 2)) / (6 * variance ** 1.5)
    if sc_term == 0:
        return prop_diff / math.sqrt(variance)
    disc = 1.0 - 4 * sc_term * (-(prop_diff / math.sqrt(variance) + sc_term))
    return (-1 + math.sqrt(max(0, disc))) / (2 * sc_term)


def _gart_nam_ci(
    p1: float, n1: int, p2: float, n2: int, confidence_level: float
) -> tuple[float, float]:
    """Gart-Nam (1988) CI for proportion difference."""
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    diff = p1 - p2

    def lower_func(theta: float) -> float:
        return _gart_nam_score(p1, n1, p2, n2, theta) - z

    def upper_func(theta: float) -> float:
        return _gart_nam_score(p1, n1, p2, n2, theta) + z

    try:
        lo = root_scalar(lower_func, bracket=[-0.999, diff]).root
        hi = root_scalar(upper_func, bracket=[diff, 0.999]).root
        return _clip_difference(lo, hi)
    except Exception:
        return (float("nan"), float("nan"))


# ---------------------------------------------------------------------------
# Public API class
# ---------------------------------------------------------------------------

class ProportionCI:
    """Confidence intervals for proportions.

    Three static methods cover:
    - ``one_sample``: CI for a single proportion
    - ``paired_samples``: CI for a paired proportion difference
    - ``independent_samples``: CI for an independent proportion difference
    """

    @staticmethod
    def one_sample(
        proportion: float,
        n: int,
        confidence_level: float = 0.95,
    ) -> ProportionCIResult:
        """Compute 12 CIs for a one-sample proportion.

        Parameters
        ----------
        proportion:
            Observed sample proportion (0 < p < 1).
        n:
            Sample size (≥ 2).
        confidence_level:
            Nominal confidence level, e.g. 0.95 for 95 % CI.

        Returns
        -------
        ProportionCIResult
            Dataclass with all 12 CI methods as ``(lower, upper)`` tuples.

        Methods
        -------
        1. Agresti-Coull
        2. Wald
        3. Wald corrected (continuity)
        4. Wilson
        5. Wilson corrected
        6. Logit
        7. Jeffreys (Bayesian)
        8. Clopper-Pearson (exact)
        9. Arcsine (Kulynskaya)
        10. Pratt
        11. Blaker (exact)
        12. Mid-p
        """
        validate_proportion(proportion)
        validate_sample_size(n)
        validate_confidence_level(confidence_level)

        x = proportion * n  # successes
        alpha = 1.0 - confidence_level
        z = float(norm.ppf(confidence_level + (1 - confidence_level) / 2))

        # 1. Agresti-Coull
        ac = proportion_confint(x, n, alpha, method="agresti_coull")
        agresti_coull = _clip_proportion(ac[0], ac[1])

        # 2. Wald
        wd = proportion_confint(x, n, alpha, method="normal")
        wald = _clip_proportion(wd[0], wd[1])

        # 3. Wald corrected (simple continuity correction 0.05/n)
        correction = 0.05 / n
        wald_corrected = _clip_proportion(wald[0] - correction, wald[1] + correction)

        # 4. Wilson
        wi = proportion_confint(x, n, alpha, method="wilson")
        wilson = _clip_proportion(wi[0], wi[1])

        # 5. Wilson corrected
        lo_wc = (2 * x + z ** 2 - 1 - z * math.sqrt(
            z ** 2 - 2 - 1 / n + 4 * (x / n) * (n * (1 - x / n) + 1)
        )) / (2 * (n + z ** 2))
        hi_wc = min(1.0, (2 * x + z ** 2 + 1 + z * math.sqrt(
            z ** 2 + 2 - 1 / n + 4 * (x / n) * (n * (1 - x / n) - 1)
        )) / (2 * (n + z ** 2)))
        wilson_corrected = _clip_proportion(lo_wc, hi_wc)

        # 6. Logit
        q = n - x
        if x > 0 and q > 0:
            lhat = math.log(x / q)
            term = math.sqrt(n / (x * q))
            logit = _clip_proportion(
                math.exp(lhat - z * term) / (1 + math.exp(lhat - z * term)),
                math.exp(lhat + z * term) / (1 + math.exp(lhat + z * term)),
            )
        else:
            logit = (0.0, 1.0)

        # 7. Jeffreys (Beta(x+0.5, n-x+0.5) prior)
        lo_j = float(beta.ppf(alpha / 2, x + 0.5, n - x + 0.5))
        hi_j = min(1.0, float(beta.ppf(1 - alpha / 2, x + 0.5, n - x + 0.5)))
        jeffreys = _clip_proportion(lo_j, hi_j)

        # 8. Clopper-Pearson (exact)
        lo_cp = float(beta.ppf(alpha / 2, x, n - x + 1))
        hi_cp = min(1.0, float(beta.ppf(1 - alpha / 2, x + 1, n - x)))
        clopper_pearson = _clip_proportion(lo_cp, hi_cp)

        # 9. Arcsine (Kulynskaya)
        ptilde = (x + 0.375) / (n + 0.75)
        lo_arc = math.sin(math.asin(math.sqrt(ptilde)) - 0.5 * z / math.sqrt(n)) ** 2
        hi_arc = min(1.0, math.sin(math.asin(math.sqrt(ptilde)) + 0.5 * z / math.sqrt(n)) ** 2)
        arcsine = _clip_proportion(lo_arc, hi_arc)

        # 10. Pratt
        x1 = x + 1
        q_alt = n - x
        if q_alt == 0:
            pratt = (1.0, 1.0)
        else:
            A = (x1 / q_alt) ** 2
            B = 81 * x1 * q_alt - 9 * n - 8
            C = -3 * z * math.sqrt(9 * x1 * q_alt * (9 * n + 5 - z ** 2) + n + 1)
            D = 81 * x1 ** 2 - 9 * x1 * (2 + z ** 2) + 1
            E = 1 + A * ((B + C) / D) ** 3

            A2 = (x / (q_alt - 1)) ** 2 if q_alt > 1 else float("inf")
            B2 = 81 * x * (q_alt - 1) - 9 * n - 8
            C2 = 3 * z * math.sqrt(9 * x * (q_alt - 1) * (9 * n + 5 - z ** 2) + n + 1)
            D2 = 81 * x ** 2 - 9 * x * (2 + z ** 2) + 1
            E2 = 1 + A2 * ((B2 + C2) / D2) ** 3 if abs(D2) > 1e-12 and A2 != float("inf") else float("inf")

            hi_pratt = min(1.0, 1.0 / E) if E != 0 else 1.0
            lo_pratt = max(0.0, 1.0 / E2) if E2 != float("inf") and E2 != 0 else 0.0
            pratt = (lo_pratt, hi_pratt)

        # 11. Blaker (exact)
        blaker = _blaker_ci(x, n, confidence_level)

        # 12. Mid-p
        try:
            mid_p = _midp_ci(x, n, confidence_level)
        except Exception:
            mid_p = None

        return ProportionCIResult(
            proportion=proportion,
            sample_size=n,
            confidence_level=confidence_level,
            agresti_coull=agresti_coull,
            wald=wald,
            wald_corrected=wald_corrected,
            wilson=wilson,
            wilson_corrected=wilson_corrected,
            logit=logit,
            jeffreys=jeffreys,
            clopper_pearson=clopper_pearson,
            arcsine=arcsine,
            pratt=pratt,
            blaker=blaker,
            mid_p=mid_p,
        )

    @staticmethod
    def paired_samples(
        proportion_1: float,
        proportion_2: float,
        proportion_both_success: float,
        n_pairs: int,
        confidence_level: float = 0.95,
    ) -> PairedProportionCIResult:
        """Compute 6 CIs for a paired proportion difference.

        This covers the McNemar-type design where both samples are measured
        on the same subjects.

        Parameters
        ----------
        proportion_1:
            Proportion of successes in sample 1 (0 < p < 1).
        proportion_2:
            Proportion of successes in sample 2 (0 < p < 1).
        proportion_both_success:
            Proportion of pairs where both samples were successes.
        n_pairs:
            Number of pairs (sample size).
        confidence_level:
            Nominal confidence level.

        Returns
        -------
        PairedProportionCIResult
            6 CI methods as ``(lower, upper)`` tuples.

        Methods
        -------
        1. Wald
        2. Wald corrected (Edwards)
        3. Wald corrected (Yates)
        4. Agresti-Min (2005)
        5. Bonett-Price (2005)
        6. Newcomb (2006, square-and-add)
        """
        validate_proportion(proportion_1)
        validate_proportion(proportion_2)
        validate_confidence_level(confidence_level)
        validate_sample_size(n_pairs)

        # Reconstruct 2x2 cell counts
        yy = proportion_both_success * n_pairs     # yes-yes
        yn = proportion_1 * n_pairs - yy            # yes-no (discordant)
        ny = proportion_2 * n_pairs - yy            # no-yes (discordant)
        nn = n_pairs - (yy + yn + ny)               # no-no

        n = yy + yn + ny + nn
        p1t = yy + yn
        p2t = yy + ny
        p1 = p1t / n
        p2 = p2t / n
        diff = p1 - p2

        z = float(norm.ppf(confidence_level + (1 - confidence_level) / 2))

        # 1. Wald
        se_wald = math.sqrt((yn + ny) - ((yn - ny) ** 2) / n) / n
        wald = _clip_difference(diff - z * se_wald, diff + z * se_wald)

        # 2. Wald corrected (Edwards / Fleiss et al. 2003)
        wald_edwards = _clip_difference(
            diff - z * se_wald - 1 / n,
            diff + z * se_wald + 1 / n,
        )

        # 3. Wald corrected (Yates)
        se_yates = math.sqrt((yn + ny) - ((yn - ny - 1) ** 2) / n) / n
        wald_yates = _clip_difference(diff - z * se_yates, diff + z * se_yates)

        # 4. Agresti-Min (2005) — add 0.5 to discordant cells
        se_am = math.sqrt(
            ((yn + 0.5) + (ny + 0.5)) - (((yn + 0.5) - (ny + 0.5)) ** 2) / (n + 2)
        ) / (n + 2)
        d_am = ((yn + 0.5) - (ny + 0.5)) / (n + 2)
        agresti_min = _clip_difference(d_am - z * se_am, d_am + z * se_am)

        # 5. Bonett-Price (2005)
        p1_adj = (yn + 1) / (n + 2)
        p2_adj = (ny + 1) / (n + 2)
        se_bp = math.sqrt((p1_adj + p2_adj - (p2_adj - p1_adj) ** 2) / (n + 2))
        bonett_price = _clip_difference(
            p1_adj - p2_adj - z * se_bp,
            p1_adj - p2_adj + z * se_bp,
        )

        # 6. Newcomb (square-and-add; 2006)
        def _wilson_bounds(p: float, nn: int) -> tuple[float, float]:
            eps = (2 * p * nn + z ** 2) / (2 * nn + 2 * z ** 2)
            half_width = z * math.sqrt(nn) / (nn + z ** 2) * math.sqrt(
                p * (1 - p) + z ** 2 / (4 * nn)
            )
            return max(0.0, eps - half_width), min(1.0, eps + half_width)

        lo1, hi1 = _wilson_bounds(p1, n)
        lo2, hi2 = _wilson_bounds(p2, n)

        # Compute correlation correction
        if p1t == 0 or p2t == 0 or (n - p1t) == 0 or (n - p2t) == 0:
            corr = 0.0
        else:
            marg_prod = p1t * p2t * (n - p1t) * (n - p2t)
            cells_prod = yy * nn - ny * yn
            if cells_prod > n / 2:
                corr = (cells_prod - n / 2) / math.sqrt(marg_prod)
            elif 0 <= cells_prod <= n / 2:
                corr = 0.0
            else:
                corr = cells_prod / math.sqrt(marg_prod)

        lo_nc = diff - math.sqrt(
            (p1 - lo1) ** 2 + (hi2 - p2) ** 2 - 2 * corr * (p1 - lo1) * (hi2 - p2)
        )
        hi_nc = diff + math.sqrt(
            (p2 - lo2) ** 2 + (hi1 - p1) ** 2 - 2 * corr * (p2 - lo2) * (hi1 - p1)
        )
        newcomb = _clip_difference(lo_nc, hi_nc)

        return PairedProportionCIResult(
            proportion_1=p1,
            proportion_2=p2,
            difference=diff,
            sample_size=n_pairs,
            confidence_level=confidence_level,
            wald=wald,
            wald_edwards=wald_edwards,
            wald_yates=wald_yates,
            agresti_min=agresti_min,
            bonett_price=bonett_price,
            newcomb=newcomb,
        )

    @staticmethod
    def independent_samples(
        proportion_1: float,
        n1: int,
        proportion_2: float,
        n2: int,
        confidence_level: float = 0.95,
    ) -> IndependentProportionCIResult:
        """Compute 12 CIs for an independent-samples proportion difference.

        Parameters
        ----------
        proportion_1:
            Proportion in group 1.
        n1:
            Sample size of group 1.
        proportion_2:
            Proportion in group 2.
        n2:
            Sample size of group 2.
        confidence_level:
            Nominal confidence level.

        Returns
        -------
        IndependentProportionCIResult
            13 CI methods as ``(lower, upper)`` tuples.

        Methods
        -------
        1. Wald
        2. Wald corrected (continuity)
        3. Haldane-Anscombe
        4. Jeffreys-Perks
        5. Miettinen-Nurminen (score)
        6. MEE (score)
        7. Agresti-Caffo
        8. Wilson (Newcomb square-and-add)
        9. Wilson corrected (Newcomb square-and-add corrected)
        10. Hauck-Anderson
        11. Brown-Li-Jeffreys
        12. Gart-Nam (1988, score)
        13. Newcomb (statsmodels)
        """
        validate_proportion(proportion_1)
        validate_proportion(proportion_2)
        validate_sample_size(n1, name="n1")
        validate_sample_size(n2, name="n2")
        validate_confidence_level(confidence_level)

        p1, p2 = proportion_1, proportion_2
        x1, x2 = p1 * n1, p2 * n2
        alpha = 1.0 - confidence_level
        diff = p1 - p2
        z = float(norm.ppf(1 - alpha / 2))

        # 1. Wald
        se_wald = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        wald = _clip_difference(diff - z * se_wald, diff + z * se_wald)

        # 2. Wald corrected
        wald_corrected = _clip_difference(
            diff - (0.5 * (1 / n1 + 1 / n2) + z * se_wald),
            diff + (0.5 * (1 / n1 + 1 / n2) + z * se_wald),
        )

        # 3. Haldane (asymmetric Wald variant)
        psi = (p1 + p2) / 2
        v = (1 / n1 - 1 / n2) / 4
        mu = (1 / n1 + 1 / n2) / 4
        theta_h = (diff + z ** 2 * v * (1 - 2 * psi)) / (1 + z ** 2 * mu)
        term_h = z / (1 + z ** 2 * mu) * math.sqrt(
            mu * (4 * psi * (1 - psi) - diff ** 2)
            + 2 * v * (1 - 2 * psi) * diff
            + 4 * z ** 2 * mu ** 2 * (1 - psi) * psi
            + z ** 2 * v ** 2 * (1 - 2 * psi) ** 2
        )
        haldane = _clip_difference(theta_h - term_h, theta_h + term_h)

        # 4. Jeffreys-Perks
        psi_jp = 0.5 * ((x1 + 0.5) / (n1 + 1) + (x2 + 0.5) / (n2 + 1))
        theta_jp = (diff + z ** 2 * v * (1 - 2 * psi_jp)) / (1 + z ** 2 * mu)
        term_jp = z / (1 + z ** 2 * mu) * math.sqrt(
            mu * (4 * psi_jp * (1 - psi_jp) - diff ** 2)
            + 2 * v * (1 - 2 * psi_jp) * diff
            + 4 * z ** 2 * mu ** 2 * (1 - psi_jp) * psi_jp
            + z ** 2 * v ** 2 * (1 - 2 * psi_jp) ** 2
        )
        jeffreys_perks = _clip_difference(theta_jp - term_jp, theta_jp + term_jp)

        # 5+6. Miettinen-Nurminen and MEE
        try:
            ci_mn, ci_mee = _mn_ci(p1, n1, p2, n2, confidence_level)
        except Exception:
            ci_mn, ci_mee = (float("nan"), float("nan")), (float("nan"), float("nan"))

        # 7. Agresti-Caffo
        p1_ac = (x1 + 1) / (n1 + 2)
        p2_ac = (x2 + 1) / (n2 + 2)
        se_ac = math.sqrt(p1_ac * (1 - p1_ac) / (n1 + 2) + p2_ac * (1 - p2_ac) / (n2 + 2))
        agresti_caffo = _clip_difference(
            p1_ac - p2_ac - z * se_ac,
            p1_ac - p2_ac + z * se_ac,
        )

        # 8. Wilson (Newcomb square-and-add)
        eps1 = (x1 + z ** 2 / 2) / (n1 + z ** 2)
        hw1 = z * math.sqrt(n1) / (n1 + z ** 2) * math.sqrt(p1 * (1 - p1) + z ** 2 / (4 * n1))
        lo_w1, hi_w1 = max(0.0, eps1 - hw1), min(1.0, eps1 + hw1)

        eps2 = (x2 + z ** 2 / 2) / (n2 + z ** 2)
        hw2 = z * math.sqrt(n2) / (n2 + z ** 2) * math.sqrt(p2 * (1 - p2) + z ** 2 / (4 * n2))
        lo_w2, hi_w2 = max(0.0, eps2 - hw2), min(1.0, eps2 + hw2)

        wilson = _clip_difference(
            diff - z * math.sqrt(lo_w1 * (1 - lo_w1) / n1 + hi_w2 * (1 - hi_w2) / n2),
            diff + z * math.sqrt(hi_w1 * (1 - hi_w1) / n1 + lo_w2 * (1 - lo_w2) / n2),
        )

        # 9. Wilson corrected
        lo_wc1 = (2 * x1 + z ** 2 - 1 - z * math.sqrt(z ** 2 - 2 - 1 / n1 + 4 * p1 * (n1 * (1 - p1) + 1))) / (2 * (n1 + z ** 2))
        hi_wc1 = (2 * x1 + z ** 2 + 1 + z * math.sqrt(z ** 2 + 2 - 1 / n1 + 4 * p1 * (n1 * (1 - p1) - 1))) / (2 * (n1 + z ** 2))
        lo_wc2 = (2 * x2 + z ** 2 - 1 - z * math.sqrt(z ** 2 - 2 - 1 / n2 + 4 * p2 * (n2 * (1 - p2) + 1))) / (2 * (n2 + z ** 2))
        hi_wc2 = (2 * x2 + z ** 2 + 1 + z * math.sqrt(z ** 2 + 2 - 1 / n2 + 4 * p2 * (n2 * (1 - p2) - 1))) / (2 * (n2 + z ** 2))
        wilson_corrected = _clip_difference(
            diff - math.sqrt((p1 - lo_wc1) ** 2 + (hi_wc2 - p2) ** 2),
            diff + math.sqrt((hi_wc1 - p1) ** 2 + (p2 - lo_wc2) ** 2),
        )

        # 10. Hauck-Anderson
        se_ha1 = p1 * (1 - p1) / (n1 - 1) if n1 > 1 else 0.0
        se_ha2 = p2 * (1 - p2) / (n2 - 1) if n2 > 1 else 0.0
        se_ha = math.sqrt(se_ha1 + se_ha2)
        ha_corr = 1 / (2 * min(n1, n2))
        hauck_anderson = _clip_difference(
            diff - ha_corr - z * se_ha,
            diff + ha_corr + z * se_ha,
        )

        # 11. Brown-Li-Jeffreys
        p1_blj = (x1 + 0.5) / (n1 + 1)
        p2_blj = (x2 + 0.5) / (n2 + 1)
        se_blj = math.sqrt(p1_blj * (1 - p1_blj) / n1 + p2_blj * (1 - p2_blj) / n2)
        brown_li_jeffreys = _clip_difference(
            p1_blj - p2_blj - z * se_blj,
            p1_blj - p2_blj + z * se_blj,
        )

        # 12. Gart-Nam (pure-Python re-implementation of dev-branch R code)
        try:
            gart_nam = _gart_nam_ci(p1, n1, p2, n2, confidence_level)
        except Exception:
            gart_nam = (float("nan"), float("nan"))

        # 13. Newcomb (statsmodels)
        try:
            nc = confint_proportions_2indep(x1, n1, x2, n2, method="newcomb", alpha=alpha)
            newcomb = _clip_difference(float(nc[0]), float(nc[1]))
        except Exception:
            newcomb = None

        return IndependentProportionCIResult(
            proportion_1=p1,
            proportion_2=p2,
            sample_size_1=n1,
            sample_size_2=n2,
            difference=diff,
            confidence_level=confidence_level,
            wald=wald,
            wald_corrected=wald_corrected,
            haldane=haldane,
            jeffreys_perks=jeffreys_perks,
            miettinen_nurminen=ci_mn,
            mee=ci_mee,
            agresti_caffo=agresti_caffo,
            wilson=wilson,
            wilson_corrected=wilson_corrected,
            hauck_anderson=hauck_anderson,
            brown_li_jeffreys=brown_li_jeffreys,
            gart_nam=gart_nam,
            newcomb=newcomb,
        )
