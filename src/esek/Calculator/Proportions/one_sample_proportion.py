"""Module for one-sample proportion tests and CIs."""

# ===== Standard library =====
import math
from dataclasses import dataclass
from typing import Optional

# ===== Third-party =====
import numpy as np
from scipy.stats import norm, beta, binom
from scipy.optimize import newton
from statsmodels.stats.proportion import proportion_confint

# ===== Internal (ESEK utils) =====
from ...utils import res, es, interfaces


@dataclass
class OneSampleProportionResults:
    """Results for one-sample proportion tests and CIs."""

    descriptive_statistics: Optional[res.DescriptiveStatistics] = None

    # Z tests (score test uses p0 in SE; Wald uses p̂ in SE). Corrected = continuity correction.
    z_test: Optional[res.InferentialStatistics] = None
    z_test_corrected: Optional[res.InferentialStatistics] = None
    z_test_wald: Optional[res.InferentialStatistics] = None
    z_test_wald_corrected: Optional[res.InferentialStatistics] = None

    # Effect sizes
    cohens_g: Optional[es.CohenG] = None
    cohens_h: Optional[es.CohenH] = None

    # Proportion CIs
    wald_type_confidence_interval: Optional[res.ConfidenceInterval] = None
    wald_type_corrected_confidence_interval: Optional[res.ConfidenceInterval] = None
    wilson_confidence_interval: Optional[res.ConfidenceInterval] = None
    wilson_corrected_confidence_interval: Optional[res.ConfidenceInterval] = None
    logit_confidence_interval: Optional[res.ConfidenceInterval] = None
    jeffreys_confidence_interval: Optional[res.ConfidenceInterval] = None
    clopper_pearson_confidence_interval: Optional[res.ConfidenceInterval] = None
    arcsine_confidence_interval: Optional[res.ConfidenceInterval] = None
    pratt_confidence_interval: Optional[res.ConfidenceInterval] = None
    blaker_confidence_interval: Optional[res.ConfidenceInterval] = None
    midp_confidence_interval: Optional[res.ConfidenceInterval] = None
    agresti_coull_confidence_interval: Optional[res.ConfidenceInterval] = None

    # Useful echoes
    sample_size: Optional[int] = None
    sample_proportion: Optional[float] = None
    population_proportion: Optional[float] = None
    confidence_level: Optional[float] = None


def blakers_ci(
    x: int, n: int, conf_level: float = 0.95, tol: float = 1e-5
) -> list[float]:
    """Blaker's exact CI for a binomial proportion."""

    def acceptance_probability(x, n, p):
        p1 = 1 - binom.cdf(x - 1, n, p)
        p2 = binom.cdf(x, n, p)
        a1 = p1 + binom.cdf(binom.ppf(p1, n, p) - 1, n, p)
        a2 = p2 + 1 - binom.cdf(binom.ppf(1 - p2, n, p), n, p)
        return min(a1, a2)

    lo = beta.ppf((1 - conf_level) / 2, max(x, 0), n - x + 1)
    hi = beta.ppf(1 - (1 - conf_level) / 2, x + 1, max(n - x, 0))

    while x != 0 and acceptance_probability(x, n, lo + tol) < (1 - conf_level):
        lo += tol
    while x != n and acceptance_probability(x, n, hi - tol) < (1 - conf_level):
        hi -= tol

    return [max(float(lo), 0.0), min(float(hi), 1.0)]


def calculate_midp(x: int, n: int, conf_level: float) -> tuple[float, float]:
    """Mid-p CI via root-finding on the binomial CDF + 0.5 * PMF."""

    def f_low(pi):
        return (
            0.5 * binom.pmf(x, n, pi) + binom.cdf(x - 1, n, pi) - (1 + conf_level) / 2
        )

    def f_up(pi):
        return (
            0.5 * binom.pmf(x, n, pi) + binom.cdf(x - 1, n, pi) - (1 - conf_level) / 2
        )

    start = (x + 0.5) / (n + 1.0)  # robust starting point
    lo = newton(f_low, start)
    hi = newton(f_up, start)
    return float(lo), float(hi)


class OneSampleProportions(interfaces.AbstractTest):
    """One-sample proportion tests and CIs."""

    @staticmethod
    def from_score():
        """Calculate results from score"""
        raise NotImplementedError("This method is not implemented yet.")

    @staticmethod
    def from_parameters(
        proportion_sample: float,
        sample_size: int,
        population_proportion: float,
        confidence_level: float = 0.95,
    ) -> OneSampleProportionResults:
        """Calculate results from parameters."""

        n = int(sample_size)
        p_hat = float(proportion_sample)
        p0 = float(population_proportion)
        df_cl = float(confidence_level)

        # --- guards & echoes
        p_hat = min(max(p_hat, 0.0), 1.0)
        p0 = min(max(p0, 0.0), 1.0)
        zcrit = norm.ppf(1 - (1 - df_cl) / 2)

        # --- z tests
        # Wald (SE uses p̂)
        se_wald = math.sqrt(p_hat * (1 - p_hat) / n) if 0 < p_hat < 1 else float("inf")
        z_wald = (
            (p_hat - p0) / se_wald
            if se_wald > 0 and math.isfinite(se_wald)
            else float("inf") * np.sign(p_hat - p0)
        )
        p_wald = float(norm.sf(abs(z_wald)) * 2)

        # Score (SE uses p0)
        se_score = math.sqrt(p0 * (1 - p0) / n) if 0 < p0 < 1 else float("inf")
        z_score = (
            (p_hat - p0) / se_score
            if se_score > 0 and math.isfinite(se_score)
            else float("inf") * np.sign(p_hat - p0)
        )
        p_score = float(norm.sf(abs(z_score)) * 2)

        # Continuity-corrected versions (±0.5 on counts)
        x = p_hat * n
        corr = 0.5
        # Wald corrected
        num_wald_c = (x - (n * p0 + corr)) / n
        se_wald_c = se_wald
        z_wald_c = (
            num_wald_c / se_wald_c
            if se_wald_c > 0 and math.isfinite(se_wald_c)
            else float("inf") * np.sign(num_wald_c)
        )
        p_wald_c = float(norm.sf(abs(z_wald_c)) * 2)
        # Score corrected
        num_score_c = (x - (n * p0 + corr)) / n
        se_score_c = se_score
        z_score_c = (
            num_score_c / se_score_c
            if se_score_c > 0 and math.isfinite(se_score_c)
            else float("inf") * np.sign(num_score_c)
        )
        p_score_c = float(norm.sf(abs(z_score_c)) * 2)

        # --- effect sizes
        # Cohen's g = |p̂ - p0|
        g_val = abs(p_hat - p0)
        g = es.CohenG(g_val, 0.0, 0.0, 0.0)

        # Cohen's h = 2*arcsin(sqrt(p̂)) - 2*arcsin(sqrt(p0))
        phi_s = 2 * math.asin(math.sqrt(p_hat))
        phi_0 = 2 * math.asin(math.sqrt(p0))
        h_val = phi_s - phi_0
        # SE for arcsine ≈ 2*sqrt(0.25*(1/n)) = 1/sqrt(n)
        se_h = 1.0 / math.sqrt(n) if n > 0 else float("inf")
        h_ci = (h_val - zcrit * se_h, h_val + zcrit * se_h)
        h = es.CohenH(h_val, h_ci[0], h_ci[1], se_h)

        # --- proportion CIs (12 methods)
        # 1) Agresti–Coull
        ac_lo, ac_hi = proportion_confint(
            int(round(x)), n, alpha=(1 - df_cl), method="agresti_coull"
        )

        # 2) Wald
        wald_lo, wald_hi = proportion_confint(
            int(round(x)), n, alpha=(1 - df_cl), method="normal"
        )

        # 3) Wald corrected (simple ± 0.05/n shift as in your original)
        shift = 0.05 / n
        wald_c_lo, wald_c_hi = max(wald_lo - shift, 0.0), min(wald_hi + shift, 1.0)

        # 4) Wilson
        wilson_lo, wilson_hi = proportion_confint(
            int(round(x)), n, alpha=(1 - df_cl), method="wilson"
        )

        # 5) Wilson corrected (per your original formula)
        z2 = zcrit * zcrit
        lower_wilson_corr = (
            2 * x
            + z2
            - 1
            - zcrit * math.sqrt(z2 - 2 - 1 / n + 4 * (x / n) * (n * (1 - x / n) + 1))
        ) / (2 * (n + z2))
        upper_wilson_corr = (
            2 * x
            + z2
            + 1
            + zcrit * math.sqrt(z2 + 2 - 1 / n + 4 * (x / n) * (n * (1 - x / n) - 1))
        ) / (2 * (n + z2))
        lower_wilson_corr = max(0.0, float(lower_wilson_corr))
        upper_wilson_corr = min(1.0, float(upper_wilson_corr))

        # 6) Logit (guard for x in {0,n})
        if 0 < x < n:
            lam = math.log(x / (n - x))
            term1 = n / (x * (n - x))
            lam_lo = lam - zcrit * math.sqrt(term1)
            lam_hi = lam + zcrit * math.sqrt(term1)
            logit_lo = math.exp(lam_lo) / (1 + math.exp(lam_lo))
            logit_hi = math.exp(lam_hi) / (1 + math.exp(lam_hi))
        else:
            # add 0.5 continuity to avoid ±∞
            lam = math.log((x + 0.5) / (n - x + 0.5))
            term1 = n / ((x + 0.5) * (n - x + 0.5))
            lam_lo = lam - zcrit * math.sqrt(term1)
            lam_hi = lam + zcrit * math.sqrt(term1)
            logit_lo = math.exp(lam_lo) / (1 + math.exp(lam_lo))
            logit_hi = math.exp(lam_hi) / (1 + math.exp(lam_hi))
        logit_lo, logit_hi = float(logit_lo), float(min(logit_hi, 1.0))

        # 7) Jeffreys
        jeff_lo = float(beta.ppf((1 - df_cl) / 2, x + 0.5, n - x + 0.5))
        jeff_hi = float(min(beta.ppf(1 - (1 - df_cl) / 2, x + 0.5, n - x + 0.5), 1.0))

        # 8) Clopper–Pearson
        cp_lo = float(beta.ppf((1 - df_cl) / 2, max(x, 0), n - x + 1))
        cp_hi = float(min(beta.ppf(1 - (1 - df_cl) / 2, x + 1, max(n - x, 0)), 1.0))

        # 9) Arcsine (Kulynskaya)
        p_tilde = (x + 0.375) / (n + 0.75)
        arcsine_lo = float(
            math.sin(math.asin(math.sqrt(p_tilde)) - 0.5 * zcrit / math.sqrt(n)) ** 2
        )
        arcsine_hi = float(
            min(
                math.sin(math.asin(math.sqrt(p_tilde)) + 0.5 * zcrit / math.sqrt(n))
                ** 2,
                1.0,
            )
        )

        # 10) Pratt (follow your original algebra)
        a = ((x + 1) / (n - x)) ** 2 if n - x > 0 else float("inf")
        b = 81 * (x + 1) * (n - x) - 9 * n - 8
        c = -3 * zcrit * math.sqrt(9 * (x + 1) * (n - x) * (9 * n + 5 - z2) + n + 1)
        d = 81 * (x + 1) ** 2 - 9 * (x + 1) * (2 + z2) + 1
        e = 1 + a * ((b + c) / d) ** 3 if math.isfinite(a) and d != 0 else float("inf")

        a2 = (x / (n - x - 1)) ** 2 if n - x - 1 > 0 else float("inf")
        b2 = 81 * x * (n - x - 1) - 9 * n - 8
        c2 = 3 * zcrit * math.sqrt(9 * x * (n - x - 1) * (9 * n + 5 - z2) + n + 1)
        d2 = 81 * x**2 - 9 * x * (2 + z2) + 1
        e2 = (
            1 + a2 * ((b2 + c2) / d2) ** 3
            if math.isfinite(a2) and d2 != 0
            else float("inf")
        )

        pratt_lo = float(max(1 / e2 if e2 not in (0.0, float("inf")) else 0.0, 0.0))
        pratt_hi = float(min(1 / e if e not in (0.0, float("inf")) else 1.0, 1.0))

        # 11) Blaker
        bl_lo, bl_hi = blakers_ci(int(round(x)), n, df_cl)

        # 12) Mid-p
        midp_lo, midp_hi = calculate_midp(int(round(x)), n, df_cl)

        # --- assemble results object
        res_obj = OneSampleProportionResults()
        res_obj.sample_size = n
        res_obj.sample_proportion = p_hat
        res_obj.population_proportion = p0
        res_obj.confidence_level = df_cl

        res_obj.descriptive_statistics = res.DescriptiveStatistics(
            mean=p_hat, standard_deviation=math.sqrt(p_hat * (1 - p_hat))
        )
        res_obj.z_test_wald = res.InferentialStatistics(p_value=p_wald, score=z_wald)
        res_obj.z_test = res.InferentialStatistics(p_value=p_score, score=z_score)
        res_obj.z_test_wald_corrected = res.InferentialStatistics(
            p_value=p_wald_c, score=z_wald_c
        )
        res_obj.z_test_corrected = res.InferentialStatistics(
            p_value=p_score_c, score=z_score_c
        )

        res_obj.cohens_g = g
        res_obj.cohens_h = h

        res_obj.agresti_coull_confidence_interval = res.ConfidenceInterval(ac_lo, ac_hi)
        res_obj.wald_type_confidence_interval = res.ConfidenceInterval(wald_lo, wald_hi)
        res_obj.wald_type_corrected_confidence_interval = res.ConfidenceInterval(
            wald_c_lo, wald_c_hi
        )
        res_obj.wilson_confidence_interval = res.ConfidenceInterval(
            wilson_lo, wilson_hi
        )
        res_obj.wilson_corrected_confidence_interval = res.ConfidenceInterval(
            lower_wilson_corr, upper_wilson_corr
        )
        res_obj.logit_confidence_interval = res.ConfidenceInterval(logit_lo, logit_hi)
        res_obj.jeffreys_confidence_interval = res.ConfidenceInterval(jeff_lo, jeff_hi)
        res_obj.clopper_pearson_confidence_interval = res.ConfidenceInterval(
            cp_lo, cp_hi
        )
        res_obj.arcsine_confidence_interval = res.ConfidenceInterval(
            arcsine_lo, arcsine_hi
        )
        res_obj.pratt_confidence_interval = res.ConfidenceInterval(pratt_lo, pratt_hi)
        res_obj.blaker_confidence_interval = res.ConfidenceInterval(bl_lo, bl_hi)
        res_obj.midp_confidence_interval = res.ConfidenceInterval(midp_lo, midp_hi)

        return res_obj

    @staticmethod
    def from_data(
        columns: list,
        population_proportion: float,
        defined_sucess_value: float,
        confidence_level: float = 0.95,
    ) -> OneSampleProportionResults:
        """Calculate results from data."""

        col = columns[0]
        n = len(col)
        x = int(np.count_nonzero(col == defined_sucess_value))
        p_hat = x / n
        return OneSampleProportions.from_parameters(
            p_hat, n, population_proportion, confidence_level
        )

    @staticmethod
    def from_frequency(
        population_proportion: float,
        number_of_successes: int,
        sample_size: int,
        confidence_level: float = 0.95,
    ) -> OneSampleProportionResults:
        """Calculate results from frequency data."""

        p_hat = number_of_successes / sample_size
        return OneSampleProportions.from_parameters(
            p_hat, sample_size, population_proportion, confidence_level
        )

    @staticmethod
    def proportion_of_hits(
        number_correct_answers: int,
        number_of_trials: int,
        number_of_choices: int,
        confidence_level: float = 0.95,
    ) -> OneSampleProportionResults:
        """
        Compute the 'proportion of hits' (π) for an mAFC task and return a
        OneSampleProportionResults object with a z-test and a Wald-style CI.

        Null hypothesis is π = 0.5 (chance level in 2AFC-equivalent units).
        """

        n = int(number_of_trials)
        k = int(number_of_choices)
        correct = int(number_correct_answers)

        # observed % correct
        p_correct = correct / n if n > 0 else 0.0

        # transform to π (proportion of hits) for mAFC
        pi = (p_correct * (k - 1)) / (1 + p_correct * (k - 2))

        # SE for π (delta method form you used)
        # guard against p_correct == 0 or 1
        denom = (
            math.sqrt(p_correct * (1 - p_correct))
            if 0 < p_correct < 1
            else float("inf")
        )
        se_pi = (
            (1 / math.sqrt(n)) * ((pi * (1 - pi)) / denom)
            if math.isfinite(denom) and n > 0
            else float("inf")
        )

        # z-test against chance π0 = 0.5
        pi0 = 0.5
        z = (
            (pi - pi0) / se_pi
            if se_pi > 0 and math.isfinite(se_pi)
            else float("inf") * np.sign(pi - pi0)
        )
        p_two_sided = float(norm.sf(abs(z)) * 2)

        # Wald-style CI for π
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
        lower_ci = pi - zcrit * se_pi
        upper_ci = pi + zcrit * se_pi

        # assemble results
        out = OneSampleProportionResults()
        out.sample_size = n
        out.sample_proportion = pi  # store π as the "proportion" we estimate
        out.population_proportion = pi0  # chance level for the z-test
        out.confidence_level = confidence_level

        # Descriptives for π
        out.descriptive_statistics = res.DescriptiveStatistics(
            mean=float(pi), standard_deviation=float("nan")
        )

        # Inference for π
        zstat = res.InferentialStatistics(p_value=p_two_sided, score=float(z))
        zstat.standard_error = float(se_pi)
        zstat.degrees_of_freedom = None
        zstat.means_difference = float(pi - pi0)
        out.z_test = zstat

        # Put the CI into a standard slot (Wald-style)
        out.wald_type_confidence_interval = res.ConfidenceInterval(
            lower=float(lower_ci), upper=float(upper_ci)
        )

        return out
