# ===== Standard library =====
from dataclasses import dataclass
from typing import Optional, Tuple
import math

# ===== Third-party =====
import numpy as np
from scipy.stats import norm, beta, binom, chi2_contingency, fisher_exact
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep
from scipy.optimize import root_scalar

# Optional backends (guarded)
try:
    from scipy.stats import barnard_exact  # SciPy >= 1.7

    _HAS_BARNARD = True
except Exception:
    _HAS_BARNARD = False

try:
    import rpy2.robjects as robjects

    _HAS_RPY2 = True
except Exception:
    _HAS_RPY2 = False

# ===== Internal (ESEK utils) =====
from ...utils import res, es, interfaces


@dataclass
class TwoIndependentProportionsResults:
    # Echoes
    n1: Optional[int] = None
    n2: Optional[int] = None
    x1: Optional[int] = None
    x2: Optional[int] = None
    p1: Optional[float] = None
    p2: Optional[float] = None
    diff: Optional[float] = None
    confidence_level: Optional[float] = None
    null_diff: float = 0.0

    # Tests on difference p1 - p2
    z_wald: Optional[res.InferentialStatistics] = None
    z_wald_pooled: Optional[res.InferentialStatistics] = None
    z_hauck_anderson: Optional[res.InferentialStatistics] = None
    z_mantel_haenszel: Optional[res.InferentialStatistics] = None
    exact_barnard_p: Optional[float] = None
    exact_fisher_p: Optional[float] = None

    # CIs for difference of proportions
    ci_diff_wald: Optional[res.ConfidenceInterval] = None
    ci_diff_wald_corrected: Optional[res.ConfidenceInterval] = None
    ci_diff_haldane: Optional[res.ConfidenceInterval] = None
    ci_diff_jeffreys_perks: Optional[res.ConfidenceInterval] = None
    ci_diff_mn: Optional[res.ConfidenceInterval] = None
    ci_diff_mee: Optional[res.ConfidenceInterval] = None
    ci_diff_agresti_caffo: Optional[res.ConfidenceInterval] = None
    ci_diff_wilson: Optional[res.ConfidenceInterval] = None
    ci_diff_wilson_corrected: Optional[res.ConfidenceInterval] = None
    ci_diff_hauck_anderson: Optional[res.ConfidenceInterval] = None
    ci_diff_blj: Optional[res.ConfidenceInterval] = None
    ci_diff_gart_nam: Optional[res.ConfidenceInterval] = None
    ci_diff_newcomb: Optional[res.ConfidenceInterval] = None

    # Odds ratio + CIs
    or_point: Optional[float] = None
    ci_or_woolf: Optional[res.ConfidenceInterval] = None
    ci_or_woolf_adj: Optional[res.ConfidenceInterval] = None
    ci_or_fisher: Optional[res.ConfidenceInterval] = None
    ci_or_midp: Optional[res.ConfidenceInterval] = None
    ci_or_mn: Optional[res.ConfidenceInterval] = None
    ci_or_bp: Optional[res.ConfidenceInterval] = None
    ci_or_sinh: Optional[res.ConfidenceInterval] = None
    ci_or_logit_adj: Optional[res.ConfidenceInterval] = None
    ci_or_fm: Optional[res.ConfidenceInterval] = None

    # Relative risk + CIs
    rr_point: Optional[float] = None
    rr_adj_walters: Optional[float] = None
    rr_unbiased_jewell: Optional[float] = None
    ci_rr_katz: Optional[res.ConfidenceInterval] = None
    ci_rr_walters: Optional[res.ConfidenceInterval] = None
    ci_rr_jewell: Optional[res.ConfidenceInterval] = None
    ci_rr_koopman: Optional[res.ConfidenceInterval] = None
    ci_rr_sinh: Optional[res.ConfidenceInterval] = None
    ci_rr_sinh_adj: Optional[res.ConfidenceInterval] = None
    ci_rr_zou_donner: Optional[res.ConfidenceInterval] = None

    # Effect sizes
    cohens_h: Optional[es.CohenH] = None
    cohens_w: Optional["es.CohenW"] | Optional[float] = (
        None  # depends on whether you add CohenW
    )
    probit_d: Optional[float] = None
    logit_d: Optional[float] = None


def _safe_div(num: float, den: float, default: float = float("inf")) -> float:
    try:
        return num / den
    except Exception:
        return default


def _wald_se_diff(p1: float, n1: int, p2: float, n2: int) -> float:
    return math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)


def _pooled_p(x1: int, n1: int, x2: int, n2: int) -> float:
    return (x1 + x2) / (n1 + n2)


def _counts_from_props(p1: float, n1: int, p2: float, n2: int) -> Tuple[int, int]:
    x1 = int(round(p1 * n1))
    x2 = int(round(p2 * n2))
    return x1, x2


def _ci_from_strict_bounds(
    lower: float, upper: float, lo: float, hi: float
) -> res.ConfidenceInterval:
    return res.ConfidenceInterval(max(lower, lo), min(upper, hi))


# Miettinen–Nurminen / MEE support (as in your code, compacted)
def _mn_mee_standard_errors(
    p1: float, n1: int, p2: float, n2: int, d_pop: float
) -> Tuple[float, float]:
    r = n2 / n1
    a = 1 + r
    b = -(1 + r + p1 + r * p2 + d_pop * (r + 2))
    c = d_pop * d_pop + d_pop * (2 * p1 + r + 1) + p1 + r * p2
    d = -p1 * d_pop * (1 + d_pop)

    v = (b / a / 3) ** 3 - b * c / (6 * a * a) + d / a / 2
    v = 0.0 if abs(v) < np.finfo(float).eps else v
    s = math.sqrt((b / a / 3) ** 2 - c / a / 3)
    u = (1 if v > 0 else -1) * s
    w = (math.pi + math.acos(max(-1.0, min(1.0, _safe_div(v, u**3, 0.0))))) / 3

    p1_hat = 2 * u * math.cos(w) - b / a / 3
    p2_hat = p1_hat - d_pop

    n = n1 + n2
    var_mn = (p1_hat * (1 - p1_hat) / n1 + p2_hat * (1 - p2_hat) / n2) * (n / (n - 1))
    se_mn = math.sqrt(max(var_mn, 0.0))

    var_mee = p1_hat * (1 - p1_hat) / n1 + p2_hat * (1 - p2_hat) / n2
    se_mee = math.sqrt(max(var_mee, 0.0))

    return se_mn, se_mee


def _ci_mn_mee(
    p1: float, n1: int, p2: float, n2: int, conf: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    # roots where p-value = alpha (two-sided)
    alpha = 1 - conf
    diff_obs = p1 - p2

    def _pvals(d_pop: float) -> Tuple[float, float]:
        se_mn, se_mee = _mn_mee_standard_errors(p1, n1, p2, n2, d_pop)
        z_mn = _safe_div(diff_obs - d_pop, se_mn, float("inf"))
        z_mee = _safe_div(diff_obs - d_pop, se_mee, float("inf"))
        p_mn = 2 * min(norm.cdf(z_mn), 1 - norm.cdf(z_mn))
        p_mee = 2 * min(norm.cdf(z_mee), 1 - norm.cdf(z_mee))
        return p_mn - alpha, p_mee - alpha

    # Brackets within [-1, 1]
    lo_br = (-1.0, diff_obs)
    hi_br = (diff_obs, 0.999999)

    mn_lo = root_scalar(lambda d0: _pvals(d0)[0], bracket=lo_br).root
    mn_hi = root_scalar(lambda d0: _pvals(d0)[0], bracket=hi_br).root
    mee_lo = root_scalar(lambda d0: _pvals(d0)[1], bracket=lo_br).root
    mee_hi = root_scalar(lambda d0: _pvals(d0)[1], bracket=hi_br).root

    return (max(-1.0, mn_lo), min(1.0, mn_hi)), (max(-1.0, mee_lo), min(1.0, mee_hi))


class TwoIndependentProportions(interfaces.AbstractTest):
    @staticmethod
    def from_parameters(
        proportion_sample_1: float,
        proportion_sample_2: float,
        sample_size_1: int,
        sample_size_2: int,
        confidence_level: float = 0.95,
        difference_in_population: float = 0.0,
    ) -> TwoIndependentProportionsResults:

        # Normalize inputs
        p1 = float(np.clip(proportion_sample_1, 0.0, 1.0))
        p2 = float(np.clip(proportion_sample_2, 0.0, 1.0))
        n1 = int(sample_size_1)
        n2 = int(sample_size_2)
        x1, x2 = _counts_from_props(p1, n1, p2, n2)
        diff = p1 - p2
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)

        out = TwoIndependentProportionsResults(
            n1=n1,
            n2=n2,
            x1=x1,
            x2=x2,
            p1=p1,
            p2=p2,
            diff=diff,
            confidence_level=confidence_level,
            null_diff=difference_in_population,
        )

        # ---------- Tests (z) ----------
        # Wald (unpooled SE)
        se_wald = _wald_se_diff(p1, n1, p2, n2)
        z_wald = _safe_div(diff - difference_in_population, se_wald, float("inf"))
        p_wald = float(norm.sf(abs(z_wald)) * 2)
        zobj = res.InferentialStatistics(p_value=p_wald, score=float(z_wald))
        zobj.standard_error = float(se_wald)
        out.z_wald = zobj

        # Wald pooled (H0)
        p_pool = _pooled_p(x1, n1, x2, n2)
        se_wald_H0 = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        z_wald_H0 = _safe_div(diff - difference_in_population, se_wald_H0, float("inf"))
        p_wald_H0 = float(norm.sf(abs(z_wald_H0)) * 2)
        zpooled = res.InferentialStatistics(p_value=p_wald_H0, score=float(z_wald_H0))
        zpooled.standard_error = float(se_wald_H0)
        out.z_wald_pooled = zpooled

        # Hauck–Anderson
        corr_HA = 1 / (2 * min(n1, n2))
        se1_HA = p1 * (1 - p1) / (n1 - 1) if n1 > 1 else float("inf")
        se2_HA = p2 * (1 - p2) / (n2 - 1) if n2 > 1 else float("inf")
        se_HA = math.sqrt(max(se1_HA + se2_HA, 0.0))
        z_HA = _safe_div(diff - difference_in_population - corr_HA, se_HA, float("inf"))
        p_HA = float(norm.sf(abs(z_HA)) * 2)
        zha = res.InferentialStatistics(p_value=p_HA, score=float(z_HA))
        zha.standard_error = float(se_HA)
        out.z_hauck_anderson = zha

        # Mantel–Haenszel (single stratum, uses 2x2 variance formula)
        # Using your formula:
        N = n1 + n2
        q1 = n1 - x1
        q2 = n2 - x2
        se_MH = (
            math.sqrt((n1 * n2 * (x1 + x2) * (q1 + q2)) / (N**2 * (N - 1)))
            if N > 1
            else float("inf")
        )
        mean_MH = n1 * (x1 + x2) / N if N > 0 else float("nan")
        z_MH = _safe_div(x1 - mean_MH, se_MH, float("inf"))
        p_MH = float(norm.sf(abs(z_MH)) * 2)
        zmh = res.InferentialStatistics(p_value=p_MH, score=float(z_MH))
        zmh.standard_error = float(se_MH)
        out.z_mantel_haenszel = zmh

        # Barnard & Fisher (optional)
        if _HAS_BARNARD:
            try:
                stat_b = barnard_exact([[x1, x2], [q1, q2]])
                out.exact_barnard_p = float(stat_b.pvalue)
            except Exception:
                out.exact_barnard_p = None
        try:
            out.exact_fisher_p = float(fisher_exact([[x1, x2], [q1, q2]])[1])
        except Exception:
            out.exact_fisher_p = None

        # ---------- Effect sizes ----------
        # Cohen's h
        phi1 = 2 * math.asin(math.sqrt(p1))
        phi2 = 2 * math.asin(math.sqrt(p2))
        h_val = abs(phi1 - phi2)
        # SE(h) approx from arcsine variance: ≈ 1/sqrt(n) per prop; for difference it's not standard—omit SE or use NaN:
        out.cohens_h = es.CohenH(
            value=h_val, ci_lower=0.0, ci_upper=0.0, standard_error=float("nan")
        )

        # Cohen's w (sqrt(chi2/N))
        chi2 = chi2_contingency(
            np.array([[x1, x2], [q1, q2]]), correction=False
        ).statistic
        w_val = math.sqrt(chi2 / (n1 + n2)) if (n1 + n2) > 0 else float("nan")
        try:
            out.cohens_w = es.CohenW(
                w_val, 0.0, 0.0, float("nan")
            )  # if you added CohenW
        except Exception:
            out.cohens_w = w_val

        # probit/logit d
        def _clip01(val: float, eps: float = 1e-12) -> float:
            return min(max(val, eps), 1 - eps)

        p1c = _clip01(p1)
        p2c = _clip01(p2)
        out.probit_d = float(norm.ppf(p1c) - norm.ppf(p2c))
        out.logit_d = float(math.log(p1c / (1 - p1c)) - math.log(p2c / (1 - p2c)))

        # ---------- CI for difference of proportions ----------
        # Wald
        lo_wald = diff - zcrit * se_wald
        hi_wald = diff + zcrit * se_wald
        out.ci_diff_wald = res.ConfidenceInterval(lo_wald, hi_wald)

        # Wald corrected (simple continuity)
        cc = 0.5 * (1 / n1 + 1 / n2)
        out.ci_diff_wald_corrected = res.ConfidenceInterval(
            diff - (cc + zcrit * se_wald), diff + (cc + zcrit * se_wald)
        )

        # Haldane
        psi = 0.5 * (p1 + p2)
        v = (1 / n1 - 1 / n2) / 4
        mu = (1 / n1 + 1 / n2) / 4
        theta = (diff + zcrit * zcrit * v * (1 - 2 * psi)) / (1 + zcrit * zcrit * mu)
        w_haldane = (zcrit / (1 + zcrit * zcrit * mu)) * math.sqrt(
            mu * (4 * psi * (1 - psi) - diff * diff)
            + 2 * v * (1 - 2 * psi) * diff
            + 4 * zcrit * zcrit * mu * mu * (1 - psi) * psi
            + zcrit * zcrit * v * v * (1 - 2 * psi) ** 2
        )
        out.ci_diff_haldane = res.ConfidenceInterval(
            theta - w_haldane, theta + w_haldane
        )

        # Jeffreys–Perks
        psi_jp = 0.5 * (((x1 + 0.5) / (n1 + 1)) + ((x2 + 0.5) / (n2 + 1)))
        theta_jp = (diff + zcrit * zcrit * v * (1 - 2 * psi_jp)) / (
            1 + zcrit * zcrit * mu
        )
        w_jp = (zcrit / (1 + zcrit * zcrit * mu)) * math.sqrt(
            mu * (4 * psi_jp * (1 - psi_jp) - diff * diff)
            + 2 * v * (1 - 2 * psi_jp) * diff
            + 4 * zcrit * zcrit * mu * mu * (1 - psi_jp) * psi_jp
            + zcrit * zcrit * v * v * (1 - 2 * psi_jp) ** 2
        )
        out.ci_diff_jeffreys_perks = res.ConfidenceInterval(
            theta_jp - w_jp, theta_jp + w_jp
        )

        # MN & MEE (score)
        try:
            (mn_lo, mn_hi), (mee_lo, mee_hi) = _ci_mn_mee(
                p1, n1, p2, n2, confidence_level
            )
            out.ci_diff_mn = res.ConfidenceInterval(mn_lo, mn_hi)
            out.ci_diff_mee = res.ConfidenceInterval(mee_lo, mee_hi)
        except Exception:
            out.ci_diff_mn = None
            out.ci_diff_mee = None

        # Agresti–Caffo (+1 per group)
        p1_ac = (x1 + 1) / (n1 + 2)
        p2_ac = (x2 + 1) / (n2 + 2)
        se_ac = math.sqrt(
            p1_ac * (1 - p1_ac) / (n1 + 2) + p2_ac * (1 - p2_ac) / (n2 + 2)
        )
        out.ci_diff_agresti_caffo = res.ConfidenceInterval(
            (p1_ac - p2_ac) - zcrit * se_ac, (p1_ac - p2_ac) + zcrit * se_ac
        )

        # Wilson-based diff CI (Newcombe’s approach mixing Wilson CIs)
        w1_lo, w1_hi = proportion_confint(
            x1, n1, alpha=(1 - confidence_level), method="wilson"
        )
        w2_lo, w2_hi = proportion_confint(
            x2, n2, alpha=(1 - confidence_level), method="wilson"
        )
        lo_wil = diff - zcrit * math.sqrt(
            max(w1_lo * (1 - w1_lo) / n1, 0) + max(w2_hi * (1 - w2_hi) / n2, 0)
        )
        hi_wil = diff + zcrit * math.sqrt(
            max(w1_hi * (1 - w1_hi) / n1, 0) + max(w2_lo * (1 - w2_lo) / n2, 0)
        )
        out.ci_diff_wilson = res.ConfidenceInterval(lo_wil, hi_wil)

        # Wilson corrected (Kulinskaya-like)
        z2 = zcrit * zcrit

        def _wilson_corr_bounds(x, n, p):
            lo = (
                2 * x
                + z2
                - 1
                - zcrit * math.sqrt(z2 - 2 - 1 / n + 4 * p * (n * (1 - p) + 1))
            ) / (2 * (n + z2))
            hi = (
                2 * x
                + z2
                + 1
                + zcrit * math.sqrt(z2 + 2 - 1 / n + 4 * p * (n * (1 - p) - 1))
            ) / (2 * (n + z2))
            return max(lo, 0.0), min(hi, 1.0)

        w1c_lo, w1c_hi = _wilson_corr_bounds(x1, n1, p1)
        w2c_lo, w2c_hi = _wilson_corr_bounds(x2, n2, p2)
        lo_wil_c = max(-1.0, diff - math.sqrt((p1 - w1c_lo) ** 2 + (w2c_hi - p2) ** 2))
        hi_wil_c = min(1.0, diff + math.sqrt((w1c_hi - p1) ** 2 + (p2 - w2c_lo) ** 2))
        out.ci_diff_wilson_corrected = res.ConfidenceInterval(lo_wil_c, hi_wil_c)

        # Hauck–Anderson CI
        out.ci_diff_hauck_anderson = res.ConfidenceInterval(
            max(-1.0, diff - corr_HA - zcrit * se_HA),
            min(1.0, diff + corr_HA + zcrit * se_HA),
        )

        # Brown–Li (Jeffreys) CI
        p1_blj = (x1 + 0.5) / (n1 + 1)
        p2_blj = (x2 + 0.5) / (n2 + 1)
        se_blj = math.sqrt(p1_blj * (1 - p1_blj) / n1 + p2_blj * (1 - p2_blj) / n2)
        out.ci_diff_blj = res.ConfidenceInterval(
            (p1_blj - p2_blj) - zcrit * se_blj, (p1_blj - p2_blj) + zcrit * se_blj
        )

        # Gart–Nam (R-backed)
        if _HAS_RPY2:
            robjects.r(
                """scoretheta <- function(x1, n1, x2, n2, theta, level = 0.95) {
                Prop_Diff <- ((x1 / n1) - (x2 / n2)) - theta
                N <- n1 + n2
                a <- (n1 + 2 * n2) * theta - N - (x1 + x2)
                b <- (a / N / 3)^3 - a * ((n2 * theta - N - 2 * x2) * theta + (x1 + x2)) / (6 * N * N) + (x2 * theta * (1 - theta)) / N / 2
                c <- ifelse(b > 0, 1, -1) * sqrt(pmax(0, (a / N / 3)^2 - ((n2 * theta - N - 2 * x2) * theta + (x1 + x2)) / N / 3))
                p2d <- pmin(1, pmax(0, round(2 * c * cos(((pi + acos(pmax(-1, pmin(1, ifelse(c == 0 & b == 0, 0, b / c^3))))) / 3)) - a / N / 3, 10)))
                p1d <- pmin(1, pmax(0, p2d + theta))
                Variance <- pmax(0, (p1d * (1 - p1d) / n1 + p2d * (1 - p2d) / n2))
                scterm <- (p1d * (1 - p1d) * (1 - 2 * p1d) / (n1^2) - p2d * (1 - p2d) * (1 - 2 * p2d) / (n2^2)) / (6 * Variance^(3 / 2))
                score <- ifelse(scterm == 0, (Prop_Diff / sqrt(Variance)), (-1 + sqrt(pmax(0, 1 - 4 * scterm * -(Prop_Diff / sqrt(Variance) + scterm))) ) / (2 * scterm))
                return(score)}
                Binary_Search <- function(score_function, max.iter = 100, tail = "lower") {
                    hi <- 1; lo <- -1; niter <- 1
                    while (niter <= max.iter) {
                        mid <- max(-1, min(1, round((hi + lo) / 2, 10)))
                        scor <- score_function(mid)
                        check <- (scor <= 0) | is.na(scor)
                        hi <- ifelse(check, mid, hi)
                        lo <- ifelse(check, lo, mid)
                        niter <- niter + 1}
                    if (tail == "lower") lo else hi}
                gart_nam <- function(x1, n1, x2, n2, level = 0.95) {
                    zcrit <- qnorm(1 - (1 - level)/2)
                    lower <- Binary_Search(function(theta) scoretheta(x1,n1,x2,n2,theta) - zcrit, tail = "lower")
                    upper <- Binary_Search(function(theta) scoretheta(x1,n1,x2,n2,theta) + zcrit, tail = "upper")
                    c(lower, upper)}"""
            )
            ci_gn = robjects.r["gart_nam"](x1, n1, x2, n2, confidence_level)  # type: ignore
            out.ci_diff_gart_nam = res.ConfidenceInterval(
                float(ci_gn[0]), float(ci_gn[1])
            )

        # Newcomb (statsmodels returns tuple)
        try:
            ci_new = confint_proportions_2indep(
                count1=x1,
                nobs1=n1,
                count2=x2,
                nobs2=n2,
                method="newcomb",
                alpha=1 - confidence_level,
            )
            out.ci_diff_newcomb = res.ConfidenceInterval(
                float(ci_new[0]), float(ci_new[1])
            )
        except Exception:
            out.ci_diff_newcomb = None

        # ---------- Odds ratio ----------
        # Wald OR (unconditional MLE)
        or_point = _safe_div(p1 * (1 - p2), p2 * (1 - p1), float("inf"))
        out.or_point = or_point

        # Woolf (log OR ± z*SElog)
        def _se_log_or(a, b, c, d, adj: float = 0.0) -> float:
            A = a + adj
            B = b + adj
            C = c + adj
            D = d + adj
            return math.sqrt(
                _safe_div(1, A) + _safe_div(1, C) + _safe_div(1, B) + _safe_div(1, D)
            )

        se_log_or = _se_log_or(x1, q1, x2, q2, 0.0)
        lo_woolf = (
            math.exp(math.log(or_point) - zcrit * se_log_or)
            if math.isfinite(or_point)
            else 0.0
        )
        hi_woolf = (
            math.exp(math.log(or_point) + zcrit * se_log_or)
            if math.isfinite(or_point)
            else float("inf")
        )
        out.ci_or_woolf = res.ConfidenceInterval(lo_woolf, hi_woolf)

        # Woolf adjusted (+0.5)
        se_log_or_adj = _se_log_or(x1, q1, x2, q2, 0.5)
        lo_woolf_a = (
            math.exp(math.log(or_point) - zcrit * se_log_or_adj)
            if math.isfinite(or_point)
            else 0.0
        )
        hi_woolf_a = (
            math.exp(math.log(or_point) + zcrit * se_log_or_adj)
            if math.isfinite(or_point)
            else float("inf")
        )
        out.ci_or_woolf_adj = res.ConfidenceInterval(lo_woolf_a, hi_woolf_a)

        # Fisher/Cornfield OR CI
        try:
            from scipy.stats.contingency import odds_ratio as _odds_ratio

            or_ci = _odds_ratio([[x1, q1], [x2, q2]]).confidence_interval(
                confidence_level
            )
            out.ci_or_fisher = res.ConfidenceInterval(
                float(or_ci.low), float(or_ci.high)
            )
        except Exception:
            out.ci_or_fisher = None

        # Mid-p OR (R-backed)
        if _HAS_RPY2:
            robjects.r(
                """odds_ratio_mid_p_value <- function(p1,q1,p2,q2, conf.level=0.95, interval=c(0, 1000)) {
                mid_p_function <- function(or_val=1) {
                    less_p <- fisher.test(matrix(c(p1,q1,p2,q2),2,2), or=or_val, alternative="less")$p.value
                    greater_p <- fisher.test(matrix(c(p1,q1,p2,q2),2,2), or=or_val, alternative="greater")$p.value
                    0.5 * (less_p - greater_p + 1)}
                alpha <- 1 - conf.level
                lower <- uniroot(function(or_val) 1 - mid_p_function(or_val) - alpha/2, interval)$root
                upper <- 1 / uniroot(function(or_val) mid_p_function(1/or_val) - alpha/2, interval)$root
                c(lower, upper)}"""
            )
            ci_mp = robjects.r["odds_ratio_mid_p_value"](x1, q1, x2, q2)  # type: ignore
            out.ci_or_midp = res.ConfidenceInterval(float(ci_mp[0]), float(ci_mp[1]))

        # OR MN (score)
        try:
            ci_mn_or = confint_proportions_2indep(
                nobs1=n1,
                count1=x1,
                nobs2=n2,
                count2=x2,
                method="score",
                compare="odds-ratio",
                alpha=1 - confidence_level,
            )
            out.ci_or_mn = res.ConfidenceInterval(
                float(ci_mn_or[0]), float(ci_mn_or[1])
            )
        except Exception:
            out.ci_or_mn = None

        # Baptista–Pike (R-backed)
        if _HAS_RPY2:
            robjects.r(
                """Baptista_Pike <- function(p1,q1,p2,q2, conf.level=0.95, orRange=c(1e-10, 1e10)) {
                x <- matrix(c(p1,q1,p2,q2), 2, 2); alpha <- 1 - conf.level
                n1 <- sum(x[1,]); n2 <- sum(x[2,]); s <- sum(x[,1]); a <- x[1,1]
                support <- max(0, s - n2):min(n1, s)
                dnhyper <- function(OR) { d <- dhyper(support,n1,n2,s,log=TRUE) + log(OR)*support; exp(d - max(d))/sum(exp(d - max(d))) }
                pnhyper <- function(x, OR, lower.tail=TRUE) { f <- dnhyper(OR); X <- if (lower.tail) support <= x else support >= x; sum(f[X]) }
                ints <- function(xl, xu, ORRange=orRange) {
                    dnh <- function(beta) dnhyper(beta)
                    find <- function(beta) sum(dnh(beta)[support<=xl]) - sum(dnh(beta)[support>=xu])
                    uniroot(find, ORRange)$root}
                ints_greater <- ints(a, a+1)
                ints_less <- ints(a-1, a)
                lower <- uniroot(function(or) alpha - pnhyper(a, or, lower.tail=FALSE), c(orRange[1], ints_less))$root
                upper <- uniroot(function(or) alpha - pnhyper(a, or, lower.tail=TRUE), c(ints_greater, orRange[2]))$root
                c(lower, upper)}"""
            )
            ci_bp = robjects.r["Baptista_Pike"](x1, q1, x2, q2)  # type: ignore
            out.ci_or_bp = res.ConfidenceInterval(float(ci_bp[0]), float(ci_bp[1]))

        # Sinh/logit adjusted
        se_log_or_sinh = 2 * math.asinh(
            0.5
            * zcrit
            * math.sqrt(
                _safe_div(1, x1)
                + _safe_div(1, q1)
                + _safe_div(1, x2)
                + _safe_div(1, q2)
            )
        )
        lo_sinh = (
            math.exp(math.log(or_point) - se_log_or_sinh)
            if math.isfinite(or_point)
            else 0.0
        )
        hi_sinh = (
            math.exp(math.log(or_point) + se_log_or_sinh)
            if math.isfinite(or_point)
            else float("inf")
        )
        out.ci_or_sinh = res.ConfidenceInterval(lo_sinh, hi_sinh)

        # Independent smooth logit (Agresti)
        p1new = x1 + 2 * n1 * (x1 + x2) / (n1 + n2) ** 2
        q1new = q1 + 2 * n1 * (q1 + q2) / (n1 + n2) ** 2
        p2new = x2 + 2 * n2 * (x1 + x2) / (n1 + n2) ** 2
        q2new = q2 + 2 * n2 * (q1 + q2) / (n1 + n2) ** 2
        log_theta = math.log(_safe_div(p1new * q2new, p2new * q1new, float("inf")))
        ci_half = zcrit * math.sqrt(
            _safe_div(1, p1new)
            + _safe_div(1, q1new)
            + _safe_div(1, p2new)
            + _safe_div(1, q2new)
        )
        out.ci_or_logit_adj = res.ConfidenceInterval(
            math.exp(log_theta - ci_half), math.exp(log_theta + ci_half)
        )

        # Farrington–Manning (R-backed)
        if _HAS_RPY2:
            robjects.r(
                """FM_CI <- function(p1,q1,p2,q2, alpha=0.05) {
                score <- function(theta0, n11, n21, n1p, n2p) {
                    p2hat <- (-(n1p * theta0 + n2p - (n11 + n21) * (theta0 - 1)) + sqrt((n1p * theta0 + n2p - (n11 + n21) * (theta0 - 1))^2 - 4 * n2p * (theta0 - 1) * -(n11 + n21))) / (2 * n2p * (theta0 - 1))
                    p1hat <- p2hat * theta0 / (1 + p2hat * (theta0 - 1))
                    ((n1p * (n11/n1p - p1hat)) * sqrt(1/(n1p*p1hat*(1-p1hat)) + 1/(n2p*p2hat*(1-p2hat))))}
                lower <- function(theta0, n11, n21, n1p, n2p, alpha) score(theta0, n11, n21, n1p, n2p) - qnorm(1 - alpha/2)
                upper <- function(theta0, n11, n21, n1p, n2p, alpha) score(theta0, n11, n21, n1p, n2p) + qnorm(1 - alpha/2)
                L <- uniroot(lower, c(1e-6, 1e6), p1, q1, (p1+q1), (p2+q2), alpha)$root
                U <- uniroot(upper, c(1e-6, 1e6), p1, q1, (p1+q1), (p2+q2), alpha)$root
                c(L,U)}"""
            )
            ci_fm = robjects.r["FM_CI"](x1, q1, x2, q2, 1 - confidence_level)  # type: ignore
            out.ci_or_fm = res.ConfidenceInterval(float(ci_fm[0]), float(ci_fm[1]))

        # ---------- Relative risk ----------
        rr = _safe_div(p1, p2, float("inf"))
        out.rr_point = rr
        rr_walters = math.exp(
            math.log(_safe_div(x1 + 0.5, n1 + 0.5))
            - math.log(_safe_div(x2 + 0.5, n2 + 0.5))
        )
        out.rr_adj_walters = rr_walters
        rr_unbiased_j = _safe_div(
            x1 * (q1 + q2 + 1), (x1 + x2) * (q1 + 1), float("inf")
        )
        out.rr_unbiased_jewell = rr_unbiased_j

        se_rr_katz = math.sqrt(_safe_div(1 - p1, x1) + _safe_div(1 - p2, x2))
        out.ci_rr_katz = res.ConfidenceInterval(
            math.exp(math.log(rr) - zcrit * se_rr_katz),
            math.exp(math.log(rr) + zcrit * se_rr_katz),
        )

        se_rr_walters = math.sqrt(
            _safe_div(1, x1 + 0.5)
            - _safe_div(1, n1 + 0.5)
            + _safe_div(1, x2 + 0.5)
            - _safe_div(1, n2 + 0.5)
        )
        out.ci_rr_walters = res.ConfidenceInterval(
            math.exp(math.log(rr_walters) - zcrit * se_rr_walters),
            math.exp(math.log(rr_walters) + zcrit * se_rr_walters),
        )

        var_rr_j = (
            (rr_unbiased_j**2) * (_safe_div(q1, x1 * (x1 + q1)))
            + _safe_div(q2, (x2 + 1) * (x2 + q2))
            - (rr_unbiased_j**2)
            * (_safe_div(q1, x1 * (x1 + q1)) * _safe_div(q2, (x2 + 1) * (x2 + q2 + 1)))
        )
        se_rr_j = math.sqrt(max(var_rr_j, 0.0))
        out.ci_rr_jewell = res.ConfidenceInterval(
            math.exp(math.log(rr_walters) - zcrit * se_rr_j),
            math.exp(math.log(rr_walters) + zcrit * se_rr_j),
        )

        # Koopman
        def _koopman_ci(x1, n1, x2, n2, conf):
            z = abs(norm.ppf((1 - conf) / 2))
            a1 = n2 * (n2 * (n2 + n1) * x1 + n1 * (n2 + x1) * (z**2))
            a2 = -n2 * (
                n2 * n1 * (x2 + x1)
                + 2 * (n2 + n1) * x2 * x1
                + n1 * (n2 + x2 + 2 * x1) * (z**2)
            )
            a3 = (
                2 * n2 * n1 * x2 * (x2 + x1)
                + (n2 + n1) * (x2**2) * x1
                + n2 * n1 * (x2 + x1) * (z**2)
            )
            a4 = -n1 * (x2**2) * (x2 + x1)
            b1 = a2 / a1
            b2 = a3 / a1
            b3 = a4 / a1
            c1 = b2 - (b1**2) / 3
            c2 = b3 - b1 * b2 / 3 + 2 * (b1**3) / 27
            ceta = math.acos(
                max(-1.0, min(1.0, math.sqrt(27) * c2 / (2 * c1 * math.sqrt(-c1))))
            )
            t1 = -2 * math.sqrt(-c1 / 3) * math.cos(math.pi / 3 - ceta / 3)
            t2 = -2 * math.sqrt(-c1 / 3) * math.cos(math.pi / 3 + ceta / 3)
            t3 = 2 * math.sqrt(-c1 / 3) * math.cos(ceta / 3)
            p01, p02, p03 = t1 - b1 / 3, t2 - b1 / 3, t3 - b1 / 3
            p0sum = p01 + p02 + p03
            p0up = min(p01, p02, p03)
            p0low = p0sum - p0up - max(p01, p02, p03)
            ul = (1 - (n1 - x1) * (1 - p0up) / (x2 + n1 - (n2 + n1) * p0up)) / p0up
            ll = (1 - (n1 - x1) * (1 - p0low) / (x2 + n1 - (n2 + n1) * p0low)) / p0low
            return res.ConfidenceInterval(ll, ul)

        try:
            out.ci_rr_koopman = _koopman_ci(x1, n1, x2, n2, confidence_level)
        except Exception:
            out.ci_rr_koopman = None

        # Inverse-sine RR
        eps_adj = math.asinh(
            0.5
            * zcrit
            * math.sqrt(
                _safe_div(1, x1)
                + _safe_div(1, x2)
                - _safe_div(1, n1 + 1)
                - _safe_div(1, n2 + 1)
            )
        )
        eps = math.asinh(
            0.5
            * zcrit
            * math.sqrt(
                _safe_div(1, x1)
                + _safe_div(1, x2)
                - _safe_div(1, n1)
                - _safe_div(1, n2)
            )
        )
        out.ci_rr_sinh_adj = res.ConfidenceInterval(
            math.exp(math.log(rr) - 2 * eps_adj), math.exp(math.log(rr) + 2 * eps_adj)
        )
        out.ci_rr_sinh = res.ConfidenceInterval(
            math.exp(math.log(rr) - 2 * eps), math.exp(math.log(rr) + 2 * eps)
        )

        # Zou–Donner
        wil1 = proportion_confint(x1, n1, alpha=(1 - confidence_level), method="wilson")
        wil2 = proportion_confint(x2, n2, alpha=(1 - confidence_level), method="wilson")
        lo_zd = math.exp(
            (math.log(_safe_div(x1, n1)))
            - (math.log(_safe_div(x2, n2)))
            - math.sqrt(
                (math.log(_safe_div(x1, n1)) - math.log(max(wil1[0], 1e-12))) ** 2
                + (math.log(min(wil2[1], 1 - 1e-12)) - math.log(_safe_div(x2, n2))) ** 2
            )
        )
        hi_zd = math.exp(
            (math.log(_safe_div(x1, n1)))
            - (math.log(_safe_div(x2, n2)))
            + math.sqrt(
                (math.log(min(wil1[1], 1 - 1e-12)) - math.log(_safe_div(x1, n1))) ** 2
                + (math.log(_safe_div(x2, n2)) - math.log(max(wil2[0], 1e-12))) ** 2
            )
        )
        out.ci_rr_zou_donner = res.ConfidenceInterval(lo_zd, hi_zd)

        return out

    @staticmethod
    def from_data(
        column_1: np.ndarray,
        column_2: np.ndarray,
        defined_success_value: float,
        confidence_level: float = 0.95,
        difference_in_population: float = 0.0,
    ) -> TwoIndependentProportionsResults:
        n1 = int(len(column_1))
        n2 = int(len(column_2))
        x1 = int(np.count_nonzero(column_1 == defined_success_value))
        x2 = int(np.count_nonzero(column_2 == defined_success_value))
        p1 = x1 / n1
        p2 = x2 / n2
        return TwoIndependentProportions.from_parameters(
            p1, p2, n1, n2, confidence_level, difference_in_population
        )

    @staticmethod
    def from_frequencies(
        number_of_successes_1: int,
        sample_size_1: int,
        number_of_successes_2: int,
        sample_size_2: int,
        confidence_level: float = 0.95,
        difference_in_population: float = 0.0,
    ) -> TwoIndependentProportionsResults:
        x1 = int(number_of_successes_1)
        x2 = int(number_of_successes_2)
        n1 = int(sample_size_1)
        n2 = int(sample_size_2)
        p1 = x1 / n1
        p2 = x2 / n2
        return TwoIndependentProportions.from_parameters(
            p1, p2, n1, n2, confidence_level, difference_in_population
        )
