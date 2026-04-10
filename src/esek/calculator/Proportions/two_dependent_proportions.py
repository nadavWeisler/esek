# src/esek/Calculator/TwoDependentProportions/dependent_proportions.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, beta
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

from ...utils import interfaces, res

@dataclass
class PairedDifferenceCIs:
    """
    CIs for difference in paired proportions (p1 - p2).
    """
    wald: Optional[res.ConfidenceInterval] = None
    wald_cc: Optional[res.ConfidenceInterval] = None  # Edwards/Fleiss CC
    wald_yates: Optional[res.ConfidenceInterval] = None
    agresti_min: Optional[res.ConfidenceInterval] = None
    bonett_price: Optional[res.ConfidenceInterval] = None
    newcomb: Optional[res.ConfidenceInterval] = None


@dataclass
class MatchedOddsRatioCIs:
    """
    CIs for matched-pairs odds ratio.
    """
    wald_log: Optional[res.ConfidenceInterval] = None
    wald_laplace: Optional[res.ConfidenceInterval] = None
    binomial: Optional[res.ConfidenceInterval] = (
        None  # Clopper–Pearson on p = b/(b+c) ⇒ transform
    )
    rigby_robinson: Optional[res.ConfidenceInterval] = None
    rigby_robinson_cc: Optional[res.ConfidenceInterval] = None
    wilson_score_transformed: Optional[res.ConfidenceInterval] = None
    fiducial: Optional[res.ConfidenceInterval] = None


@dataclass
class MatchedRiskRatioCIs:
    """
    CIs for matched-pairs risk ratio.
    """
    wald: Optional[res.ConfidenceInterval] = None
    tang: Optional[res.ConfidenceInterval] = None
    bonett_price: Optional[res.ConfidenceInterval] = None
    bonett_price_cc: Optional[res.ConfidenceInterval] = None
    mover_wilson: Optional[res.ConfidenceInterval] = None


@dataclass
class TwoDependentProportionsResults:
    """
    Results for paired/dependent proportions analysis.
    """
    # basic counts
    n: int
    yes_yes: int
    yes_no: int
    no_yes: int
    no_no: int

    # proportions
    p1: float  # proportion success in var1
    p2: float  # proportion success in var2
    diff_p: float  # p1 - p2
    prop_matching: float  # (a + d)/n
    prop_not_matching: float  # 1 - prop_matching

    # McNemar (inference)
    mcnemar_chi2: Optional[res.InferentialStatistics] = None
    mcnemar_chi2_cc: Optional[res.InferentialStatistics] = None
    mcnemar_exact_p: Optional[float] = None

    # difference CIs
    diff_cis: Optional[PairedDifferenceCIs] = None

    # matched-pairs odds ratio θ = b/c (b=yes_no, c=no_yes)
    matched_or: Optional[float] = None
    matched_or_se: Optional[float] = None
    matched_or_se_adj: Optional[float] = None
    matched_or_cis: Optional[MatchedOddsRatioCIs] = None

    # matched-pairs risk ratio RR = (a+b)/(a+c)
    matched_rr: Optional[float] = None
    matched_rr_cis: Optional[MatchedRiskRatioCIs] = None

    # other epidemiologic measures
    nnt: Optional[float] = None
    pop_attrib_risk: Optional[float] = None
    ir_exposed: Optional[float] = None
    ir_unexposed: Optional[float] = None
    ir_population: Optional[float] = None


class TwoDependentProportions(interfaces.AbstractTest):
    """
    Paired/dependent proportions (2x2 matched) with:
    - McNemar tests
    - CIs for p1 - p2
    - Matched-pairs OR and RR with multiple CI methods
    """

    @staticmethod
    def from_parameters(
        p1: float,
        p2: float,
        p_yes_yes: float,
        n: int,
        confidence_level: float = 95.0,  # kept for API symmetry; not used directly here
    ) -> TwoDependentProportionsResults:

        cl = confidence_level
        zcrit = norm.ppf(cl + (1 - cl) / 2)

        # counts
        a = p_yes_yes * n
        b = p1 * n - a
        c = p2 * n - a
        d = n - (a + b + c)

        # guard: tiny numerical drift
        a, b, c, d = float(a), float(b), float(c), float(d)
        a_i, b_i, c_i, d_i = int(round(a)), int(round(b)), int(round(c)), int(round(d))
        n_i = a_i + b_i + c_i + d_i

        # proportions
        diff_p = p1 - p2
        prop_match = (a + d) / n
        prop_not_match = 1.0 - prop_match

        # McNemar
        # exact uses integer table
        exact = mcnemar([[a_i, b_i], [c_i, d_i]], exact=True)
        chi2_ = ((abs(b - c)) ** 2) / (b + c) if (b + c) > 0 else 0.0
        chi2_cc = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0
        p_chi2 = 1 - chi2.cdf(chi2_, 1)
        p_chi2_cc = 1 - chi2.cdf(chi2_cc, 1)

        mcnemar_stat = res.InferentialStatistics(score=chi2_, p_value=p_chi2)
        mcnemar_stat.degrees_of_freedom = 1
        mcnemar_cc_stat = res.InferentialStatistics(score=chi2_cc, p_value=p_chi2_cc)
        mcnemar_cc_stat.degrees_of_freedom = 1

        # ===== Difference CIs for paired proportions =====
        # 1) Wald
        se_wald = math.sqrt((b + c) - ((b - c) ** 2) / n) / n if n > 0 else 0.0
        ci_wald = res.ConfidenceInterval(
            diff_p - zcrit * se_wald, diff_p + zcrit * se_wald
        )

        # 2) Wald + continuity correction (Edwards/Fleiss)
        se_wald_cc = se_wald  # same SE, add ±1/n
        ci_wald_cc = res.ConfidenceInterval(
            diff_p - zcrit * se_wald_cc - (1 / n),
            diff_p + zcrit * se_wald_cc + (1 / n),
        )

        # 3) Wald + Yates
        se_wald_y = math.sqrt((b + c) - ((b - c - 1) ** 2) / n) / n if n > 0 else 0.0
        ci_wald_y = res.ConfidenceInterval(
            diff_p - zcrit * se_wald_y, diff_p + zcrit * se_wald_y
        )

        # 4) Agresti & Min (2005)
        se_am = math.sqrt(
            ((b + 0.5) + (c + 0.5)) - (((b + 0.5) - (c + 0.5)) ** 2) / (n + 2)
        ) / (n + 2)
        center_am = ((b + 0.5) - (c + 0.5)) / (n + 2)
        ci_am = res.ConfidenceInterval(
            center_am - zcrit * se_am, center_am + zcrit * se_am
        )

        # 5) Bonett & Price (2012)
        p1_adj = (b + 1) / (n + 2)
        p2_adj = (c + 1) / (n + 2)
        se_bp = math.sqrt((p1_adj + p2_adj - (p2_adj - p1_adj) ** 2) / (n + 2))
        center_bp = p1_adj - p2_adj
        ci_bp = res.ConfidenceInterval(
            center_bp - zcrit * se_bp, center_bp + zcrit * se_bp
        )

        # 6) Newcomb square-and-add
        n1tot = a + b
        n2tot = a + c
        A1 = (2 * n * (n1tot / n) + zcrit**2) / (2 * n + 2 * zcrit**2)
        B1 = (
            zcrit * math.sqrt(zcrit**2 + 4 * n * (n1tot / n) * (1 - (n1tot / n)))
        ) / (2 * n + 2 * zcrit**2)
        A2 = (2 * n * (n2tot / n) + zcrit**2) / (2 * n + 2 * zcrit**2)
        B2 = (
            zcrit * math.sqrt(zcrit**2 + 4 * n * (n2tot / n) * (1 - (n2tot / n)))
        ) / (2 * n + 2 * zcrit**2)
        lp1, up1 = A1 - B1, A1 + B1
        lp2, up2 = A2 - B2, A2 + B2

        # correlation/product correction
        if n1tot == 0 or n2tot == 0 or (n - n1tot) == 0 or (n - n2tot) == 0:
            prod_corr = 0.0
        else:
            marg_prod = n1tot * n2tot * (n - n1tot) * (n - n2tot)
            cell_prod = a * d - c * b
            if cell_prod > n / 2:
                prod_corr = (cell_prod - n / 2) / math.sqrt(marg_prod)
            elif 0 <= cell_prod <= n / 2:
                prod_corr = 0.0
            else:
                prod_corr = cell_prod / math.sqrt(marg_prod)

        ci_newcomb = res.ConfidenceInterval(
            diff_p
            - math.sqrt(
                (p1 - lp1) ** 2
                + (up2 - p2) ** 2
                - 2 * prod_corr * (p1 - lp1) * (up2 - p2)
            ),
            diff_p
            + math.sqrt(
                (p2 - lp2) ** 2
                + (up1 - p1) ** 2
                - 2 * prod_corr * (p2 - lp2) * (up1 - p1)
            ),
        )

        diff_cis = PairedDifferenceCIs(
            wald=ci_wald,
            wald_cc=ci_wald_cc,
            wald_yates=ci_wald_y,
            agresti_min=ci_am,
            bonett_price=ci_bp,
            newcomb=ci_newcomb,
        )

        # ===== Matched-pairs Odds Ratio θ = b/c =====
        theta = (b / c) if c > 0 else math.inf
        se_or = math.sqrt((1 / b) + (1 / c)) if (b > 0 and c > 0) else math.inf
        se_or_adj = math.sqrt((1 / (b + 1)) + (1 / (c + 1)))  # Jewell

        # 1) Wald (log) CI
        ci_or_wald_log = res.ConfidenceInterval(
            (
                theta * math.exp(-zcrit * se_or)
                if np.isfinite(se_or) and theta > 0
                else 0.0
            ),
            (
                theta * math.exp(+zcrit * se_or)
                if np.isfinite(se_or) and theta > 0
                else math.inf
            ),
        )

        # 2) Laplace (+1) adjusted Wald
        ci_or_laplace = res.ConfidenceInterval(
            math.exp(math.log((b + 1) / (c + 1)) - zcrit * se_or_adj),
            math.exp(math.log((b + 1) / (c + 1)) + zcrit * se_or_adj),
        )

        # 3) Binomial (Clopper–Pearson on p=b/(b+c))
        if (b + c) > 0:
            lo_p, hi_p = proportion_confint(
                count=int(round(b)),
                nobs=int(round(b + c)),
                alpha=(1 - cl),
                method="beta",
            )
            ci_or_binom = res.ConfidenceInterval(lo_p / (1 - lo_p), hi_p / (1 - hi_p))
        else:
            ci_or_binom = res.ConfidenceInterval(0.0, math.inf)

        # 4) Rigby & Robinson (McNemar-based)
        rr_lo = (
            theta ** (1 - zcrit / math.sqrt(chi2_)) if chi2_ > 0 and theta > 0 else 0.0
        )
        rr_hi = (
            theta ** (1 + zcrit / math.sqrt(chi2_))
            if chi2_ > 0 and theta > 0
            else math.inf
        )
        ci_or_rr = res.ConfidenceInterval(rr_lo, rr_hi)

        # 5) Rigby & Robinson with CC
        rrcc_lo = (
            theta ** (1 - zcrit / math.sqrt(chi2_cc))
            if chi2_cc > 0 and theta > 0
            else 0.0
        )
        rrcc_hi = (
            theta ** (1 + zcrit / math.sqrt(chi2_cc))
            if chi2_cc > 0 and theta > 0
            else math.inf
        )
        ci_or_rrcc = res.ConfidenceInterval(rrcc_lo, rrcc_hi)

        # 6) Transformed Wilson score (closed form)
        A_ws = (2 * b * c) + zcrit**2 * (b + c)
        disc = max(A_ws**2 - (2 * b * c) ** 2, 0.0)
        ci_ws_lo = (A_ws - math.sqrt(disc)) / (2 * (c**2)) if c > 0 else 0.0
        ci_ws_hi = (A_ws + math.sqrt(disc)) / (2 * (c**2)) if c > 0 else math.inf

        # 7) Fiducial (beta with 0.5)
        beta_lo = (
            beta.ppf((1 - cl) / 2, b + 0.5, c + 0.5) if (b >= 0 and c >= 0) else 0.0
        )
        beta_hi = (
            beta.ppf(1 - (1 - cl) / 2, b + 0.5, c + 0.5) if (b >= 0 and c >= 0) else 1.0
        )
        ci_fid = res.ConfidenceInterval(
            beta_lo / (1 - beta_lo), beta_hi / (1 - beta_hi)
        )

        or_cis = MatchedOddsRatioCIs(
            wald_log=ci_or_wald_log,
            wald_laplace=ci_or_laplace,
            binomial=ci_or_binom,
            rigby_robinson=ci_or_rr,
            rigby_robinson_cc=ci_or_rrcc,
            wilson_score_transformed=res.ConfidenceInterval(ci_ws_lo, ci_ws_hi),
            fiducial=ci_fid,
        )

        # ===== Matched-pairs Risk Ratio RR = (a+b)/(a+c) =====
        rr = ((a + b) / (a + c)) if (a + c) > 0 else math.inf

        # Wald for RR (delta on log scale)
        se_rr_log = (
            math.sqrt((b + c) / (n1tot * n2tot))
            if (n1tot > 0 and n2tot > 0)
            else math.inf
        )
        ci_rr_wald = res.ConfidenceInterval(
            rr * math.exp(-zcrit * se_rr_log), rr * math.exp(zcrit * se_rr_log)
        )

        # Bonett–Price (score-based, without/with CC)
        n_star = n - d
        A = (
            math.sqrt((b + c + 2) / ((n1tot + 1) * (n2tot + 1)))
            if (n1tot + 1) > 0 and (n2tot + 1) > 0
            else math.inf
        )
        B = (
            math.sqrt((1 - (n1tot + 1) / (n_star + 2)) / (n1tot + 1))
            if (n_star + 2) > 0 and (n1tot + 1) > 0
            else math.inf
        )
        C = (
            math.sqrt((1 - (n2tot + 1) / (n_star + 2)) / (n2tot + 1))
            if (n_star + 2) > 0 and (n2tot + 1) > 0
            else math.inf
        )
        z_bp = (
            (A / (B + C)) * zcrit
            if np.isfinite(A) and np.isfinite(B) and np.isfinite(C)
            else math.inf
        )

        def _wilson_center_margin(k, N, z):
            center = 2 * k + z**2
            margin = z * math.sqrt(z**2 + 4 * k * (1 - k / N))
            denom = 2 * (N + z**2)
            return (center - margin) / denom, (center + margin) / denom

        l1, u1 = _wilson_center_margin(n1tot, n_star, z_bp)
        l2, u2 = _wilson_center_margin(n2tot, n_star, z_bp)
        ci_rr_bp = res.ConfidenceInterval(l1 / u2, u1 / l2)

        # CC variant
        def _wilson_lower_cc(k, N, z):
            num = (
                2 * k
                + z**2
                - 1
                - z * math.sqrt(z**2 - 2 - (1 / N) + 4 * (k / N) * (N - k + 1))
            )
            den = 2 * (N + z**2)
            return num / den

        def _wilson_upper_cc(k, N, z):
            num = (
                2 * k
                + z**2
                + 1
                + z * math.sqrt(z**2 + 2 - (1 / N) + 4 * (k / N) * (N - k - 1))
            )
            den = 2 * (N + z**2)
            return num / den

        l1_cc = _wilson_lower_cc(n1tot, n_star, z_bp)
        u1_cc = _wilson_upper_cc(n1tot, n_star, z_bp)
        l2_cc = _wilson_lower_cc(n2tot, n_star, z_bp)
        u2_cc = _wilson_upper_cc(n2tot, n_star, z_bp)
        ci_rr_bp_cc = res.ConfidenceInterval(l1_cc / u2_cc, u1_cc / l2_cc)

        # MOVER Wilson (Tang et al. 2010 style)
        l1_M, u1_M = _wilson_center_margin(n1tot, n, zcrit)
        l2_M, u2_M = _wilson_center_margin(n2tot, n, zcrit)
        # correlation term
        corr = (a * d - b * c) / math.sqrt(
            max(n1tot, 1e-12)
            * max(n2tot, 1e-12)
            * max(d + c, 1e-12)
            * max(d + b, 1e-12)
        )
        A_mv = (p1 - l1_M) * (u2_M - p2) * corr
        B_mv = (u1_M - p1) * (p2 - l2_M) * corr
        disc_lo = (A_mv - p1 * p2) ** 2 - l1_M * (2 * p1 - l1_M) * u2_M * (
            2 * p2 - u2_M
        )
        disc_hi = (B_mv - p1 * p2) ** 2 - u1_M * (2 * p1 - u1_M) * l2_M * (
            2 * p2 - l2_M
        )
        lower_MOVER = (A_mv - p1 * p2 + math.sqrt(max(disc_lo, 0.0))) / (
            u2_M * (u2_M - 2 * p2)
        )
        upper_MOVER = (B_mv - p1 * p2 - math.sqrt(max(disc_hi, 0.0))) / (
            l2_M * (l2_M - 2 * p2)
        )
        ci_rr_mover = res.ConfidenceInterval(lower_MOVER, upper_MOVER)

        # Tang method (test-inversion; compact version)
        x_vec = np.array([a, b, c, d], dtype=float)

        def _tang_T1(delta: float):
            N = np.sum(x_vec)
            Stheta = (x_vec[1] + x_vec[0]) - (x_vec[2] + x_vec[0]) * delta
            A_ = N * (1 + delta)
            B_ = (x_vec[0] + x_vec[2]) * delta**2 - (x_vec[0] + x_vec[1] + 2 * x_vec[2])
            C_ = x_vec[2] * (1 - delta) * (x_vec[0] + x_vec[1] + x_vec[2]) / N
            num = -B_ + math.sqrt(max(B_**2 - 4 * A_ * C_, 0.0))
            q21 = num / (2 * A_)
            var = max(
                0.0,
                N * (1 + delta) * q21 + (x_vec[0] + x_vec[1] + x_vec[2]) * (delta - 1),
            )
            z_stat = Stheta / math.sqrt(var) if var > 0 else 0.0
            return z_stat

        def _bsearch(f, lower=True, iters=100):
            hi, lo = 1.0, -1.0
            for _ in range(iters):
                mid = max(-1.0, min(1.0, (hi + lo) / 2))
                # map to (0,∞) using tan transform like original code
                delta = math.tan(math.pi * (mid + 1) / 4)
                val = f(delta)
                check = (val <= 0) or np.isnan(val)
                if lower:
                    hi = mid if check else hi
                    lo = lo if check else mid
                else:
                    hi = hi if check else mid
                    lo = mid if check else lo
            return math.tan(math.pi * ((lo if lower else hi) + 1) / 4)

        ci_rr_tang = res.ConfidenceInterval(
            _bsearch(lambda dlt: _tang_T1(dlt) - zcrit, lower=True),
            _bsearch(lambda dlt: _tang_T1(dlt) + zcrit, lower=False),
        )

        rr_cis = MatchedRiskRatioCIs(
            wald=ci_rr_wald,
            tang=ci_rr_tang,
            bonett_price=ci_rr_bp,
            bonett_price_cc=ci_rr_bp_cc,
            mover_wilson=ci_rr_mover,
        )

        # epidemiologic extras
        irr_exp = p1
        irr_unexp = p2
        irr_pop = ((a + b) + (a + c)) / n if n > 0 else float("nan")
        nnt = (1.0 / diff_p) if diff_p != 0 else float("inf")
        pop_attrib_risk = (
            (((a + b) / (a + b + a + c)) * ((rr - 1) / rr))
            if np.isfinite(rr) and rr > 0
            else float("nan")
        )

        return TwoDependentProportionsResults(
            n=n_i,
            yes_yes=a_i,
            yes_no=b_i,
            no_yes=c_i,
            no_no=d_i,
            p1=p1,
            p2=p2,
            diff_p=diff_p,
            prop_matching=prop_match,
            prop_not_matching=prop_not_match,
            mcnemar_chi2=mcnemar_stat,
            mcnemar_chi2_cc=mcnemar_cc_stat,
            mcnemar_exact_p=float(exact.pvalue),
            diff_cis=diff_cis,
            matched_or=float(theta),
            matched_or_se=float(se_or),
            matched_or_se_adj=float(se_or_adj),
            matched_or_cis=or_cis,
            matched_rr=float(rr),
            matched_rr_cis=rr_cis,
            nnt=float(nnt),
            pop_attrib_risk=float(pop_attrib_risk),
            ir_exposed=float(irr_exp),
            ir_unexposed=float(irr_unexp),
            ir_population=float(irr_pop),
        )

    @staticmethod
    def from_data(
        col1: np.ndarray,
        col2: np.ndarray,
        success_value,
        confidence_level: float = 95.0,
    ) -> TwoDependentProportionsResults:
        df = pd.DataFrame({"c1": col1, "c2": col2})
        df = df.dropna(subset=["c1", "c2"])
        n = df.shape[0]
        p1 = float((df["c1"] == success_value).sum() / n)
        p2 = float((df["c2"] == success_value).sum() / n)
        a = int(((df["c1"] == success_value) & (df["c2"] == success_value)).sum())
        return TwoDependentProportions.from_parameters(
            p1=p1, p2=p2, p_yes_yes=a / n, n=n, confidence_level=confidence_level
        )

    @staticmethod
    def from_frequencies(
        yes_yes: int,
        yes_no: int,
        no_yes: int,
        no_no: int,
        confidence_level: float = 95.0,
    ) -> TwoDependentProportionsResults:
        n = yes_yes + yes_no + no_yes + no_no
        p1 = (yes_yes + yes_no) / n
        p2 = (yes_yes + no_yes) / n
        return TwoDependentProportions.from_parameters(
            p1=p1, p2=p2, p_yes_yes=yes_yes / n, n=n, confidence_level=confidence_level
        )