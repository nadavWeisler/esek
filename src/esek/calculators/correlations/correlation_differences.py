"""Tests for difference between Pearson correlations.

Migrated and refactored from:
``stats/Differecnes/DifferencesBetweenCorrelations/Pearson Correlations/
Pearson_Correlations_diff.py``
in the ``dev`` branch.

Provides significance tests and CIs for:
- Independent samples: r₁₂ vs r₃₄ from different samples
- Dependent non-overlapping: r₁₂ vs r₃₄ from the same sample (no shared variable)
- Dependent overlapping: r₁₂ vs r₁₃ from the same sample (shared variable 1)

Statistical references:
    - Pearson & Filon (1898)
    - Hotelling (1940)
    - Williams (1959)
    - Dunn & Clark (1969)
    - Steiger (1980)
    - Raghunathan, Rosenthal & Rubin (1996)
    - Silver, Hittner & May (2004)
    - Meng, Rosenthal & Rubin (1992)
    - Hittner, May & Silver (2003)
    - Zou (2007)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm, t


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationDifferenceResult:
    """Result of a significance test for the difference between two correlations.

    Attributes:
        r1: First correlation coefficient.
        r2: Second correlation coefficient.
        difference: r1 − r2.
        cohens_q: Cohen's q = atanh(r1) − atanh(r2) (effect size for difference).
        tests: Dict of {method_name: (statistic, p_value)}.
        ci_zou: Zou (2007) CI for the difference (lower, upper).
        ci_meng: Meng (1992) CI (only for overlapping design; None otherwise).
        r1_ci: Fisher z CI for r1 (lower, upper).
        r2_ci: Fisher z CI for r2 (lower, upper).
        r1_t_stat: t-statistic for H₀: ρ₁ = 0.
        r1_p_value: p-value for H₀: ρ₁ = 0 (one-tailed).
        r2_t_stat: t-statistic for H₀: ρ₂ = 0.
        r2_p_value: p-value for H₀: ρ₂ = 0 (one-tailed).
        n: Sample size (for dependent designs) or None.
        n1: Sample size for first correlation (independent design) or None.
        n2: Sample size for second correlation (independent design) or None.
        confidence_level: Nominal CI level.
        design: ``"independent"``, ``"dependent_non_overlapping"``,
                or ``"dependent_overlapping"``.
        metadata: Additional quantities.
    """

    r1: float
    r2: float
    difference: float
    cohens_q: float
    tests: dict[str, tuple[float, float]]
    ci_zou: tuple[float, float]
    r1_ci: tuple[float, float]
    r2_ci: tuple[float, float]
    r1_t_stat: float
    r1_p_value: float
    r2_t_stat: float
    r2_p_value: float
    confidence_level: float
    design: str
    n: int | None = None
    n1: int | None = None
    n2: int | None = None
    ci_meng: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _t_and_p(r: float, n: int) -> tuple[float, float]:
    """Two-tailed t-test for H₀: ρ = 0."""
    df = n - 2
    if abs(r) >= 1.0:
        return math.inf, 0.0
    t_stat = r * math.sqrt(df) / math.sqrt(1.0 - r**2)
    p_value = float(t.sf(abs(t_stat), df)) * 2  # two-tailed
    return float(t_stat), float(p_value)


def _fisher_ci(r: float, n: int, confidence_level: float) -> tuple[float, float]:
    """Fisher z-transform CI for a single correlation."""
    z_crit = float(norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))
    se = math.sqrt(1.0 / max(n - 3, 1))
    zr = math.atanh(max(min(r, 0.9999999), -0.9999999))
    return (
        float(math.tanh(zr - z_crit * se)),
        float(math.tanh(zr + z_crit * se)),
    )


# ---------------------------------------------------------------------------
# Public calculator
# ---------------------------------------------------------------------------


class PearsonCorrelationDifference:
    """Tests for differences between two Pearson correlations.

    Three design types are supported:

    1. **Independent** — two correlations from completely separate samples.
       Uses: Fisher z-transformation z-test.
       CI: Fisher, Zou (2007).

    2. **Dependent non-overlapping** — two correlations from the *same* sample,
       sharing *no* variable: e.g., r(X₁, X₂) vs r(X₃, X₄).
       Five test methods (Pearson-Filon, Dunn-Clark, Steiger, Raghunathan,
       Silver-Hittner-May).
       CI: Zou (2007).

    3. **Dependent overlapping** — two correlations from the *same* sample,
       sharing *one* variable: e.g., r(X₁, X₂) vs r(X₁, X₃).
       Seven test methods (Pearson-Filon, Hotelling, Williams, Olkin, Dunn,
       Hendrickson, Steiger, Meng, Hittner).
       CIs: Zou (2007), Meng (1992).

    Example::

        from esek.calculators.correlations import PearsonCorrelationDifference

        # Two independent correlations
        result = PearsonCorrelationDifference.independent(
            r1=0.5, n1=80, r2=0.3, n2=90
        )
        print(result.cohens_q, result.ci_zou)

        # Dependent overlapping (r12 vs r13 with r23=0.4)
        result = PearsonCorrelationDifference.dependent_overlapping(
            r12=0.6, r13=0.4, r23=0.4, n=100
        )
        print(result.tests)
    """

    @staticmethod
    def independent(
        r1: float,
        n1: int,
        r2: float,
        n2: int,
        confidence_level: float = 0.95,
    ) -> CorrelationDifferenceResult:
        """Test the difference between two independent Pearson correlations.

        Parameters:
            r1: First correlation (from sample 1).
            n1: Sample size for r1.
            r2: Second correlation (from sample 2).
            n2: Sample size for r2.
            confidence_level: Nominal CI level.

        Returns:
            :class:`CorrelationDifferenceResult`.
        """
        _validate_inputs(r1, r2, confidence_level)
        if n1 < 4 or n2 < 4:
            raise ValueError(f"Each sample must have n ≥ 4 (got n1={n1}, n2={n2}).")

        zr1 = math.atanh(max(min(r1, 0.9999999), -0.9999999))
        zr2 = math.atanh(max(min(r2, 0.9999999), -0.9999999))
        cohens_q = zr1 - zr2
        difference = r1 - r2

        se_diff = math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
        z_stat = cohens_q / se_diff
        p_value = float(norm.sf(abs(z_stat))) * 2  # two-tailed

        z_crit = float(norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))
        ci_fisher = (
            math.tanh(cohens_q - z_crit * se_diff),
            math.tanh(cohens_q + z_crit * se_diff),
        )

        # Zou (2007) CI
        r1_ci = _fisher_ci(r1, n1, confidence_level)
        r2_ci = _fisher_ci(r2, n2, confidence_level)
        ci_zou = (
            float(difference - math.sqrt((r1 - r1_ci[0])**2 + (r2_ci[1] - r2)**2)),
            float(difference + math.sqrt((r1_ci[1] - r1)**2 + (r2 - r2_ci[0])**2)),
        )

        t1, p1 = _t_and_p(r1, n1)
        t2, p2 = _t_and_p(r2, n2)

        return CorrelationDifferenceResult(
            r1=round(r1, 6),
            r2=round(r2, 6),
            difference=round(difference, 6),
            cohens_q=round(cohens_q, 6),
            tests={"fisher_z": (round(z_stat, 6), round(p_value, 6))},
            ci_zou=(round(ci_zou[0], 6), round(ci_zou[1], 6)),
            r1_ci=(round(r1_ci[0], 6), round(r1_ci[1], 6)),
            r2_ci=(round(r2_ci[0], 6), round(r2_ci[1], 6)),
            r1_t_stat=round(t1, 6),
            r1_p_value=round(p1, 6),
            r2_t_stat=round(t2, 6),
            r2_p_value=round(p2, 6),
            confidence_level=confidence_level,
            design="independent",
            n1=int(n1),
            n2=int(n2),
            metadata={"ci_fisher": (round(ci_fisher[0], 6), round(ci_fisher[1], 6))},
        )

    @staticmethod
    def dependent_non_overlapping(
        r12: float,
        r34: float,
        r13: float,
        r14: float,
        r23: float,
        r24: float,
        n: int,
        confidence_level: float = 0.95,
    ) -> CorrelationDifferenceResult:
        """Test the difference between two non-overlapping dependent correlations.

        The two correlations r(X₁,X₂) and r(X₃,X₄) share no variable but
        come from the *same* sample (n).

        Parameters:
            r12: Correlation between variables 1 and 2.
            r34: Correlation between variables 3 and 4.
            r13: Correlation between variables 1 and 3.
            r14: Correlation between variables 1 and 4.
            r23: Correlation between variables 2 and 3.
            r24: Correlation between variables 2 and 4.
            n: Sample size.
            confidence_level: Nominal CI level.

        Returns:
            :class:`CorrelationDifferenceResult`.

        References:
            Pearson & Filon (1898), Dunn & Clark (1969),
            Steiger (1980), Raghunathan et al. (1996), Silver et al. (2004).
        """
        _validate_inputs(r12, r34, confidence_level)
        if n < 6:
            raise ValueError(f"n must be ≥ 6 for non-overlapping design (got {n}).")

        zr12 = math.atanh(max(min(r12, 0.9999999), -0.9999999))
        zr34 = math.atanh(max(min(r34, 0.9999999), -0.9999999))
        mean_r = (r12 + r34) / 2.0
        mean_zr = math.atan((math.atanh(r12) + math.atanh(r34)) / 2.0)
        cohens_q = zr12 - zr34
        difference = r12 - r34

        # Asymptotic term (used by multiple tests)
        term1 = (
            (r13 - r12 * r23) * (r24 - r23 * r34)
            + (r14 - r13 * r34) * (r23 - r12 * r13)
            + (r13 - r14 * r34) * (r24 - r12 * r14)
            + (r14 - r12 * r24) * (r23 - r24 * r34)
        )
        term2 = (
            (r13 - mean_r * r23) * (r24 - r23 * mean_r)
            + (r14 - r13 * mean_r) * (r23 - mean_r * r13)
            + (r13 - r14 * mean_r) * (r24 - mean_r * r14)
            + (r14 - mean_r * r24) * (r23 - r24 * mean_r)
        )
        term3 = (
            (r13 - mean_zr * r23) * (r24 - r23 * mean_zr)
            + (r14 - r13 * mean_zr) * (r23 - mean_zr * r13)
            + (r13 - r14 * mean_zr) * (r24 - mean_zr * r14)
            + (r14 - mean_zr * r24) * (r23 - r24 * mean_zr)
        )

        denom1 = max(((1 - r12**2)**2 + (1 - r34**2)**2) - term1, 1e-12)
        denom2 = max(2 - 2 * (term1 / (2 * (1 - r12**2) * (1 - r34**2))), 1e-12)
        denom3 = max(2 - 2 * (term2 / (2 * (1 - mean_r**2)**2)), 1e-12)
        denom4 = max(1 - (term1 / (2 * (1 - r12**2) * (1 - r34**2))), 1e-12)
        denom5 = max(2 - 2 * (term3 / (2 * (1 - mean_zr**2)**2)), 1e-12)

        stat_pearson = math.sqrt(n) * difference / math.sqrt(denom1)
        stat_dunn = math.sqrt(n - 3) * cohens_q / math.sqrt(denom2)
        stat_steiger = math.sqrt(n - 3) * cohens_q / math.sqrt(denom3)
        stat_raghunathan = math.sqrt((n - 3) / 2.0) * cohens_q / math.sqrt(denom4)
        stat_silver = math.sqrt(n - 3) * cohens_q / math.sqrt(denom5)

        tests = {
            "pearson_filon_1898": (round(stat_pearson, 6), round(float(norm.sf(abs(stat_pearson))) * 2, 6)),
            "dunn_clark_1969": (round(stat_dunn, 6), round(float(norm.sf(abs(stat_dunn))) * 2, 6)),
            "steiger_1980": (round(stat_steiger, 6), round(float(norm.sf(abs(stat_steiger))) * 2, 6)),
            "raghunathan_1996": (round(stat_raghunathan, 6), round(float(norm.sf(abs(stat_raghunathan))) * 2, 6)),
            "silver_2004": (round(stat_silver, 6), round(float(norm.sf(abs(stat_silver))) * 2, 6)),
        }

        # CIs
        r12_ci = _fisher_ci(r12, n, confidence_level)
        r34_ci = _fisher_ci(r34, n, confidence_level)

        # Zou CI uses cross-term c
        c = (
            0.5 * r12 * r34 * (r13**2 + r14**2 + r23**2 + r24**2)
            + r13 * r24 + r14 * r23
            - (r12 * r13 * r14 + r12 * r23 * r24 + r13 * r23 * r34 + r14 * r24 * r34)
        ) / ((1 - r12**2) * (1 - r34**2))

        ci_zou = (
            float(difference - math.sqrt(
                (r12 - r12_ci[0])**2 + (r34_ci[1] - r34)**2
                - 2 * c * (r12 - r12_ci[0]) * (r34_ci[1] - r34)
            )),
            float(difference + math.sqrt(
                (r12_ci[1] - r12)**2 + (r34 - r34_ci[0])**2
                - 2 * c * (r12_ci[1] - r12) * (r34 - r34_ci[0])
            )),
        )

        t12, p12 = _t_and_p(r12, n)
        t34, p34 = _t_and_p(r34, n)

        return CorrelationDifferenceResult(
            r1=round(r12, 6),
            r2=round(r34, 6),
            difference=round(difference, 6),
            cohens_q=round(cohens_q, 6),
            tests=tests,
            ci_zou=(round(ci_zou[0], 6), round(ci_zou[1], 6)),
            r1_ci=(round(r12_ci[0], 6), round(r12_ci[1], 6)),
            r2_ci=(round(r34_ci[0], 6), round(r34_ci[1], 6)),
            r1_t_stat=round(t12, 6),
            r1_p_value=round(p12, 6),
            r2_t_stat=round(t34, 6),
            r2_p_value=round(p34, 6),
            confidence_level=confidence_level,
            design="dependent_non_overlapping",
            n=int(n),
            metadata={"cross_term_c": round(c, 6)},
        )

    @staticmethod
    def dependent_overlapping(
        r12: float,
        r13: float,
        r23: float,
        n: int,
        confidence_level: float = 0.95,
    ) -> CorrelationDifferenceResult:
        """Test the difference between two overlapping dependent correlations.

        The two correlations r(X₁,X₂) and r(X₁,X₃) share variable X₁ and
        come from the *same* sample.

        Parameters:
            r12: Correlation between variables 1 and 2.
            r13: Correlation between variables 1 and 3.
            r23: Correlation between variables 2 and 3.
            n: Sample size.
            confidence_level: Nominal CI level.

        Returns:
            :class:`CorrelationDifferenceResult`.

        References:
            Hotelling (1940), Williams (1959), Olkin (1967), Dunn & Clark (1969),
            Hendrickson (1970), Steiger (1980), Meng et al. (1992),
            Hittner et al. (2003), Zou (2007).
        """
        _validate_inputs(r12, r13, confidence_level)
        if n < 5:
            raise ValueError(f"n must be ≥ 5 for overlapping design (got {n}).")

        zr12 = math.atanh(max(min(r12, 0.9999999), -0.9999999))
        zr13 = math.atanh(max(min(r13, 0.9999999), -0.9999999))
        cohens_q = zr12 - zr13
        difference = r12 - r13
        df = n - 3
        mean_r = (r12 + r13) / 2.0
        mean_r_sq = (r12**2 + r13**2) / 2.0
        mean_zr = math.tanh((math.atanh(r12) + math.atanh(r13)) / 2.0)

        # Shared cross-term
        term1 = (r12**2 + r13**2 - 2 * r23 * r12 * r13) / max(1.0 - r23**2, 1e-12)
        term2 = (
            r23 * (1 - r12**2 - r13**2) - 0.5 * r12 * r13 * (1 - r12**2 - r13**2 - r23**2)
        ) / ((1 - r12**2) * (1 - r13**2))
        term3 = (
            r23 * (1 - 2 * mean_r**2) - 0.5 * mean_r**2 * (1 - 2 * mean_r**2 - r23**2)
        ) / max((1 - mean_r**2)**2, 1e-12)
        term4 = (
            r23 * (1 - 2 * mean_zr**2) - 0.5 * mean_zr**2 * (1 - 2 * mean_zr**2 - r23**2)
        ) / max((1 - mean_zr**2)**2, 1e-12)

        # Meng's f correction factor
        raw_f = (1 - r23) / (2.0 * (1 - mean_r_sq))
        meng_f = min(raw_f, 1.0) if not math.isnan(raw_f) else 1.0

        denom_pearson = math.sqrt(max(
            (1 - r12**2)**2 + (1 - r13**2)**2
            - 2 * r23**3 - (2 * r23 - r12 * r13) * (1 - r12**2 - r13**2 - r23**2),
            1e-12,
        ))
        denom_hotelling = math.sqrt(max(
            2 * (1 + 2 * r12 * r13 * r23 - r12**2 - r13**2 - r23**2), 1e-12
        ))
        denom_williams = math.sqrt(max(
            2 * ((n - 1) / (n - 3))
            * (1 + 2 * r12 * r13 * r23 - r12**2 - r13**2 - r23**2)
            + ((r12 + r13) / 2.0)**2 * (1 - r23)**3,
            1e-12,
        ))
        denom_olkin = math.sqrt(max(
            (1 - r12**2)**2 + (1 - r13**2)**2
            - 2 * r23**3 - (2 * r23 - r12 * r13) * (1 - r12**2 - r13**2 - r23**2),
            1e-12,
        ))
        denom_dunn = math.sqrt(max(2.0 - 2.0 * term2, 1e-12))
        denom_hendrickson = math.sqrt(max(
            2 * (1 + 2 * r12 * r13 * r23 - r12**2 - r13**2 - r23**2)
            + (((r12 - r13)**2 * (1 - r23)**3) / (4 * (n - 1))),
            1e-12,
        ))
        denom_steiger = math.sqrt(max(2.0 - 2.0 * term3, 1e-12))
        denom_meng = math.sqrt(max(
            2.0 * (1 - r23) * ((1 - meng_f * mean_r_sq) / max(1 - mean_r_sq, 1e-12)) / max(df, 1),
            1e-12,
        ))
        denom_hittner = math.sqrt(max(2.0 - 2.0 * term4, 1e-12))

        stat_pearson = math.sqrt(n) * difference / denom_pearson
        stat_hotelling = difference * math.sqrt((n - 3) * (1 + r23)) / denom_hotelling
        stat_williams = difference * math.sqrt(((n - 1) * (1 + r23))) / denom_williams
        stat_olkin = difference * math.sqrt(n) / denom_olkin
        stat_dunn = math.sqrt(df) * cohens_q / denom_dunn
        stat_hendrickson = difference * math.sqrt((n - 3) * (1 + r23)) / denom_hendrickson
        stat_steiger = math.sqrt(df) * cohens_q / denom_steiger
        stat_meng = cohens_q / denom_meng
        stat_hittner = math.sqrt(df) * cohens_q / denom_hittner

        n_arr = float(np.array(n))  # ensure float for t.sf

        tests = {
            "pearson_filon_1898": (round(stat_pearson, 6), round(float(norm.sf(abs(stat_pearson))) * 2, 6)),
            "hotelling_1940": (round(stat_hotelling, 6), round(float(t.sf(abs(stat_hotelling), n - 2)) * 2, 6)),
            "williams_1959": (round(stat_williams, 6), round(float(t.sf(abs(stat_williams), n - 3)) * 2, 6)),
            "olkin_1967": (round(stat_olkin, 6), round(float(norm.sf(abs(stat_olkin))) * 2, 6)),
            "dunn_clark_1969": (round(stat_dunn, 6), round(float(norm.sf(abs(stat_dunn))) * 2, 6)),
            "hendrickson_1970": (round(stat_hendrickson, 6), round(float(t.sf(abs(stat_hendrickson), n - 3)) * 2, 6)),
            "steiger_1980": (round(stat_steiger, 6), round(float(norm.sf(abs(stat_steiger))) * 2, 6)),
            "meng_1992": (round(stat_meng, 6), round(float(norm.sf(abs(stat_meng))) * 2, 6)),
            "hittner_2003": (round(stat_hittner, 6), round(float(norm.sf(abs(stat_hittner))) * 2, 6)),
        }

        # CIs
        r12_ci = _fisher_ci(r12, n, confidence_level)
        r13_ci = _fisher_ci(r13, n, confidence_level)

        # Zou (2007) CI for overlapping case
        c_zou = (
            (r23 - 0.5 * r12 * r13) * (1 - r12**2 - r13**2 - r23**2) + r23**3
        ) / ((1 - r12**2) * (1 - r13**2))
        ci_zou = (
            float(difference - math.sqrt(
                (r12 - r12_ci[0])**2 + (r13_ci[1] - r13)**2
                - 2 * c_zou * (r12 - r12_ci[0]) * (r13_ci[1] - r13)
            )),
            float(difference + math.sqrt(
                (r12_ci[1] - r12)**2 + (r13 - r13_ci[0])**2
                - 2 * c_zou * (r12_ci[1] - r12) * (r13 - r13_ci[0])
            )),
        )

        # Meng (1992) CI (on Fisher z scale)
        se_meng = math.sqrt(
            2.0 * (1 - r23) * ((1 - meng_f * mean_r_sq) / max(1 - mean_r_sq, 1e-12)) / max(df, 1)
        )
        z_crit = float(norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))
        ci_meng = (
            float(cohens_q - z_crit * se_meng),
            float(cohens_q + z_crit * se_meng),
        )

        t12, p12 = _t_and_p(r12, n)
        t13, p13 = _t_and_p(r13, n)

        return CorrelationDifferenceResult(
            r1=round(r12, 6),
            r2=round(r13, 6),
            difference=round(difference, 6),
            cohens_q=round(cohens_q, 6),
            tests=tests,
            ci_zou=(round(ci_zou[0], 6), round(ci_zou[1], 6)),
            r1_ci=(round(r12_ci[0], 6), round(r12_ci[1], 6)),
            r2_ci=(round(r13_ci[0], 6), round(r13_ci[1], 6)),
            r1_t_stat=round(t12, 6),
            r1_p_value=round(p12, 6),
            r2_t_stat=round(t13, 6),
            r2_p_value=round(p13, 6),
            confidence_level=confidence_level,
            design="dependent_overlapping",
            n=int(n),
            ci_meng=(round(ci_meng[0], 6), round(ci_meng[1], 6)),
            metadata={"meng_f": round(meng_f, 6), "cross_term_c_zou": round(c_zou, 6)},
        )


def _validate_inputs(r1: float, r2: float, confidence_level: float) -> None:
    if not (-1.0 <= r1 <= 1.0):
        raise ValueError(f"r1 must be in [-1, 1] (got {r1}).")
    if not (-1.0 <= r2 <= 1.0):
        raise ValueError(f"r2 must be in [-1, 1] (got {r2}).")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1) (got {confidence_level}).")
