"""2×2 Contingency Table calculator — effect sizes, CIs, and chi-square tests.

Migrated and refactored from:
``stats/Calculator/ContingencyTables/9A_2x2Table/ContingencyTable2x2.py``
in the ``dev`` branch.

The 2×2 table is assumed to have the following layout::

        |        | Category 1 | Category 2 |
        |--------|-----------|-----------|
        | Row 1  |     a     |     b     |
        | Row 2  |     c     |     d     |

Statistical assumptions:
    - Large-sample approximations are used for CIs (z-based, Wilson-like).
    - The tetrachoric correlation approximations (Pearson 1900; Bonett 2005)
      assume an underlying bivariate normal distribution.
    - Wallis' swing d assumes row or column marginals are fixed by design.
    - All cells a, b, c, d must be ≥ 0.  Cells with 0 values may produce NaN
      for measures that involve division by a cell count (e.g. odds ratio).
      Corrected versions add 0.5 to all cells in that case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.stats as st


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContingencyTable2x2Result:
    """Result of a 2×2 contingency table analysis.

    Attributes:
        phi: Phi coefficient (= Cramér's V for 2×2 tables).
        cramer_v: Cramér's V (equals phi for 2×2 tables).
        bias_corrected_cramer: Bias-corrected Cramér's V (Bergsma, 2013).
        max_corrected_phi: Phi corrected by its theoretical maximum (Liu, 1980).
        odds_ratio: Raw odds ratio (AD / BC).
        tetrachoric_r: Tetrachoric correlation approximation (Pearson 1900).
        tetrachoric_r_corrected: Bias-corrected tetrachoric r (Bonett, 2005).
        chambers_r: Chambers' r measure.
        wallis_swing_d_cols: Wallis' swing d (columns-independent version).
        wallis_swing_d_rows: Wallis' swing d (rows-independent version).
        chi_square: χ² statistic.
        p_value: Two-tailed p-value for χ².
        n: Total sample size.
        confidence_level: Nominal CI level (0–1).
        ci_tetrachoric: CI for tetrachoric r (lower, upper).
        ci_tetrachoric_corrected: CI for corrected tetrachoric r (lower, upper).
        ci_swing_d_cols: CI for swing-d columns (lower, upper).
        ci_swing_d_rows: CI for swing-d rows (lower, upper).
        se_tetrachoric: SE for tetrachoric correlation.
        se_tetrachoric_corrected: SE for corrected tetrachoric correlation.
        metadata: Cell counts and marginals.
    """

    phi: float
    cramer_v: float
    bias_corrected_cramer: float
    max_corrected_phi: float
    odds_ratio: float
    tetrachoric_r: float
    tetrachoric_r_corrected: float
    chambers_r: float
    wallis_swing_d_cols: float
    wallis_swing_d_rows: float
    chi_square: float
    p_value: float
    n: int
    confidence_level: float
    ci_tetrachoric: tuple[float, float]
    ci_tetrachoric_corrected: tuple[float, float]
    ci_swing_d_cols: tuple[float, float]
    ci_swing_d_rows: tuple[float, float]
    se_tetrachoric: float
    se_tetrachoric_corrected: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main calculator class
# ---------------------------------------------------------------------------


class ContingencyTable2x2:
    """Calculate effect sizes and CIs for a 2×2 contingency table.

    The table is passed as a 2×2 NumPy array::

        table = np.array([[a, b], [c, d]])
        result = ContingencyTable2x2.from_table(table, confidence_level=0.95)

    All cells must be non-negative integers.  Zero cells are handled via
    continuity correction (+ 0.5) where mathematically necessary.
    """

    @staticmethod
    def from_table(
        table: np.ndarray,
        confidence_level: float = 0.95,
    ) -> ContingencyTable2x2Result:
        """Compute effect sizes and CIs from a 2×2 contingency table.

        Parameters:
            table: 2×2 array with non-negative integer cell counts.
            confidence_level: Nominal CI level (default 0.95).

        Returns:
            :class:`ContingencyTable2x2Result`.

        Raises:
            ValueError: For invalid table shape, negative cells, or bad CI level.
        """
        table = np.asarray(table, dtype=float)
        if table.shape != (2, 2):
            raise ValueError(f"table must be a 2×2 array (got shape {table.shape}).")
        if np.any(table < 0):
            raise ValueError("All table cells must be ≥ 0.")
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in (0, 1) (got {confidence_level}).")

        a, b = table[0, 0], table[0, 1]
        c, d = table[1, 0], table[1, 1]
        n = a + b + c + d
        if n == 0:
            raise ValueError("Table is all zeros — no observations.")

        z_crit = st.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)

        # ----------------------------------------------------------------
        # Effect sizes
        # ----------------------------------------------------------------

        # 1. Phi / Cramér's V
        denom_phi = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
        phi = (a * d - b * c) / denom_phi if denom_phi > 0 else math.nan
        chi_sq = phi**2 * n if not math.isnan(phi) else math.nan
        p_value = float(st.chi2.sf(abs(chi_sq), 1)) if not math.isnan(chi_sq) else math.nan

        # 2. Bias-corrected Cramér (Bergsma, 2013)
        corrected_levels = 2.0 - 1.0 / (n - 1)
        bias_corr_cramer = (
            math.sqrt(phi**2 / (corrected_levels - 1))
            if not math.isnan(phi) and corrected_levels > 1
            else math.nan
        )

        # 3. Maximum-corrected phi (Liu, 1980)
        max_phi1 = math.sqrt(((a + b) / (c + d)) * ((b + d) / (a + c))) if (c + d) > 0 and (a + c) > 0 else math.nan
        max_phi2 = math.sqrt(((a + c) / (b + d)) * ((a + b) / (c + d))) if (b + d) > 0 and (c + d) > 0 else math.nan
        if not (math.isnan(max_phi1) or math.isnan(max_phi2)):
            max_phi = max_phi1 if (c + d) > (b + a) else max_phi2
            max_corr_phi = phi / max_phi if max_phi > 0 else math.nan
        else:
            max_corr_phi = math.nan

        # 4. Odds ratio (with and without correction for zeros)
        bc_prod = b * c
        odds_ratio = (a * d / bc_prod) if bc_prod > 0 else math.nan
        # Corrected odds ratio (Bonett, 2005) — add 0.5 to all cells
        odds_ratio_corrected = (a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5))

        # 5. Tetrachoric correlation (Pearson 1900 approximation)
        if not math.isnan(odds_ratio) and odds_ratio > 0:
            tetrachoric_r = math.cos(math.pi / (1.0 + math.sqrt(odds_ratio)))
            tetrachoric_basic = (odds_ratio**0.74 - 1.0) / (odds_ratio**0.74 + 1.0)
        else:
            tetrachoric_r = math.nan
            tetrachoric_basic = math.nan

        # 5b. Corrected tetrachoric (Bonett, 2005)
        r1 = (a + b + 1.0) / (n + 2.0)
        r2 = (c + d + 1.0) / (n + 2.0)
        c1 = (a + c + 1.0) / (n + 2.0)
        c2 = (b + d + 1.0) / (n + 2.0)
        min_p = min(c1, c2, r1, r2)
        correction = (1.0 - abs(r1 - c1) / 5.0 - (0.5 - min_p) ** 2) / 2.0
        corr_tetrachoric_r = math.cos(math.pi / (1.0 + odds_ratio_corrected**correction))

        # 6. Chambers' r
        if not math.isnan(odds_ratio) and odds_ratio != 1.0:
            chambers_r = (
                (odds_ratio + 1.0) / (odds_ratio - 1.0)
                - (2.0 * odds_ratio * math.log(odds_ratio)) / (odds_ratio - 1.0) ** 2
            )
        else:
            chambers_r = math.nan

        # 7. Wallis' swing d
        wallis_d_cols = a / (a + b) - c / (c + d) if (a + b) > 0 and (c + d) > 0 else math.nan
        wallis_d_rows = a / (a + c) - b / (b + d) if (a + c) > 0 and (b + d) > 0 else math.nan
        est_cols = (a + c) / n
        se_cols = math.sqrt(est_cols * (1.0 - est_cols) * (1.0 / (a + b) + 1.0 / (c + d))) if (a + b) > 0 and (c + d) > 0 else math.nan
        est_rows = (a + b) / n
        se_rows = math.sqrt(est_rows * (1.0 - est_rows) * (1.0 / (a + c) + 1.0 / (b + d))) if (a + c) > 0 and (b + d) > 0 else math.nan

        # ----------------------------------------------------------------
        # Confidence intervals
        # ----------------------------------------------------------------

        # Tetrachoric correlation CIs (Bonett, 2005 — SE via delta method on OR)
        se_or = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d) if min(a, b, c, d) > 0 else math.nan
        se_or_corr = math.sqrt(1.0 / (a + 0.5) + 1.0 / (b + 0.5) + 1.0 / (c + 0.5) + 1.0 / (d + 0.5))

        if not math.isnan(odds_ratio) and odds_ratio > 0:
            k = (
                math.pi
                * 0.5
                * odds_ratio**0.5
                * math.sin(math.pi / (1.0 + odds_ratio**0.5))
                / (1.0 + odds_ratio**0.5) ** 2
            )
        else:
            k = math.nan
        k_corr = (
            math.pi
            * correction
            * odds_ratio_corrected**correction
            * math.sin(math.pi / (1.0 + odds_ratio_corrected**correction))
            / (1.0 + odds_ratio_corrected**correction) ** 2
        )
        se_tetrachoric = k * se_or_corr if not math.isnan(k) else math.nan
        se_tetrachoric_corr = k_corr * se_or_corr

        if not math.isnan(tetrachoric_r) and not math.isnan(se_tetrachoric):
            ci_tet = (tetrachoric_r - z_crit * se_tetrachoric, tetrachoric_r + z_crit * se_tetrachoric)
        else:
            ci_tet = (math.nan, math.nan)
        ci_tet_corr = (
            corr_tetrachoric_r - z_crit * se_tetrachoric_corr,
            corr_tetrachoric_r + z_crit * se_tetrachoric_corr,
        )

        # Wallis swing d CIs
        ci_swing_cols = (
            (wallis_d_cols - z_crit * se_cols, wallis_d_cols + z_crit * se_cols)
            if not (math.isnan(wallis_d_cols) or math.isnan(se_cols))
            else (math.nan, math.nan)
        )
        ci_swing_rows = (
            (wallis_d_rows - z_crit * se_rows, wallis_d_rows + z_crit * se_rows)
            if not (math.isnan(wallis_d_rows) or math.isnan(se_rows))
            else (math.nan, math.nan)
        )

        # Phi variance (Bishop et al.) — for documentation / reporting
        p1p = (a + b) / n
        p2p = (c + d) / n
        pp1 = (a + c) / n
        pp2 = (b + d) / n
        prob_prod = p1p * p2p * pp1 * pp2
        if not math.isnan(phi) and prob_prod > 0:
            term1 = phi + 0.5 * phi**3
            term2 = ((p1p - p2p) * (pp1 - pp2)) / math.sqrt(prob_prod)
            term3 = 0.75 * phi**2 * (
                (p1p - p2p) ** 2 / (p1p * p2p) + (pp1 - pp2) ** 2 / (pp1 * pp2)
            )
            phi_variance = (1.0 / n) * (1.0 - phi**2 + term1 * term2 - term3)
        else:
            phi_variance = math.nan

        return ContingencyTable2x2Result(
            phi=round(float(phi), 6) if not math.isnan(phi) else math.nan,
            cramer_v=round(float(phi), 6) if not math.isnan(phi) else math.nan,
            bias_corrected_cramer=round(float(bias_corr_cramer), 6) if not math.isnan(bias_corr_cramer) else math.nan,
            max_corrected_phi=round(float(max_corr_phi), 6) if not math.isnan(max_corr_phi) else math.nan,
            odds_ratio=round(float(odds_ratio), 6) if not math.isnan(odds_ratio) else math.nan,
            tetrachoric_r=round(float(tetrachoric_r), 6) if not math.isnan(tetrachoric_r) else math.nan,
            tetrachoric_r_corrected=round(float(corr_tetrachoric_r), 6),
            chambers_r=round(float(chambers_r), 6) if not math.isnan(chambers_r) else math.nan,
            wallis_swing_d_cols=round(float(wallis_d_cols), 6) if not math.isnan(wallis_d_cols) else math.nan,
            wallis_swing_d_rows=round(float(wallis_d_rows), 6) if not math.isnan(wallis_d_rows) else math.nan,
            chi_square=round(float(chi_sq), 6) if not math.isnan(chi_sq) else math.nan,
            p_value=round(float(p_value), 6) if not math.isnan(p_value) else math.nan,
            n=int(n),
            confidence_level=confidence_level,
            ci_tetrachoric=(
                round(ci_tet[0], 6) if not math.isnan(ci_tet[0]) else math.nan,
                round(ci_tet[1], 6) if not math.isnan(ci_tet[1]) else math.nan,
            ),
            ci_tetrachoric_corrected=(round(ci_tet_corr[0], 6), round(ci_tet_corr[1], 6)),
            ci_swing_d_cols=(
                round(ci_swing_cols[0], 6) if not math.isnan(ci_swing_cols[0]) else math.nan,
                round(ci_swing_cols[1], 6) if not math.isnan(ci_swing_cols[1]) else math.nan,
            ),
            ci_swing_d_rows=(
                round(ci_swing_rows[0], 6) if not math.isnan(ci_swing_rows[0]) else math.nan,
                round(ci_swing_rows[1], 6) if not math.isnan(ci_swing_rows[1]) else math.nan,
            ),
            se_tetrachoric=round(float(se_tetrachoric), 6) if not math.isnan(se_tetrachoric) else math.nan,
            se_tetrachoric_corrected=round(float(se_tetrachoric_corr), 6),
            metadata={
                "a": int(a), "b": int(b), "c": int(c), "d": int(d),
                "row1_total": int(a + b), "row2_total": int(c + d),
                "col1_total": int(a + c), "col2_total": int(b + d),
                "phi_variance": round(float(phi_variance), 8) if not math.isnan(phi_variance) else math.nan,
                "odds_ratio_corrected": round(float(odds_ratio_corrected), 6),
                "tetrachoric_basic_approx": round(float(tetrachoric_basic), 6) if not math.isnan(tetrachoric_basic) else math.nan,
            },
        )
