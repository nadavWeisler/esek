"""Stratified 2×2 contingency-table differences.

Migrated from
``stats/Differecnes/DifferencesBetweenCorrelations/Contingency Tables/Diff_Contingency_Tables.ipynb``
on the ``dev`` branch.  Uses ``statsmodels.stats.contingency_tables.StratifiedTable``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.contingency_tables import StratifiedTable

from ..core import InvalidInputError
from ..core.validation import validate_confidence_level, validate_non_empty


@dataclass(frozen=True)
class StratifiedTwoByTwoResult:
    """Mantel–Haenszel stratified 2×2 analysis results."""

    n_strata: int
    confidence_level: float
    common_odds_ratio: float
    common_odds_ratio_ci: tuple[float, float]
    common_log_odds_ratio: float
    common_log_odds_ratio_se: float
    risk_ratio: float
    risk_ratio_ci: tuple[float, float]
    test_null_or_statistic: float
    test_null_or_p_value: float
    test_equal_odds_statistic: float
    test_equal_odds_p_value: float
    metadata: dict[str, Any] = field(default_factory=dict)


class StratifiedTwoByTwo:
    """Stratified analysis of multiple 2×2 tables."""

    @staticmethod
    def from_tables(
        tables: Sequence[np.ndarray | list[list[float]]],
        confidence_level: float = 0.95,
    ) -> StratifiedTwoByTwoResult:
        """Analyze a sequence of stratum-specific 2×2 tables.

        Parameters
        ----------
        tables:
            Iterable of 2×2 contingency tables, one per stratum.
        confidence_level:
            Confidence level in ``(0, 1)``.
        """
        validate_confidence_level(confidence_level)
        validate_non_empty(tables, name="tables")
        prepared: list[list[list[float]]] = []
        for i, table in enumerate(tables):
            arr = np.asarray(table, dtype=float)
            if arr.shape != (2, 2):
                raise InvalidInputError(
                    f"tables[{i}] must be a 2×2 matrix, got shape {arr.shape}."
                )
            if np.any(arr < 0):
                raise InvalidInputError(f"tables[{i}] must be non-negative.")
            prepared.append(arr.tolist())

        return _from_stratified_table(
            StratifiedTable(prepared),
            confidence_level=confidence_level,
            n_strata=len(prepared),
            metadata={"tables": prepared},
        )

    @staticmethod
    def from_data(
        variable1: Sequence[Any],
        variable2: Sequence[Any],
        stratum: Sequence[Any],
        confidence_level: float = 0.95,
    ) -> StratifiedTwoByTwoResult:
        """Build per-stratum 2×2 tables from raw categorical data.

        Parameters
        ----------
        variable1, variable2:
            Paired categorical outcomes (two levels each).
        stratum:
            Stratum labels aligned with the outcomes.
        confidence_level:
            Confidence level in ``(0, 1)``.
        """
        validate_confidence_level(confidence_level)
        validate_non_empty(variable1, name="variable1")
        if not (len(variable1) == len(variable2) == len(stratum)):
            raise InvalidInputError(
                "'variable1', 'variable2', and 'stratum' must have equal length."
            )

        df = pd.DataFrame(
            {
                "variable1": list(variable1),
                "variable2": list(variable2),
                "stratum": list(stratum),
            }
        )
        tables: list[np.ndarray] = []
        for level, group in df.groupby("stratum", sort=False):
            table = pd.crosstab(group["variable1"], group["variable2"]).values
            if table.shape != (2, 2):
                raise InvalidInputError(
                    f"Stratum {level!r} does not form a complete 2×2 table "
                    f"(got shape {table.shape})."
                )
            tables.append(table.astype(float))

        return StratifiedTwoByTwo.from_tables(tables, confidence_level=confidence_level)


def _from_stratified_table(
    stratified: StratifiedTable,
    *,
    confidence_level: float,
    n_strata: int,
    metadata: dict[str, Any],
) -> StratifiedTwoByTwoResult:
    alpha = 1.0 - confidence_level
    oddsratio = float(stratified.oddsratio_pooled)
    log_or = float(stratified.logodds_pooled)
    log_or_se = float(stratified.logodds_pooled_se)
    or_ci_raw = stratified.oddsratio_pooled_confint(alpha=alpha)
    or_ci = (float(or_ci_raw[0]), float(or_ci_raw[1]))

    rr = float(stratified.riskratio_pooled)
    # Approximate log-risk-ratio SE from normal theory on the OR CI width when
    # statsmodels does not expose a dedicated SE attribute.
    z = float(norm.ppf(1.0 - alpha / 2.0))
    if rr > 0 and np.isfinite(rr):
        # Use delta-method proxy via log-OR SE scaled by RR/OR when both positive.
        if oddsratio > 0 and np.isfinite(log_or_se):
            rr_log_se = log_or_se  # shared large-sample log-scale uncertainty proxy
            rr_ci = (
                float(np.exp(np.log(rr) - z * rr_log_se)),
                float(np.exp(np.log(rr) + z * rr_log_se)),
            )
        else:
            rr_ci = (float("nan"), float("nan"))
    else:
        rr_ci = (float("nan"), float("nan"))

    null_test = stratified.test_null_odds()
    equal_test = stratified.test_equal_odds()

    return StratifiedTwoByTwoResult(
        n_strata=n_strata,
        confidence_level=float(confidence_level),
        common_odds_ratio=oddsratio,
        common_odds_ratio_ci=or_ci,
        common_log_odds_ratio=log_or,
        common_log_odds_ratio_se=log_or_se,
        risk_ratio=rr,
        risk_ratio_ci=rr_ci,
        test_null_or_statistic=float(null_test.statistic),
        test_null_or_p_value=float(null_test.pvalue),
        test_equal_odds_statistic=float(equal_test.statistic),
        test_equal_odds_p_value=float(equal_test.pvalue),
        metadata=metadata,
    )
