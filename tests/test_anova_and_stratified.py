"""Unit tests for stratified contingency and two-way mixed ANOVA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esek.calculators.anova import (
    MixedANOVAEffect,
    TwoWayMixedANOVA,
    TwoWayMixedANOVAResult,
)
from esek.calculators.stratified_contingency import (
    StratifiedTwoByTwo,
    StratifiedTwoByTwoResult,
)
from esek.core import InvalidInputError


def _mixed_anova_frame(seed: int = 1) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for subject in range(12):
        group = "A" if subject < 6 else "B"
        for time, mu in (("pre", 0.0), ("post", 1.0 if group == "A" else 0.2)):
            rows.append(
                {
                    "subject": subject,
                    "group": group,
                    "time": time,
                    "score": float(rng.normal(mu, 1.0)),
                }
            )
    return pd.DataFrame(rows)


class TestStratifiedTwoByTwoUnit:
    def test_from_tables_result_type(self):
        result = StratifiedTwoByTwo.from_tables(
            [[[10, 5], [3, 12]], [[8, 6], [4, 10]]]
        )
        assert isinstance(result, StratifiedTwoByTwoResult)

    def test_from_tables_values(self):
        result = StratifiedTwoByTwo.from_tables(
            [[[10, 5], [3, 12]], [[8, 6], [4, 10]]]
        )
        assert result.n_strata == 2
        assert result.common_odds_ratio > 1.0
        lo, hi = result.common_odds_ratio_ci
        assert lo < result.common_odds_ratio < hi
        assert result.risk_ratio > 0
        assert 0.0 <= result.test_null_or_p_value <= 1.0
        assert 0.0 <= result.test_equal_odds_p_value <= 1.0

    def test_from_data(self):
        tables = [[[5, 2], [1, 6]], [[4, 3], [2, 5]]]
        result = StratifiedTwoByTwo.from_tables(tables)
        assert result.n_strata == 2

        v1, v2, s = [], [], []
        for level, table in enumerate(tables):
            for i in range(2):
                for j in range(2):
                    count = table[i][j]
                    v1.extend([f"r{i}"] * count)
                    v2.extend([f"c{j}"] * count)
                    s.extend([f"stratum{level}"] * count)
        from_data = StratifiedTwoByTwo.from_data(v1, v2, s)
        assert from_data.n_strata == 2
        assert from_data.common_odds_ratio == pytest.approx(
            result.common_odds_ratio, rel=1e-6
        )

    def test_invalid_table_shape(self):
        with pytest.raises(InvalidInputError, match="2×2"):
            StratifiedTwoByTwo.from_tables([[[1, 2, 3], [4, 5, 6]]])

    def test_negative_counts(self):
        with pytest.raises(InvalidInputError, match="non-negative"):
            StratifiedTwoByTwo.from_tables([[[-1, 2], [3, 4]]])

    def test_empty_tables(self):
        with pytest.raises(InvalidInputError):
            StratifiedTwoByTwo.from_tables([])

    def test_from_data_length_mismatch(self):
        with pytest.raises(InvalidInputError, match="equal length"):
            StratifiedTwoByTwo.from_data([1, 0], [1], [0, 1])

    def test_incomplete_stratum_raises(self):
        with pytest.raises(InvalidInputError, match="complete 2×2"):
            StratifiedTwoByTwo.from_data(
                variable1=["a", "a", "b", "b"],
                variable2=["x", "x", "x", "x"],  # only one column level
                stratum=["s", "s", "s", "s"],
            )


class TestTwoWayMixedANOVAUnit:
    def test_returns_result_type(self):
        result = TwoWayMixedANOVA.from_data(
            _mixed_anova_frame(),
            dependent="score",
            subject="subject",
            within="time",
            between="group",
            include_pairwise=False,
        )
        assert isinstance(result, TwoWayMixedANOVAResult)

    def test_effects_and_eta_ci(self):
        result = TwoWayMixedANOVA.from_data(
            _mixed_anova_frame(),
            dependent="score",
            subject="subject",
            within="time",
            between="group",
            include_pairwise=False,
        )
        assert len(result.effects) >= 2
        assert all(isinstance(effect, MixedANOVAEffect) for effect in result.effects)
        sources = {effect.source.lower() for effect in result.effects}
        assert any("time" in s for s in sources)
        assert any("group" in s for s in sources)
        for effect in result.effects:
            assert effect.partial_eta_squared >= 0.0
            if effect.eta_squared_ci is not None:
                lo, hi = effect.eta_squared_ci.ci_partial_eta_sq_fleishman
                assert lo <= hi

    def test_pairwise_included(self):
        result = TwoWayMixedANOVA.from_data(
            _mixed_anova_frame(),
            dependent="score",
            subject="subject",
            within="time",
            between="group",
            include_pairwise=True,
        )
        assert result.pairwise is not None
        assert "factor" in result.pairwise.columns
        assert len(result.pairwise) >= 1

    def test_missing_column_raises(self):
        with pytest.raises(InvalidInputError, match="missing"):
            TwoWayMixedANOVA.from_data(
                _mixed_anova_frame(),
                dependent="missing",
                subject="subject",
                within="time",
                between="group",
            )

    def test_non_dataframe_raises(self):
        with pytest.raises(InvalidInputError, match="DataFrame"):
            TwoWayMixedANOVA.from_data(
                [[1, 2]],  # type: ignore[arg-type]
                dependent="score",
                subject="subject",
                within="time",
                between="group",
            )

    def test_invalid_confidence(self):
        with pytest.raises(InvalidInputError):
            TwoWayMixedANOVA.from_data(
                _mixed_anova_frame(),
                dependent="score",
                subject="subject",
                within="time",
                between="group",
                confidence_level=1.2,
            )
