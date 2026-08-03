"""Tests for agreement measures, stratified contingency, and mixed ANOVA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from esek.calculators.agreements import (
    AikenAlpha,
    BhapkarTest,
    CohensKappa,
    FleissKappa,
    GwetAC,
    IntraclassCorrelation,
    KendallsW,
    KrippendorffAlpha,
)
from esek.calculators.anova import TwoWayMixedANOVA
from esek.calculators.correlations import OrdinalByOrdinal, OrdinalByOrdinalResult
from esek.calculators.stratified_contingency import StratifiedTwoByTwo
from esek.core import InvalidInputError


class TestCohensKappa:
    def test_perfect_agreement(self):
        table = np.array([[10, 0], [0, 10]], dtype=float)
        result = CohensKappa.from_table(table)
        assert result.kappa == pytest.approx(1.0)

    def test_known_table(self):
        table = np.array([[4, 4, 2], [2, 6, 0], [0, 2, 0]], dtype=float)
        result = CohensKappa.from_table(table)
        assert -1.0 <= result.kappa <= 1.0
        assert result.ci[0] <= result.kappa <= result.ci[1]

    def test_weighted(self):
        table = np.array([[4, 4, 2], [2, 6, 0], [0, 2, 0]], dtype=float)
        linear = CohensKappa.from_table(table, weight_type="linear")
        quadratic = CohensKappa.from_table(table, weight_type="quadratic")
        assert linear.weight_type == "linear"
        assert quadratic.weight_type == "quadratic"

    def test_non_square_raises(self):
        with pytest.raises(InvalidInputError):
            CohensKappa.from_table([[1, 2, 3], [4, 5, 6]])


class TestFleissKappa:
    def test_fleiss_demo_matrix(self):
        fleiss = np.array(
            [
                [0, 0, 0, 6, 0],
                [0, 3, 0, 0, 3],
                [0, 1, 4, 0, 1],
                [0, 0, 0, 0, 6],
                [0, 3, 0, 3, 0],
                [2, 0, 4, 0, 0],
                [0, 0, 4, 0, 2],
                [2, 0, 3, 1, 0],
                [2, 0, 0, 4, 0],
                [0, 0, 0, 0, 6],
                [1, 0, 0, 5, 0],
                [1, 1, 0, 4, 0],
                [0, 3, 3, 0, 0],
                [1, 0, 0, 5, 0],
                [0, 2, 0, 3, 1],
                [0, 0, 5, 0, 1],
                [3, 0, 0, 1, 2],
                [5, 1, 0, 0, 0],
                [0, 2, 0, 4, 0],
                [1, 0, 2, 0, 3],
                [0, 0, 0, 0, 6],
                [0, 1, 0, 5, 0],
                [0, 2, 0, 1, 3],
                [2, 0, 0, 4, 0],
                [1, 0, 0, 4, 1],
                [0, 5, 0, 1, 0],
                [4, 0, 0, 0, 2],
                [0, 2, 0, 4, 0],
                [1, 0, 5, 0, 0],
                [0, 0, 0, 0, 6],
            ],
            dtype=float,
        )
        result = FleissKappa.from_counts(fleiss)
        assert result.n_raters == 6
        assert -1.0 <= result.fleiss_kappa <= 1.0
        assert result.ci[0] <= result.fleiss_kappa <= result.ci[1]


class TestBhapkarAndKendall:
    def test_bhapkar(self):
        table = np.array([[4, 4, 2], [2, 6, 0], [0, 2, 0]], dtype=float)
        result = BhapkarTest.from_table(table)
        assert result.degrees_of_freedom == 2
        assert 0.0 <= result.p_value <= 1.0

    def test_kendalls_w(self):
        data = np.array(
            [
                [10.4, 7.4, 17.0],
                [10.8, 7.6, 17.0],
                [11.1, 7.9, 20.0],
                [10.2, 7.2, 14.5],
                [10.3, 7.4, 15.5],
                [10.2, 7.1, 13.0],
                [10.7, 7.4, 19.5],
                [10.5, 7.2, 16.0],
                [10.8, 7.8, 21.0],
                [11.2, 7.7, 20.0],
                [10.6, 7.8, 18.0],
                [11.4, 8.3, 22.0],
            ]
        )
        result = KendallsW.from_data(data)
        assert 0.0 <= result.w_tie_corrected <= 1.0 + 1e-9


class TestICCGwetKrippendorffAiken:
    def test_icc(self):
        data = np.array(
            [
                9.30, 9.70, 8.90,
                8.90, 8.80, 8.10,
                8.00, 8.10, 7.30,
                9.10, 9.00, 8.20,
                9.10, 9.20, 8.30,
                8.90, 9.00, 7.70,
                8.30, 8.70, 8.10,
                9.30, 9.40, 8.20,
                9.40, 9.80, 9.40,
            ]
        ).reshape(9, 3)
        result = IntraclassCorrelation.from_data(data)
        assert result.icc2.icc > 0
        assert result.icc2.ci[0] <= result.icc2.icc <= result.icc2.ci[1]

    def test_gwet_and_krippendorff(self):
        ratings = [
            [1, 2, np.nan, 3, 4],
            [2, 2, 3, 2, np.nan],
            [1, 1, 1, 2, 2],
            [4, 4, 3, 4, 2],
            [3, 3, 4, np.nan, 3],
            [2, np.nan, 1, 1, np.nan],
            [3, 3, 3, 4, np.nan],
            [4, 3, 4, 3, 4],
            [np.nan, 3, np.nan, np.nan, np.nan],
        ]
        gwet = GwetAC.from_data(ratings, weight_method="ordinal")
        kripp = KrippendorffAlpha.from_data(ratings, weight_method="ordinal")
        assert -1.0 <= gwet.ac <= 1.0
        assert -1.0 <= kripp.alpha <= 1.0

    def test_aiken(self):
        flat = [55, 10, 2, 6, 4, 10, 2, 5, 6]
        result = AikenAlpha.from_table(flat)
        assert -1.0 <= result.alpha <= 1.0
        assert result.standard_error > 0


class TestStratifiedAndANOVA:
    def test_stratified_from_tables(self):
        tables = [[[10, 5], [3, 12]], [[8, 6], [4, 10]]]
        result = StratifiedTwoByTwo.from_tables(tables)
        assert result.n_strata == 2
        assert result.common_odds_ratio > 0
        assert result.common_odds_ratio_ci[0] < result.common_odds_ratio_ci[1]

    def test_mixed_anova(self):
        rows = []
        rng = np.random.default_rng(1)
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
        data = pd.DataFrame(rows)
        result = TwoWayMixedANOVA.from_data(
            data,
            dependent="score",
            subject="subject",
            within="time",
            between="group",
            include_pairwise=True,
        )
        assert len(result.effects) >= 2
        assert result.pairwise is not None


class TestOrdinalTypedAPI:
    def test_typed_from_data(self):
        x = [1, 2, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1]
        y = [1, 2, 2, 3, 1, 3, 2, 2, 4, 2, 3, 1]
        result = OrdinalByOrdinal.from_data(x, y, confidence_level=0.95, n_bootstrap=10)
        assert isinstance(result, OrdinalByOrdinalResult)
        assert result.n_bootstrap == 10

    def test_legacy_params_still_works(self):
        result = OrdinalByOrdinal.from_data(
            params={
                "Variable 1": [1, 2, 3, 2, 1, 3, 4, 2, 1, 3],
                "Variable 2": [1, 1, 3, 2, 2, 3, 4, 3, 1, 2],
                "Confidence Level": 95,
                "Number Of Bootstraps Samples": 10,
            }
        )
        assert isinstance(result, dict)
