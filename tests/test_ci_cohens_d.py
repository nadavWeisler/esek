from __future__ import annotations

import math

import pytest

from esek import CohensDCI
from esek.confidence_intervals import CohensDCIResult


class TestCohensDCI:
    def test_one_sample_z_known_value(self):
        result = CohensDCI.one_sample_z(d=0.5, n=25)
        expected_se = math.sqrt((1 / 25) + ((0.5**2) / (2 * 25)))
        assert isinstance(result, CohensDCIResult)
        assert result.se == pytest.approx(expected_se, abs=1e-12)
        assert result.ci_low == pytest.approx(0.0842288527, abs=1e-6)
        assert result.ci_high == pytest.approx(0.9157711473, abs=1e-6)

    def test_paired_z_negative_d(self):
        result = CohensDCI.paired_z(d=-0.4, n=25)
        assert isinstance(result, CohensDCIResult)
        assert result.design == "paired"
        assert result.ci_low < result.ci_high
        assert result.ci_low <= result.d <= result.ci_high
        assert result.d < 0

    def test_independent_z_known_value(self):
        result = CohensDCI.independent_z(d=0.5, n1=20, n2=22)
        expected_se = math.sqrt(((20 + 22) / (20 * 22)) + (0.5**2 / (2 * (20 + 22))))
        assert isinstance(result, CohensDCIResult)
        assert result.se == pytest.approx(expected_se, abs=1e-12)
        assert result.ci_low == pytest.approx(-0.1149126920, abs=1e-6)
        assert result.ci_high == pytest.approx(1.1149126920, abs=1e-6)

    def test_one_sample_t_central_returns_se_variants(self):
        results = CohensDCI.one_sample_t_central(d=0.5, n=25)
        assert isinstance(results, list)
        assert len(results) == 7
        assert all(isinstance(result, CohensDCIResult) for result in results)
        assert [result.metadata["se_method"] for result in results] == [
            "true",
            "morris",
            "hedges",
            "hedges_olkin",
            "mle",
            "large_n",
            "small_n",
        ]
        assert results[0].ci_low == pytest.approx(0.0627798541, abs=1e-6)
        assert results[0].ci_high == pytest.approx(0.9372201459, abs=1e-6)

    def test_one_sample_t_ncp_known_value(self):
        result = CohensDCI.one_sample_t_ncp(d=0.5, n=25)
        assert isinstance(result, CohensDCIResult)
        assert result.ci_low == pytest.approx(0.1079439340, abs=1e-6)
        assert result.ci_high == pytest.approx(0.9873340520, abs=1e-6)

    def test_one_sample_t_pivotal_known_value(self):
        result = CohensDCI.one_sample_t_pivotal(t_stat=2.5, n=25)
        assert isinstance(result, CohensDCIResult)
        assert result.d == pytest.approx(0.5, abs=1e-12)
        assert result.ci_low == pytest.approx(0.0788879395, abs=1e-6)
        assert result.ci_high == pytest.approx(0.9118652344, abs=1e-6)

    def test_paired_t_central_returns_se_variants(self):
        results = CohensDCI.paired_t_central(d=0.5, n=25)
        assert len(results) == 7
        assert all(result.design == "paired" for result in results)
        assert results[0].ci_low == pytest.approx(0.0627798541, abs=1e-6)
        assert results[0].ci_high == pytest.approx(0.9372201459, abs=1e-6)

    def test_paired_t_pooled_central_returns_se_variants(self):
        results = CohensDCI.paired_t_pooled_central(d=0.5, n=25, r=0.4)
        assert len(results) == 7
        assert all(result.metadata["variant"] == "pooled" for result in results)
        assert results[0].ci_low == pytest.approx(0.0259884866, abs=1e-6)
        assert results[0].ci_high == pytest.approx(0.9740115134, abs=1e-6)

    def test_paired_t_ncp_known_value(self):
        result = CohensDCI.paired_t_ncp(d=0.5, n=25)
        assert isinstance(result, CohensDCIResult)
        assert result.ci_low == pytest.approx(0.1079439340, abs=1e-6)
        assert result.ci_high == pytest.approx(0.9873340520, abs=1e-6)

    def test_paired_t_morris_known_value(self):
        result = CohensDCI.paired_t_morris(d=0.5, n=25, r=0.4)
        assert isinstance(result, CohensDCIResult)
        assert result.ci_low == pytest.approx(0.0409837302, abs=1e-6)
        assert result.ci_high == pytest.approx(0.9590162698, abs=1e-6)

    def test_paired_t_mag_known_value(self):
        result = CohensDCI.paired_t_mag(d=0.5, sd1=1.2, sd2=1.0, n=25, r=0.4)
        assert isinstance(result, CohensDCIResult)
        assert result.ci_low == pytest.approx(0.0373778162, abs=1e-6)
        assert result.ci_high == pytest.approx(0.9899827663, abs=1e-6)
        assert result.metadata["corrected_correlation"] == pytest.approx(0.3934426230, abs=1e-8)

    def test_paired_t_algina_keselman_known_value(self):
        result = CohensDCI.paired_t_algina_keselman(
            d=0.5,
            sd1=1.2,
            sd2=1.0,
            n=25,
            r=0.4,
        )
        assert isinstance(result, CohensDCIResult)
        assert result.ci_low == pytest.approx(0.0412139893, abs=1e-6)
        assert result.ci_high == pytest.approx(0.9494018555, abs=1e-6)

    def test_independent_t_central_returns_se_variants(self):
        results = CohensDCI.independent_t_central(d=0.5, n1=20, n2=22)
        assert len(results) == 7
        assert all(result.design == "independent" for result in results)
        assert results[0].ci_low == pytest.approx(-0.1318201563, abs=1e-6)
        assert results[0].ci_high == pytest.approx(1.1318201563, abs=1e-6)

    def test_independent_t_pivotal_known_value(self):
        result = CohensDCI.independent_t_pivotal(t_stat=2.2, n1=20, n2=22)
        assert isinstance(result, CohensDCIResult)
        assert result.d == pytest.approx(0.6797058187, abs=1e-6)
        assert result.ci_low == pytest.approx(0.0522308121, abs=1e-6)
        assert result.ci_high == pytest.approx(1.2991740368, abs=1e-6)

    def test_independent_t_ncp_known_value(self):
        result = CohensDCI.independent_t_ncp(t_stat=2.2, n1=20, n2=22)
        assert isinstance(result, CohensDCIResult)
        assert result.ci_low == pytest.approx(0.0522308121, abs=1e-6)
        assert result.ci_high == pytest.approx(1.2991740368, abs=1e-6)
        assert result.metadata["ncp_low"] < result.metadata["ncp_high"]

    def test_zero_effect_gives_symmetric_ncp_interval(self):
        result = CohensDCI.one_sample_t_ncp(d=0.0, n=20)
        assert isinstance(result, CohensDCIResult)
        assert result.ci_low == pytest.approx(-result.ci_high, abs=1e-10)
        assert result.d == 0.0

    def test_negative_t_preserves_sign_order(self):
        result = CohensDCI.independent_t_pivotal(t_stat=-2.2, n1=20, n2=22)
        assert result.d < 0
        assert result.ci_low < result.ci_high
        assert result.ci_low <= result.d <= result.ci_high
        assert result.ci_high < 0

    @pytest.mark.parametrize(
        "call",
        [
            lambda: CohensDCI.one_sample_z(d=0.2, n=1),
            lambda: CohensDCI.one_sample_t_central(d=0.2, n=3),
            lambda: CohensDCI.paired_t_central(d=0.2, n=3),
            lambda: CohensDCI.independent_z(d=0.2, n1=1, n2=10),
            lambda: CohensDCI.independent_t_central(d=0.2, n1=2, n2=2),
            lambda: CohensDCI.paired_t_mag(d=0.2, sd1=0.0, sd2=1.0, n=10, r=0.2),
        ],
    )
    def test_invalid_inputs_raise(self, call):
        with pytest.raises(ValueError):
            call()

    @pytest.mark.parametrize(
        "call",
        [
            lambda: CohensDCI.one_sample_z(d=0.2, n=10, confidence_level=1.1),
            lambda: CohensDCI.one_sample_t_ncp(d=0.2, n=10, confidence_level=0.0),
            lambda: CohensDCI.paired_t_morris(d=0.2, n=10, r=0.2, confidence_level=-0.5),
            lambda: CohensDCI.independent_t_pivotal(t_stat=1.5, n1=10, n2=10, confidence_level=2.0),
        ],
    )
    def test_invalid_confidence_level_raises(self, call):
        with pytest.raises(ValueError, match="confidence_level"):
            call()

    def test_top_level_export_available(self):
        result = CohensDCI.one_sample_z(d=0.3, n=30)
        assert isinstance(result, CohensDCIResult)
