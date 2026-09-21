"""Tests for the clustering design-effect bound and the sample-definition table."""

from __future__ import annotations

import pytest

from bkmediation.clustering import design_effect_sensitivity, se_from_ci
from bkmediation.comparisons import (
    SAMPLE_DEFINITIONS,
    compare_sample_definitions,
    sample_definition_markdown,
)


class TestStandardErrorFromInterval:
    def test_recovers_the_standard_error_of_a_known_interval(self):
        # 95% interval of +/- 1.96 SE around an estimate (1.96 is the rounded
        # critical value, so allow the corresponding rounding slack).
        assert se_from_ci(-1.96, 1.96) == pytest.approx(1.0, rel=1e-4)

    def test_wider_interval_means_larger_standard_error(self):
        assert se_from_ci(-4, 4) > se_from_ci(-2, 2)

    def test_reversed_bounds_raise(self):
        with pytest.raises(ValueError, match="must not be below"):
            se_from_ci(1.0, -1.0)


class TestDesignEffectSensitivity:
    def test_borderline_estimate_needs_almost_no_clustering(self):
        """z just above 1.96: a tiny ICC suffices to overturn it."""
        s = design_effect_sensitivity(2.0, se=1.0, mean_cluster_size=5)
        assert s.critical_design_effect == pytest.approx((2.0 / 1.959964) ** 2, rel=1e-6)
        assert 0 < s.critical_icc < 0.02

    def test_strong_estimate_needs_implausible_clustering(self):
        s = design_effect_sensitivity(10.0, se=1.0, mean_cluster_size=5)
        assert s.critical_icc > 1
        assert "unattainable" in s.describe() or "would not make" in s.describe()

    def test_already_nonsignificant_estimate_is_reported_as_such(self):
        s = design_effect_sensitivity(1.0, se=1.0, mean_cluster_size=5)
        assert s.already_nonsignificant
        assert s.critical_design_effect == 1.0
        assert "already non-significant" in s.describe()

    def test_larger_clusters_need_a_smaller_icc(self):
        small = design_effect_sensitivity(4.0, se=1.0, mean_cluster_size=3)
        large = design_effect_sensitivity(4.0, se=1.0, mean_cluster_size=20)
        assert large.critical_icc < small.critical_icc

    def test_design_effect_does_not_depend_on_cluster_size(self):
        """DEFF is a property of the z statistic; only the ICC rescales it."""
        a = design_effect_sensitivity(4.0, se=1.0, mean_cluster_size=3)
        b = design_effect_sensitivity(4.0, se=1.0, mean_cluster_size=20)
        assert a.critical_design_effect == pytest.approx(b.critical_design_effect)

    def test_interval_input_matches_standard_error_input(self):
        from_se = design_effect_sensitivity(0.5, se=0.1, mean_cluster_size=4)
        from_ci = design_effect_sensitivity(
            0.5, ci=(0.5 - 1.959964 * 0.1, 0.5 + 1.959964 * 0.1), mean_cluster_size=4)
        assert from_ci.critical_icc == pytest.approx(from_se.critical_icc, rel=1e-6)

    def test_sign_of_the_estimate_does_not_matter(self):
        pos = design_effect_sensitivity(3.0, se=1.0, mean_cluster_size=5)
        neg = design_effect_sensitivity(-3.0, se=1.0, mean_cluster_size=5)
        assert pos.critical_icc == pytest.approx(neg.critical_icc)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError, match="either se or ci"):
            design_effect_sensitivity(1.0)
        with pytest.raises(ValueError, match="se must be positive"):
            design_effect_sensitivity(1.0, se=0.0)
        with pytest.raises(ValueError, match="at least 1"):
            design_effect_sensitivity(1.0, se=0.1, mean_cluster_size=0.5)


class TestSampleDefinitions:
    @pytest.fixture(scope="class")
    @staticmethod
    def table(analytic_sample):
        return compare_sample_definitions(n_boot=300)

    def test_full_raw_keeps_all_ten_thousand_rows(self, table):
        """The dataset processed in its raw state, with nothing excluded."""
        raw = table.set_index("Sample").loc["full_raw"]
        assert raw["n"] == 10000
        assert raw["Excluded from raw"] == 0

    def test_definitions_are_nested_in_decreasing_size(self, table):
        sizes = table.set_index("Sample")["n"]
        assert sizes["full_raw"] >= sizes["age_filtered"] >= sizes["primary"]

    def test_every_definition_reaches_the_same_conclusion(self, table):
        assert table["Conclusion"].nunique() == 1
        assert table["Conclusion"].iloc[0] == "PARTIAL MEDIATION"

    def test_indirect_effect_is_stable_across_definitions(self, table):
        """Cleaning moves the estimate in the third decimal place, not beyond."""
        spread = table["a*b"].max() - table["a*b"].min()
        assert spread < 1e-4
        assert (table["a*b CI high"] < 0).all()

    def test_all_four_definitions_are_reported(self, table):
        assert list(table["Sample"]) == [key for key, _, _ in SAMPLE_DEFINITIONS]

    def test_markdown_states_the_raw_sample_size(self, table):
        text = sample_definition_markdown(table)
        assert "10,000" in text
        assert "raw state" in text
