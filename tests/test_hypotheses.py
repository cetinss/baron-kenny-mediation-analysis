"""Tests for the H1-H3 decision rules."""

from __future__ import annotations

import pytest

from bkmediation.hypotheses import HYPOTHESES, evaluate_hypotheses
from bkmediation.mediation import run_mediation

from .conftest import make_simulated

COVS = ["Cov1", "Cov2"]


@pytest.fixture(scope="module")
def supportive_result():
    """a > 0, b < 0 => c < 0 and a*b < 0: all three hypotheses should hold."""
    return run_mediation(make_simulated(a=0.004, b=-0.35, cp=-0.0009, seed=7),
                         covariates=COVS, n_boot=800, seed=42)


class TestStructure:
    def test_three_hypotheses_are_declared(self):
        assert [h.key for h in HYPOTHESES] == ["H1", "H2", "H3"]

    def test_each_hypothesis_states_a_direction_and_a_rule(self):
        for h in HYPOTHESES:
            assert h.expected_sign in (-1, 1)
            assert h.rule
            assert h.statement.endswith(".")

    def test_table_has_one_row_per_hypothesis_with_a_decision(self, supportive_result):
        table = evaluate_hypotheses(supportive_result)
        assert list(table["Hypothesis"]) == ["H1", "H2", "H3"]
        assert set(table["Decision"]) <= {"SUPPORTED", "NOT SUPPORTED"}
        for col in ("Estimate", "Estimate (per 100 mg)", "95% CI low",
                    "95% CI high", "p", "Decision rule", "CI method"):
            assert col in table.columns

    def test_h3_is_judged_on_the_bootstrap_interval(self, supportive_result):
        row = evaluate_hypotheses(supportive_result).set_index("Hypothesis").loc["H3"]
        assert row["CI method"] == "percentile bootstrap"
        assert row["Tested quantity"] == "a*b"


class TestSupportedCase:
    def test_all_three_are_supported_when_the_process_matches_them(self, supportive_result):
        table = evaluate_hypotheses(supportive_result).set_index("Hypothesis")
        assert table.loc["H1", "Decision"] == "SUPPORTED"
        assert table.loc["H2", "Decision"] == "SUPPORTED"
        assert table.loc["H3", "Decision"] == "SUPPORTED"

    def test_signs_follow_the_hypothesised_directions(self, supportive_result):
        table = evaluate_hypotheses(supportive_result).set_index("Hypothesis")
        assert table.loc["H1", "Estimate"] < 0
        assert table.loc["H2", "Estimate"] > 0
        assert table.loc["H3", "Estimate"] < 0

    def test_per_100_mg_column_rescales_the_estimate(self, supportive_result):
        table = evaluate_hypotheses(supportive_result).set_index("Hypothesis")
        for key in ("H1", "H2", "H3"):
            assert table.loc[key, "Estimate (per 100 mg)"] == pytest.approx(
                table.loc[key, "Estimate"] * 100)


class TestRejectedCases:
    def test_h2_is_not_supported_when_caffeine_lowers_stress(self):
        """Wrong sign must fail even though the coefficient is significant."""
        result = run_mediation(make_simulated(a=-0.004, seed=21), covariates=COVS,
                               n_boot=600, seed=42)
        table = evaluate_hypotheses(result).set_index("Hypothesis")
        assert table.loc["H2", "Decision"] == "NOT SUPPORTED"
        assert table.loc["H2", "Estimate"] < 0

    def test_h3_is_not_supported_without_an_indirect_path(self):
        result = run_mediation(make_simulated(b=0.0, cp=-0.002, n=2000, seed=11),
                               covariates=COVS, n_boot=600, seed=3)
        table = evaluate_hypotheses(result).set_index("Hypothesis")
        assert table.loc["H3", "Decision"] == "NOT SUPPORTED"

    def test_h1_is_not_supported_when_there_is_no_total_effect(self):
        """a*b and cp cancel, so c is ~0: H1 fails while H2 still holds."""
        result = run_mediation(
            make_simulated(a=0.004, b=-0.35, cp=+0.0014, n=3000, seed=31),
            covariates=COVS, n_boot=600, seed=42)
        table = evaluate_hypotheses(result).set_index("Hypothesis")
        assert table.loc["H1", "Decision"] == "NOT SUPPORTED"
        assert table.loc["H2", "Decision"] == "SUPPORTED"


class TestPublishedHypotheses:
    def test_all_three_hold_in_the_analytic_sample(self, analytic_result):
        table = evaluate_hypotheses(analytic_result).set_index("Hypothesis")
        assert list(table["Decision"]) == ["SUPPORTED"] * 3
