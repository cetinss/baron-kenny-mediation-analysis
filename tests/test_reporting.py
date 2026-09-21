"""Tests for the report writers and the descriptive/diagnostic helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from bkmediation.descriptives import category_counts, describe, key_correlations
from bkmediation.diagnostics import residual_diagnostics, run_diagnostics, vif_table
from bkmediation.reporting import analysis_markdown, write_analysis_outputs

from .conftest import make_simulated


class TestDescriptives:
    def test_describe_reports_one_row_per_variable(self, analytic_sample):
        table = describe(analytic_sample.df)
        assert len(table) == 7
        assert {"Mean", "SD", "Median", "Min", "Max", "Skewness", "Kurtosis"} <= set(table.columns)

    def test_category_counts_percentages_sum_to_100(self, analytic_sample):
        counts = category_counts(analytic_sample.df, "Gender")
        assert counts["%"].sum() == pytest.approx(100.0, abs=0.2)
        assert set(counts["Category"]) == {"Male", "Female", "Other"}

    def test_key_correlations_cover_the_three_model_paths(self, analytic_sample):
        table = key_correlations(analytic_sample.df)
        assert len(table) == 3
        assert table["r"].between(-1, 1).all()


class TestDiagnostics:
    def test_vif_table_excludes_the_intercept(self, analytic_sample):
        preds = ["Caffeine_mg", "Stress_Score", *analytic_sample.covariates]
        table = vif_table(analytic_sample.df, preds)
        assert "const" not in set(table["Variable"])
        assert len(table) == len(preds)

    def test_residual_diagnostics_return_the_expected_statistics(self, simulated, sim_covariates):
        stats = residual_diagnostics(simulated["Sleep_Hours"],
                                     simulated[["Caffeine_mg", *sim_covariates]])
        for key in ("durbin_watson", "breusch_pagan_p", "jarque_bera_p", "r2"):
            assert key in stats
        assert 0 <= stats["r2"] <= 1
        assert 0 < stats["durbin_watson"] < 4

    def test_run_diagnostics_covers_all_three_models(self, analytic_sample):
        diagnostics = run_diagnostics(analytic_sample)
        assert set(diagnostics) == {"vif", "total_model", "mediated_model", "mediator_model"}


class TestAnalysisReport:
    @pytest.fixture(scope="class")
    @staticmethod
    def written(tmp_path_factory, analytic_sample, analytic_result):
        out = tmp_path_factory.mktemp("reports")
        copy_dir = tmp_path_factory.mktemp("outputs")
        # markdown_copy_dir defaults to the project's outputs/ directory; point
        # it at a temporary one so running the tests never rewrites the
        # committed reports with low-resample test values.
        return write_analysis_outputs(analytic_sample, analytic_result, out,
                                      markdown_copy_dir=copy_dir), (out, copy_dir)

    def test_every_expected_artefact_is_written(self, written):
        files, (out, _copy_dir) = written
        for name in ("descriptive_statistics.csv", "correlation_matrix.csv",
                     "mediation_results.csv", "mediation_paths_table.csv",
                     "mediation_effects_table.csv", "hypothesis_tests.csv",
                     "assumption_diagnostics.csv", "residual_diagnostics.csv",
                     "sample_exclusions.csv", "analysis_report.md"):
            assert name in files
            assert (out / name).exists()
            assert (out / name).stat().st_size > 0

    def test_results_csv_contains_the_confidence_limits(self, written):
        _, (out, _copy_dir) = written
        row = pd.read_csv(out / "mediation_results.csv").iloc[0]
        for path in ("a", "b", "c", "cp"):
            assert f"{path}_ci_lo" in row.index
            assert f"{path}_ci_hi" in row.index
            assert f"{path}_boot_lo" in row.index

    def test_report_states_the_hypotheses_and_the_scale(self, analytic_sample, analytic_result):
        text = analysis_markdown(analytic_sample, analytic_result)
        assert "H1" in text and "H2" in text and "H3" in text
        assert "per 100 mg" in text
        assert "95% CI" in text

    def test_report_refuses_to_sell_f2_as_a_mediation_effect_size(
            self, analytic_sample, analytic_result):
        text = analysis_markdown(analytic_sample, analytic_result)
        assert "not** a mediation effect size" in text or \
               "not a mediation effect size" in text

    def test_report_carries_the_synthetic_data_notice(self, analytic_sample, analytic_result):
        text = analysis_markdown(analytic_sample, analytic_result)
        assert "Synthetic data" in text
        assert "not evidence of a biological mechanism" in text

    def test_the_markdown_copy_goes_where_it_is_told(self, written):
        files, (_out, copy_dir) = written
        assert (copy_dir / "analysis_report.md").exists()
        assert "outputs/analysis_report.md" in files

    def test_report_works_for_a_non_synthetic_sample(self, tmp_path):
        """The notice must disappear when the data are flagged as real."""
        from bkmediation.data import AnalyticSample
        from bkmediation.mediation import run_mediation

        frame = make_simulated(n=500, seed=99)
        sample = AnalyticSample(df=frame, covariates=["Cov1", "Cov2"],
                                gender_scheme="three_level", n_raw=500,
                                n_after_age=500, n_after_missing=500,
                                is_synthetic=False, source="simulated.csv")
        result = run_mediation(sample, n_boot=200, seed=1)
        text = analysis_markdown(sample, result)
        assert "Synthetic data" not in text
        assert "real-world" in text
