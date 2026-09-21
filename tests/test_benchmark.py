"""Benchmark tests: timing shape, scaling behaviour and cross-software agreement.

The agreement tests are the substantive ones. They check that the indirect
effect this package reports is the same number that an independent
statsmodels fit and (when installed) pingouin report for the same model, so
the claim of numerical correctness rests on comparison rather than assertion.
"""

from __future__ import annotations

import numpy as np
import pytest

from bkmediation.benchmark import (
    INDIRECT_COL,
    benchmark_markdown,
    compare_software,
    scalability_curve,
    time_pipeline,
)

pytestmark = pytest.mark.slow


class TestTiming:
    def test_every_stage_is_timed(self, analytic_sample):
        table = time_pipeline(n_boot=100, repeats=1)
        stages = set(table["Stage"])
        assert "load_and_clean" in stages
        assert "fit_paths" in stages
        assert "TOTAL (single analysis)" in stages
        assert (table["Median seconds"].dropna() >= 0).all()

    def test_fitting_is_far_cheaper_than_bootstrapping(self, analytic_sample):
        table = time_pipeline(n_boot=200, repeats=1).set_index("Stage")
        assert table.loc["fit_paths", "Median seconds"] < \
               table.loc["bootstrap_200", "Median seconds"]


class TestScalability:
    def test_curve_covers_the_requested_sizes(self, analytic_sample):
        table = scalability_curve(sizes=[500, 2000], n_boot=50)
        assert list(table["n"]) == [500, 2000]
        assert (table["Bootstrap seconds (B=50)"] > 0).all()

    def test_larger_samples_are_simulated_beyond_the_dataset(self, analytic_sample):
        table = scalability_curve(sizes=[500, 20000], n_boot=20)
        assert table.iloc[0]["Source"].startswith("analytic sample")
        assert table.iloc[1]["Source"] == "simulated"

    def test_cost_per_replicate_grows_with_sample_size(self, analytic_sample):
        table = scalability_curve(sizes=[500, 8000], n_boot=100)
        per_rep = table["Bootstrap us per replicate"].to_numpy()
        assert per_rep[1] > per_rep[0]


class TestCrossSoftwareAgreement:
    @pytest.fixture(scope="class")
    @staticmethod
    def table(analytic_sample):
        return compare_software(analytic_sample, n_boot=300)

    def test_the_comparison_includes_an_independent_statsmodels_fit(self, table):
        assert any("statsmodels OLS" in s for s in table["Software"])

    def test_closed_form_statsmodels_fit_agrees_to_machine_precision(self, table):
        """Same least-squares problem solved by a different route."""
        own = table.iloc[0][INDIRECT_COL]
        closed = table[table["Software"].str.startswith("statsmodels OLS")].iloc[0]
        assert closed[INDIRECT_COL] == pytest.approx(own, rel=1e-10)
        assert closed["Direct effect (per 100 mg)"] == pytest.approx(
            table.iloc[0]["Direct effect (per 100 mg)"], rel=1e-10)
        assert closed["Total effect (per 100 mg)"] == pytest.approx(
            table.iloc[0]["Total effect (per 100 mg)"], rel=1e-10)

    def test_simulation_estimator_agrees_within_its_monte_carlo_noise(self, table):
        own = table.iloc[0][INDIRECT_COL]
        rows = table[table["Software"].str.contains("Mediation")]
        acme = rows.iloc[0][INDIRECT_COL]
        if np.isnan(acme):
            pytest.skip("statsmodels Mediation unavailable in this environment")
        mc_se = rows.iloc[0]["Monte Carlo SE"]
        assert abs(acme - own) < 5 * mc_se
        assert abs(acme - own) / abs(own) < 0.05

    def test_pingouin_bootstrap_agrees_when_it_is_installed(self, table):
        rows = table[table["Software"].str.startswith("pingouin")]
        value = rows.iloc[0][INDIRECT_COL]
        if np.isnan(value):
            pytest.skip("pingouin not installed")
        assert value == pytest.approx(table.iloc[0][INDIRECT_COL], rel=1e-6)

    def test_every_available_interval_excludes_zero(self, table):
        with_ci = table[table["95% CI low"].notna()]
        assert len(with_ci) >= 1
        for _, row in with_ci.iterrows():
            assert row["95% CI low"] < row["95% CI high"] < 0

    def test_markdown_report_quotes_the_difference(self, table):
        text = benchmark_markdown({
            "timing": time_pipeline(n_boot=50, repeats=1),
            "scalability": scalability_curve(sizes=[500], n_boot=20),
            "software": table,
        })
        assert "Agreement with established mediation software" in text
        assert "Scalability" in text
        assert "machine precision" in text
