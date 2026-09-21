"""Tests for the estimator itself.

The simulated fixtures have a known indirect effect (TRUE_AB), so these are
tests against a right answer, not just regression tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from bkmediation.config import CAFFEINE_SCALE
from bkmediation.mediation import bootstrap_paths, fit_paths, run_mediation

from .conftest import TRUE_A, TRUE_AB, TRUE_B, TRUE_CP, make_simulated


def _arrays(df, covs):
    return (df["Caffeine_mg"].to_numpy(), df["Stress_Score"].to_numpy(),
            df["Sleep_Hours"].to_numpy(), df[covs].to_numpy())


class TestPathRecovery:
    def test_paths_recover_the_true_coefficients(self, simulated, sim_covariates):
        X, M, Y, C = _arrays(simulated, sim_covariates)
        paths, _ = fit_paths(X, M, Y, C)
        assert paths["a"]["coef"] == pytest.approx(TRUE_A, rel=0.05)
        assert paths["b"]["coef"] == pytest.approx(TRUE_B, rel=0.05)
        assert paths["cp"]["coef"] == pytest.approx(TRUE_CP, rel=0.10)

    def test_total_effect_decomposes_into_direct_plus_indirect(self, simulated, sim_covariates):
        """c = c' + a*b holds exactly for OLS with the same covariate set."""
        X, M, Y, C = _arrays(simulated, sim_covariates)
        paths, _ = fit_paths(X, M, Y, C)
        decomposed = paths["cp"]["coef"] + paths["a"]["coef"] * paths["b"]["coef"]
        assert decomposed == pytest.approx(paths["c"]["coef"], rel=1e-9, abs=1e-12)

    def test_every_path_has_a_confidence_interval_that_brackets_its_estimate(
            self, simulated, sim_covariates):
        X, M, Y, C = _arrays(simulated, sim_covariates)
        paths, _ = fit_paths(X, M, Y, C)
        for name, p in paths.items():
            assert p["ci_lo"] < p["coef"] < p["ci_hi"], name
            assert p["ci_lo_classical"] < p["coef"] < p["ci_hi_classical"], name

    def test_robust_covariance_changes_only_the_standard_errors(
            self, simulated, sim_covariates):
        X, M, Y, C = _arrays(simulated, sim_covariates)
        hc3, _ = fit_paths(X, M, Y, C, cov_type="HC3")
        classical, _ = fit_paths(X, M, Y, C, cov_type="nonrobust")
        for name in ("a", "b", "c", "cp"):
            assert hc3[name]["coef"] == pytest.approx(classical[name]["coef"], rel=1e-12)
        assert hc3["b"]["se"] != pytest.approx(hc3["b"]["se_classical"], rel=1e-9)


    def test_mean_centring_leaves_every_slope_unchanged(self, simulated, sim_covariates):
        """No interaction terms, so centring moves intercepts only.

        The manuscript describes mean-centring; the main pipeline does not
        centre. This test shows the two descriptions are equivalent for the
        reported quantities.
        """
        X, M, Y, C = _arrays(simulated, sim_covariates)
        raw, _ = fit_paths(X, M, Y, C)
        centred, _ = fit_paths(X - X.mean(), M - M.mean(), Y - Y.mean(),
                               C - C.mean(axis=0))
        for name in ("a", "b", "c", "cp"):
            assert centred[name]["coef"] == pytest.approx(raw[name]["coef"], rel=1e-9)
            assert centred[name]["se"] == pytest.approx(raw[name]["se"], rel=1e-9)


class TestBootstrap:
    def test_same_seed_gives_identical_draws(self, simulated, sim_covariates):
        X, M, Y, C = _arrays(simulated, sim_covariates)
        first = bootstrap_paths(X, M, Y, C, n_boot=200, seed=123)
        second = bootstrap_paths(X, M, Y, C, n_boot=200, seed=123)
        assert np.allclose(first["ab"], second["ab"])

    def test_different_seeds_give_different_draws(self, simulated, sim_covariates):
        X, M, Y, C = _arrays(simulated, sim_covariates)
        first = bootstrap_paths(X, M, Y, C, n_boot=200, seed=1)
        second = bootstrap_paths(X, M, Y, C, n_boot=200, seed=2)
        assert not np.allclose(first["ab"], second["ab"])

    def test_bootstrap_interval_covers_the_true_indirect_effect(
            self, simulated, sim_covariates):
        X, M, Y, C = _arrays(simulated, sim_covariates)
        boot = bootstrap_paths(X, M, Y, C, n_boot=1000, seed=42)
        lo, hi = np.percentile(boot["ab"], [2.5, 97.5])
        assert lo < TRUE_AB < hi

    def test_all_replicates_are_usable_on_well_conditioned_data(
            self, simulated, sim_covariates):
        X, M, Y, C = _arrays(simulated, sim_covariates)
        boot = bootstrap_paths(X, M, Y, C, n_boot=100, seed=5)
        assert boot["n_valid"] == 100
        assert boot["ab"].shape == (100,)


class TestResultObject:
    @pytest.fixture(scope="class")
    @staticmethod
    def result():
        return run_mediation(make_simulated(), covariates=["Cov1", "Cov2"],
                             n_boot=500, seed=42)

    def test_indirect_effect_equals_a_times_b(self, result):
        expected = result.paths["a"]["coef"] * result.paths["b"]["coef"]
        assert result.effect_sizes["ab"] == pytest.approx(expected, rel=1e-12)

    def test_indirect_effect_is_detected_when_it_exists(self, result):
        assert result.indirect_is_significant
        assert result.mediation_type in {"PARTIAL MEDIATION", "FULL MEDIATION"}

    def test_rescaled_coefficients_are_per_100_mg(self, result):
        table = result.paths_table().set_index("Path")
        for path in ("a", "c", "c'"):
            row = table.loc[path]
            assert row["Coef (rescaled)"] == pytest.approx(row["Coef"] * CAFFEINE_SCALE)
            assert row["Scale"] == "per 100 mg/day"
        b_row = table.loc["b"]
        assert b_row["Coef (rescaled)"] == pytest.approx(b_row["Coef"])
        assert b_row["Scale"] == "per stress point"

    def test_paths_table_reports_three_kinds_of_interval(self, result):
        table = result.paths_table()
        for col in ("95% CI low", "95% CI low (classical)", "95% CI low (bootstrap)",
                    "95% CI high", "95% CI high (classical)", "95% CI high (bootstrap)"):
            assert col in table.columns
            assert table[col].notna().all()

    def test_proportion_mediated_comes_with_a_bootstrap_interval(self, result):
        lo, hi = result.effect_sizes["prop_ci"]
        assert lo < result.effect_sizes["prop"] < hi

    def test_f2_is_labelled_as_a_variance_increment_not_a_mediation_effect_size(self, result):
        assert "f2_mediator_increment" in result.effect_sizes
        assert "f2" not in result.effect_sizes
        assert "not a mediation effect size" in result.summary().lower()

    def test_effects_table_lists_indirect_direct_and_total(self, result):
        effects = result.effects_table()["Effect"].tolist()
        assert any("Indirect (a*b)" == e for e in effects)
        assert any("Direct (c')" == e for e in effects)
        assert any("Total (c)" == e for e in effects)

    def test_legacy_dict_keeps_the_keys_the_figure_code_uses(self, result):
        legacy = result.legacy_dict()
        for key in ("a", "b", "c", "cp", "ab", "ci_lo", "ci_hi", "prop", "f2",
                    "med_type", "cond", "boot", "model1", "model2", "model3",
                    "r2_total", "r2_med", "sob_z", "n"):
            assert key in legacy

    def test_to_dict_is_flat_and_csv_ready(self, result):
        flat = result.to_dict()
        assert all(not isinstance(v, (dict, list, tuple)) or isinstance(v, str)
                   for v in flat.values())
        assert flat["n"] == result.n


class TestNoMediationCase:
    def test_no_mediation_is_reported_when_the_mediator_is_irrelevant(self):
        """b = 0 by construction: the interval for a*b must cover zero."""
        data = make_simulated(n=2000, b=0.0, cp=-0.002, seed=11)
        result = run_mediation(data, covariates=["Cov1", "Cov2"], n_boot=500, seed=3)
        lo, hi = result.effect_sizes["ab_ci"]
        assert lo <= 0 <= hi
        assert result.mediation_type == "NO MEDIATION"
        assert not result.indirect_is_significant

    def test_no_mediation_when_the_exposure_does_not_move_the_mediator(self):
        data = make_simulated(n=2000, a=0.0, seed=13)
        result = run_mediation(data, covariates=["Cov1", "Cov2"], n_boot=500, seed=3)
        assert result.mediation_type == "NO MEDIATION"


class TestInputValidation:
    def test_missing_covariate_column_raises_keyerror(self, simulated):
        with pytest.raises(KeyError, match="Covariates missing"):
            run_mediation(simulated, covariates=["Cov1", "NotThere"], n_boot=10)


class TestPublishedNumbers:
    """Pins manuscript Table 2 so a refactor cannot move it silently."""

    def test_paths_match_the_reported_values(self, analytic_result):
        p = analytic_result.paths
        assert p["a"]["coef"] == pytest.approx(0.002121, abs=5e-6)
        assert p["b"]["coef"] == pytest.approx(-0.481648, abs=5e-6)
        assert p["c"]["coef"] == pytest.approx(-0.001687, abs=5e-6)
        assert p["cp"]["coef"] == pytest.approx(-0.000665, abs=5e-6)

    def test_indirect_effect_and_conclusion_match(self, analytic_result):
        assert analytic_result.effect_sizes["ab"] == pytest.approx(-0.001021, abs=5e-6)
        assert analytic_result.effect_sizes["prop"] == pytest.approx(60.6, abs=0.2)
        assert analytic_result.mediation_type == "PARTIAL MEDIATION"

    def test_analysis_is_flagged_as_synthetic(self, analytic_result):
        assert analytic_result.is_synthetic
        assert "synthetic" in analytic_result.summary().lower()
