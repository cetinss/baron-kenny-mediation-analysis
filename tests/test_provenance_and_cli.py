"""Tests for provenance reporting, the gender-coding comparison and the CLI."""

from __future__ import annotations

import json

import pytest

from bkmediation.cli import build_parser, main
from bkmediation.comparisons import compare_gender_coding, gender_coding_markdown
from bkmediation.config import N_BOOTSTRAP, RANDOM_SEED
from bkmediation.provenance import (
    TRACKED_PACKAGES,
    collect_provenance,
    provenance_markdown,
    write_provenance,
)


class TestProvenance:
    def test_reports_python_packages_seed_and_git(self):
        info = collect_provenance()
        assert info["python"]
        assert info["analysis_settings"]["random_seed"] == RANDOM_SEED
        assert info["analysis_settings"]["n_bootstrap"] == N_BOOTSTRAP
        for package in ("numpy", "pandas", "statsmodels"):
            assert package in info["packages"]
            assert info["packages"][package] != "not installed"
        assert set(info["packages"]) == set(TRACKED_PACKAGES)
        assert "commit" in info["git"]

    def test_extra_fields_are_merged(self):
        info = collect_provenance({"dataset": "example.csv"})
        assert info["dataset"] == "example.csv"

    def test_markdown_mentions_the_seed_and_the_commit(self):
        text = provenance_markdown()
        assert "Random seed" in text
        assert "Commit" in text
        assert "Python" in text

    def test_write_provenance_emits_json_and_markdown(self, tmp_path):
        info = write_provenance(tmp_path)
        assert (tmp_path / "provenance.json").exists()
        assert (tmp_path / "provenance.md").exists()
        reloaded = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
        assert reloaded["bkmediation_version"] == info["bkmediation_version"]


class TestGenderCodingComparison:
    @pytest.fixture(scope="class")
    @staticmethod
    def table(analytic_sample):
        return compare_gender_coding(n_boot=300)

    def test_both_schemes_are_estimated(self, table):
        assert set(table["Gender coding"]) == {"three_level", "binary_male"}

    def test_the_correction_does_not_change_the_conclusion(self, table):
        assert table["Conclusion"].nunique() == 1
        corrected = table.set_index("Gender coding").loc["three_level"]
        original = table.set_index("Gender coding").loc["binary_male"]
        assert corrected["a*b"] == pytest.approx(original["a*b"], abs=1e-5)

    def test_markdown_explains_what_changed(self, table):
        text = gender_coding_markdown(table)
        assert "Other" in text
        assert "reference category" in text


class TestCli:
    def test_parser_exposes_every_command(self):
        parser = build_parser()
        actions = {a.dest: a for a in parser._actions}
        assert set(actions["command"].choices) == {
            "analyze", "hypotheses", "samples", "gender", "benchmark",
            "provenance", "all"}

    def test_parser_defaults_match_the_configuration(self):
        args = build_parser().parse_args(["analyze"])
        assert args.seed == RANDOM_SEED
        assert args.n_boot == N_BOOTSTRAP
        assert args.gender_scheme == "three_level"
        assert args.cov_type == "HC3"

    def test_provenance_command_runs(self, capsys, tmp_path):
        assert main(["provenance", "--out-dir", str(tmp_path)]) == 0
        assert "seed" in capsys.readouterr().out
        assert (tmp_path / "provenance.json").exists()

    @pytest.mark.slow
    def test_hypotheses_command_prints_three_decisions(self, capsys, tmp_path,
                                                       raw_dataset_available):
        if not raw_dataset_available:
            pytest.skip("dataset not available")
        assert main(["hypotheses", "--n-boot", "200", "--quiet",
                     "--out-dir", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert out.count("SUPPORTED") >= 3
        for key in ("H1", "H2", "H3"):
            assert key in out
