"""Tests for loading, coding and cleaning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bkmediation.config import AGE_RANGE, STRESS_CODING
from bkmediation.data import (
    build_analytic_sample,
    encode_gender,
    encode_stress,
    load_raw,
    remove_outliers_iqr,
)


class TestStressCoding:
    def test_labels_map_to_the_documented_scores(self, small_frame):
        out = encode_stress(small_frame)
        assert out.loc[0, "Stress_Score"] == STRESS_CODING["Low"]
        assert out.loc[1, "Stress_Score"] == STRESS_CODING["Medium"]
        assert out.loc[2, "Stress_Score"] == STRESS_CODING["High"]

    def test_alternative_coding_is_honoured(self, small_frame):
        out = encode_stress(small_frame, {"Low": 1, "Medium": 2, "High": 3})
        assert out["Stress_Score"].tolist() == [1, 2, 3, 1, 2]

    def test_unknown_label_becomes_missing_rather_than_a_silent_zero(self, small_frame):
        frame = small_frame.copy()
        frame.loc[0, "Stress_Level"] = "Extreme"
        out = encode_stress(frame)
        assert np.isnan(out.loc[0, "Stress_Score"])

    def test_encoding_does_not_mutate_the_input(self, small_frame):
        encode_stress(small_frame)
        assert "Stress_Score" not in small_frame.columns


class TestGenderCoding:
    def test_other_is_not_pooled_with_women(self, small_frame):
        """The correction Reviewer 3 asked for."""
        out, cols = encode_gender(small_frame, "three_level")
        other = out[out["Gender"] == "Other"]
        female = out[out["Gender"] == "Female"]
        assert cols == ["Gender_Male", "Gender_Other"]
        assert other["Gender_Other"].eq(1).all()
        assert other["Gender_Male"].eq(0).all()
        assert female["Gender_Other"].eq(0).all()
        assert female["Gender_Male"].eq(0).all()
        # Other and Female must not share an identical covariate row.
        assert not other[cols].iloc[0].equals(female[cols].iloc[0])

    def test_legacy_scheme_reproduces_the_original_single_indicator(self, small_frame):
        out, cols = encode_gender(small_frame, "binary_male")
        assert cols == ["Gender_Num"]
        assert out["Gender_Num"].tolist() == [1, 0, 0, 0, 1]

    def test_gender_num_is_always_present_for_downstream_scripts(self, small_frame):
        out, _ = encode_gender(small_frame, "three_level")
        assert "Gender_Num" in out.columns

    def test_unknown_scheme_raises(self, small_frame):
        with pytest.raises(ValueError, match="Unknown gender scheme"):
            encode_gender(small_frame, "nonexistent")


class TestOutlierRule:
    def test_extreme_value_is_removed_and_logged(self):
        frame = pd.DataFrame({"x": [*np.arange(50, dtype=float), 10_000.0]})
        out, log = remove_outliers_iqr(frame, ["x"])
        assert log["x"] == 1
        assert out["x"].max() < 100

    def test_larger_k_keeps_more_rows(self):
        frame = pd.DataFrame({"x": [*np.arange(50, dtype=float), 120.0]})
        strict, _ = remove_outliers_iqr(frame, ["x"], k=1.5)
        loose, _ = remove_outliers_iqr(frame, ["x"], k=3.0)
        assert len(loose) >= len(strict)

    def test_clean_column_removes_nothing(self):
        frame = pd.DataFrame({"x": np.arange(100, dtype=float)})
        out, log = remove_outliers_iqr(frame, ["x"])
        assert log["x"] == 0
        assert len(out) == 100


class TestLoading:
    def test_missing_file_raises_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            load_raw(tmp_path / "does_not_exist.csv")

    def test_roundtrip_from_a_temporary_csv(self, small_frame, tmp_path):
        path = tmp_path / "mini.csv"
        small_frame.to_csv(path, index=False)
        assert len(load_raw(path)) == len(small_frame)


class TestAnalyticSample:
    def test_age_filter_is_inclusive_on_both_bounds(self, small_frame, tmp_path):
        path = tmp_path / "mini.csv"
        small_frame.to_csv(path, index=False)
        sample = build_analytic_sample(path=path, outlier_k=None)
        assert sample.df["Age"].min() >= AGE_RANGE[0]
        assert sample.df["Age"].max() <= AGE_RANGE[1]
        assert sample.n_raw == len(small_frame)
        assert sample.n_after_age == 3  # ages 18, 40, 65

    def test_covariates_follow_the_gender_scheme(self, small_frame, tmp_path):
        path = tmp_path / "mini.csv"
        small_frame.to_csv(path, index=False)
        corrected = build_analytic_sample(path=path, outlier_k=None)
        legacy = build_analytic_sample(path=path, outlier_k=None,
                                       gender_scheme="binary_male")
        assert corrected.covariates == ["Age", "Gender_Male", "Gender_Other",
                                        "BMI", "Physical_Activity_Hours", "Heart_Rate"]
        assert legacy.covariates == ["Age", "Gender_Num", "BMI",
                                     "Physical_Activity_Hours", "Heart_Rate"]

    def test_rows_missing_an_analysis_variable_are_dropped(self, small_frame, tmp_path):
        frame = small_frame.copy()
        frame.loc[2, "BMI"] = np.nan
        path = tmp_path / "mini.csv"
        frame.to_csv(path, index=False)
        sample = build_analytic_sample(path=path, outlier_k=None)
        assert sample.n_after_missing == sample.n_after_age - 1

    def test_exclusion_table_accounts_for_every_dropped_row(self, small_frame, tmp_path):
        path = tmp_path / "mini.csv"
        small_frame.to_csv(path, index=False)
        sample = build_analytic_sample(path=path)
        table = sample.exclusion_table()
        assert table.iloc[0]["n remaining"] == sample.n_raw
        assert table.iloc[-1]["n remaining"] == sample.n
        assert table["n removed"].sum() == sample.n_raw - sample.n

    def test_synthetic_flag_is_carried_through(self, small_frame, tmp_path):
        path = tmp_path / "mini.csv"
        small_frame.to_csv(path, index=False)
        assert build_analytic_sample(path=path).is_synthetic is True
        assert build_analytic_sample(path=path, is_synthetic=False).is_synthetic is False
        assert "SYNTHETIC" in build_analytic_sample(path=path).describe_source()


class TestPublishedSample:
    """Pins the sample size the manuscript reports."""

    def test_analytic_sample_size_is_9795(self, analytic_sample):
        assert analytic_sample.n == 9795

    def test_no_respondent_is_lost_to_the_sleep_quality_column(self, analytic_sample):
        # The "Excellent" category used to be unmapped and silently dropped
        # ~1,500 rows. Sleep quality is not an analysis variable any more.
        assert "Sleep_Quality_Score" not in analytic_sample.df.columns
        assert analytic_sample.n_after_missing == analytic_sample.n_after_age

    def test_other_gender_respondents_survive_cleaning(self, analytic_sample):
        counts = analytic_sample.df["Gender"].value_counts()
        assert counts.get("Other", 0) > 0
        assert analytic_sample.df["Gender_Other"].sum() == counts["Other"]
