"""Data loading, coding and cleaning.

The entire path from the raw CSV to the analytic sample lives in this module
and is expressed as small functions that can be tested one at a time.

Provenance note
---------------
The primary dataset (`data/synthetic_coffee_health_10000.csv`) is SYNTHETIC.
It was generated for teaching/demonstration purposes and its `Stress_Level`
field was produced partly from sleep- and lifestyle-related quantities, so the
mediator and the outcome are not independently measured. Every estimate
obtained from it is therefore a statistical association inside a simulated
data-generating process and a demonstration of the analysis pipeline - not
evidence about a biological mechanism. `is_synthetic` on the returned
:class:`AnalyticSample` carries this flag through the rest of the code so that
reports can state it automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import (
    AGE_RANGE,
    BASE_COVARIATES,
    CONTINUOUS_FOR_OUTLIERS,
    DEFAULT_DATASET,
    DEFAULT_GENDER_SCHEME,
    EXPOSURE,
    GENDER_COVARIATES,
    GENDER_SCHEMES,
    MEDIATOR,
    OUTCOME,
    OUTLIER_IQR_K,
    STRESS_CODING,
)

__all__ = [
    "AnalyticSample",
    "build_analytic_sample",
    "encode_gender",
    "encode_stress",
    "load_and_clean_data",
    "load_raw",
    "remove_outliers_iqr",
]


@dataclass
class AnalyticSample:
    """The cleaned analysis dataset plus everything needed to describe it."""

    df: pd.DataFrame
    covariates: list
    gender_scheme: str
    n_raw: int
    n_after_age: int
    n_after_missing: int
    outlier_log: dict = field(default_factory=dict)
    is_synthetic: bool = True
    source: str = ""

    @property
    def n(self) -> int:
        return len(self.df)

    @property
    def n_removed(self) -> int:
        return self.n_raw - self.n

    def exclusion_table(self) -> pd.DataFrame:
        """Row-by-row account of every exclusion, for the Methods section."""
        rows = [
            ("Raw records", self.n_raw, 0),
            ("After age filter ({}-{} years)".format(*AGE_RANGE),
             self.n_after_age, self.n_raw - self.n_after_age),
            ("After listwise deletion on analysis variables",
             self.n_after_missing, self.n_after_age - self.n_after_missing),
        ]
        running = self.n_after_missing
        for col, removed in self.outlier_log.items():
            running -= removed
            rows.append((f"After {OUTLIER_IQR_K}x IQR outlier rule: {col}", running, removed))
        rows.append(("Final analytic sample", self.n, 0))
        return pd.DataFrame(rows, columns=["Step", "n remaining", "n removed"])

    def describe_source(self) -> str:
        kind = "SYNTHETIC" if self.is_synthetic else "real-world"
        return f"{Path(self.source).name} ({kind}, n = {self.n:,})"


def load_raw(path=None) -> pd.DataFrame:
    """Read the raw dataset without any filtering."""
    path = Path(path) if path is not None else DEFAULT_DATASET
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place the CSV under data/ or pass an "
            f"explicit path."
        )
    return pd.read_csv(path)


def encode_stress(df: pd.DataFrame, coding=None) -> pd.DataFrame:
    """Map the ordinal `Stress_Level` labels onto numeric `Stress_Score`.

    The default spacing (Low=2, Medium=5, High=8) is equal-interval. It is an
    assumption, not a measurement; `sensitivity_analysis.py` re-estimates the
    model under alternative and coding-free specifications.
    """
    coding = STRESS_CODING if coding is None else coding
    out = df.copy()
    out[MEDIATOR] = out["Stress_Level"].map(coding)
    return out


def encode_gender(df: pd.DataFrame, scheme: str = DEFAULT_GENDER_SCHEME):
    """Create gender covariates and return (df, column names).

    scheme="three_level"  Female is the reference category; `Gender_Male` and
                          `Gender_Other` are separate indicators, so
                          respondents who selected "Other" are NOT pooled with
                          women.
    scheme="binary_male"  legacy Male = 1 / not-Male = 0 single indicator.

    `Gender_Num` is always written as well, because the external-validation
    and sensitivity scripts refer to it by name.
    """
    if scheme not in GENDER_SCHEMES:
        raise ValueError(f"Unknown gender scheme {scheme!r}; use one of {GENDER_SCHEMES}")
    out = df.copy()
    out["Gender_Male"] = (out["Gender"] == "Male").astype(int)
    out["Gender_Other"] = (out["Gender"] == "Other").astype(int)
    out["Gender_Num"] = out["Gender_Male"]
    return out, list(GENDER_COVARIATES[scheme])


def remove_outliers_iqr(df: pd.DataFrame, columns, k: float = OUTLIER_IQR_K, verbose: bool = False):
    """Drop rows outside Q1 - k*IQR .. Q3 + k*IQR on each column in turn.

    Returns the filtered frame and a {column: n_removed} log. Columns are
    processed sequentially, so the log is sequential, not marginal.
    """
    out = df.copy()
    log = {}
    for col in columns:
        q1, q3 = out[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        before = len(out)
        out = out[(out[col] >= lo) & (out[col] <= hi)]
        log[col] = before - len(out)
        if verbose and log[col]:
            print(f"  {col}: {log[col]} outliers removed (range: {lo:.1f} - {hi:.1f})")
    return out, log


def build_analytic_sample(
    path=None,
    gender_scheme: str = DEFAULT_GENDER_SCHEME,
    stress_coding=None,
    outlier_k=OUTLIER_IQR_K,
    age_range=AGE_RANGE,
    is_synthetic: bool = True,
    verbose: bool = False,
) -> AnalyticSample:
    """Run the full cleaning pipeline and return an :class:`AnalyticSample`.

    Steps, in order:
      1. read the raw file;
      2. restrict to adults within `age_range`;
      3. encode the ordinal mediator and the gender covariates;
      4. listwise-delete rows missing any analysis variable;
      5. apply the k x IQR outlier rule (skipped when `outlier_k` is None).

    `Sleep_Quality` is deliberately not required: the single outcome is sleep
    duration, and requiring a quality score would silently drop the ~1,500
    respondents in the unmapped "Excellent" category.
    """
    path = Path(path) if path is not None else DEFAULT_DATASET
    raw = load_raw(path)
    n_raw = len(raw)
    if verbose:
        print(f"Raw data loaded: {n_raw:,} records, {len(raw.columns)} variables")

    df = raw[(raw["Age"] >= age_range[0]) & (raw["Age"] <= age_range[1])].copy()
    n_after_age = len(df)
    if verbose:
        print(f"After age filter ({age_range[0]}-{age_range[1]}): {n_after_age:,} records")

    df = encode_stress(df, stress_coding)
    df, gender_cols = encode_gender(df, gender_scheme)

    key_vars = [EXPOSURE, MEDIATOR, OUTCOME, "Age", *gender_cols, *BASE_COVARIATES[1:]]
    df = df.dropna(subset=key_vars)
    n_after_missing = len(df)
    if verbose:
        print(f"After missing-value removal: {n_after_missing:,} "
              f"(dropped {n_after_age - n_after_missing})")

    if outlier_k is None:
        log = {}
    else:
        df, log = remove_outliers_iqr(df, CONTINUOUS_FOR_OUTLIERS, k=outlier_k, verbose=verbose)

    covariates = ["Age", *gender_cols, *BASE_COVARIATES[1:]]
    sample = AnalyticSample(
        df=df.reset_index(drop=True),
        covariates=covariates,
        gender_scheme=gender_scheme,
        n_raw=n_raw,
        n_after_age=n_after_age,
        n_after_missing=n_after_missing,
        outlier_log=log,
        is_synthetic=is_synthetic,
        source=str(path),
    )
    if verbose:
        print(f"\nFinal analytic sample: n = {sample.n:,}")
        print(f"Total records removed: {sample.n_removed:,} "
              f"({sample.n_removed / n_raw * 100:.1f}%)")
    return sample


def load_and_clean_data(verbose: bool = True, **kwargs) -> pd.DataFrame:
    """Backward-compatible wrapper returning just the cleaned DataFrame.

    `src/sensitivity_analysis.py`, `src/make_manuscript_figures.py` and the
    original `src/main.py` entry point call this name.
    """
    return build_analytic_sample(verbose=verbose, **kwargs).df
