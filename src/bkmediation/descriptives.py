"""Descriptive statistics and bivariate correlations (manuscript Table 1)."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from .config import EXPOSURE, MEDIATOR, OUTCOME, STRESS_LEVELS

__all__ = ["VARIABLE_LABELS", "describe", "category_counts", "correlation_matrix",
           "key_correlations"]

VARIABLE_LABELS = {
    EXPOSURE: "Daily Caffeine Intake (mg)",
    MEDIATOR: "Stress Score (2-8)",
    OUTCOME: "Sleep Duration (hours)",
    "Age": "Age (years)",
    "BMI": "BMI (kg/m2)",
    "Physical_Activity_Hours": "Physical Activity (h/week)",
    "Heart_Rate": "Resting Heart Rate (bpm)",
}


def describe(df: pd.DataFrame, variables=None) -> pd.DataFrame:
    """Mean/SD/median/range/skewness/kurtosis for the analysis variables."""
    variables = list(variables or VARIABLE_LABELS)
    variables = [v for v in variables if v in df.columns]
    rows = []
    for var in variables:
        s = df[var]
        rows.append({
            "Variable": VARIABLE_LABELS.get(var, var),
            "n": int(s.count()),
            "Mean": round(float(s.mean()), 2),
            "SD": round(float(s.std()), 2),
            "Median": round(float(s.median()), 2),
            "Min": round(float(s.min()), 2),
            "Max": round(float(s.max()), 2),
            "Skewness": round(float(s.skew()), 3),
            "Kurtosis": round(float(s.kurtosis()), 3),
        })
    return pd.DataFrame(rows)


def category_counts(df: pd.DataFrame, column: str, order=None) -> pd.DataFrame:
    """Counts and percentages for a categorical variable (gender, stress)."""
    counts = df[column].value_counts()
    if order is not None:
        counts = counts.reindex(list(order)).fillna(0).astype(int)
    total = len(df)
    return pd.DataFrame({
        "Category": counts.index,
        "n": counts.to_numpy(),
        "%": (counts.to_numpy() / total * 100).round(1),
    })


def stress_counts(df: pd.DataFrame) -> pd.DataFrame:
    return category_counts(df, "Stress_Level", order=STRESS_LEVELS)


def correlation_matrix(df: pd.DataFrame, variables=None, method: str = "pearson") -> pd.DataFrame:
    variables = [v for v in (variables or VARIABLE_LABELS) if v in df.columns]
    return df[variables].corr(method=method)


def key_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """The three correlations that map onto the mediation paths."""
    pairs = [
        (EXPOSURE, MEDIATOR, "Caffeine <-> Stress"),
        (EXPOSURE, OUTCOME, "Caffeine <-> Sleep duration"),
        (MEDIATOR, OUTCOME, "Stress <-> Sleep duration"),
    ]
    rows = []
    for v1, v2, label in pairs:
        r, p = stats.pearsonr(df[v1], df[v2])
        rows.append({"Pair": label, "r": round(float(r), 4), "p": float(p),
                     "n": int(len(df))})
    return pd.DataFrame(rows)
