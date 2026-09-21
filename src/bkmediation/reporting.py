"""Report writers: CSV tables and markdown summaries for the manuscript.

Everything the manuscript quotes is written here from a single
:class:`~bkmediation.mediation.MediationResult`, so a table in the paper and
the number in the code cannot disagree.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CAFFEINE_SCALE_LABEL, OUTPUTS_DIR, REPORTS_DIR
from .descriptives import (
    category_counts,
    correlation_matrix,
    describe,
    key_correlations,
    stress_counts,
)
from .diagnostics import run_diagnostics
from .hypotheses import evaluate_hypotheses

__all__ = ["write_analysis_outputs", "analysis_markdown", "SYNTHETIC_DATA_NOTICE"]

SYNTHETIC_DATA_NOTICE = (
    "> **Synthetic data.** The primary dataset is synthetic. Its `Stress_Level` "
    "field was generated partly from sleep- and lifestyle-related quantities, so "
    "mediator and outcome are not independently measured and the indirect effect "
    "reported below is a statistical association inside a simulated "
    "data-generating process, not evidence of a biological mechanism. External "
    "validation against LEMURS and NHANES is reported separately under `outputs/`."
)


def _fmt_p(p) -> str:
    if pd.isna(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def analysis_markdown(sample, result, diagnostics=None) -> str:
    """Full human-readable report for one mediation analysis."""
    df = sample.df
    hyp = evaluate_hypotheses(result)
    paths = result.paths_table()
    effects = result.effects_table()
    diagnostics = diagnostics or run_diagnostics(sample)

    path_view = paths[["Path", "Description", "Coef", "SE (HC3)", "t", "p",
                       "95% CI low", "95% CI high", "Coef (rescaled)",
                       "CI low (rescaled)", "CI high (rescaled)", "Scale"]].copy()
    path_view["p"] = path_view["p"].map(_fmt_p)
    for col in ("Coef", "SE (HC3)", "95% CI low", "95% CI high"):
        path_view[col] = path_view[col].map(lambda v: f"{v:.6f}")
    for col in ("Coef (rescaled)", "CI low (rescaled)", "CI high (rescaled)"):
        path_view[col] = path_view[col].map(lambda v: f"{v:+.4f}")
    path_view["t"] = path_view["t"].map(lambda v: f"{v:.2f}")

    hyp_view = hyp[["Hypothesis", "Statement", "Tested quantity", "Expected sign",
                    "Estimate (per 100 mg)", "95% CI low (per 100 mg)",
                    "95% CI high (per 100 mg)", "CI method", "p", "Decision"]].copy()
    for col in ("Estimate (per 100 mg)", "95% CI low (per 100 mg)",
                "95% CI high (per 100 mg)"):
        hyp_view[col] = hyp_view[col].map(lambda v: f"{v:+.4f}")
    hyp_view["p"] = hyp_view["p"].map(_fmt_p)

    eff_view = effects.copy()
    for col in ("Estimate", "95% CI low", "95% CI high"):
        eff_view[col] = eff_view[col].map(lambda v: "n/a" if pd.isna(v) else f"{v:.6f}")

    vif = diagnostics["vif"]
    total_d, med_d = diagnostics["total_model"], diagnostics["mediated_model"]

    if sample.gender_scheme == "three_level":
        gender_note = ("Female is the reference category; respondents reporting "
                       "\"Other\" have their own indicator and are not pooled with women")
    else:
        gender_note = ("legacy coding: Male = 1, all other respondents = 0 "
                       "(kept only for backward comparison)")
    heteroskedastic = not med_d["homoskedastic_at_05"]
    se_note = (
        "Breusch-Pagan rejects homoskedasticity in the mediated model "
        f"(p = {_fmt_p(med_d['breusch_pagan_p'])}), which is why HC3 standard "
        "errors are the reported ones."
        if heteroskedastic else
        "Breusch-Pagan does not reject homoskedasticity "
        f"(p = {_fmt_p(med_d['breusch_pagan_p'])}); HC3 standard errors are "
        "reported anyway, as they remain valid under homoskedasticity."
    )

    lines = [
        "# Mediation Analysis Report",
        "",
        f"Caffeine intake -> perceived stress -> sleep duration. "
        f"Source: {sample.describe_source()}.",
        "",
        SYNTHETIC_DATA_NOTICE if sample.is_synthetic else "",
        "",
        "## 1. Sample",
        "",
        sample.exclusion_table().to_markdown(index=False),
        "",
        f"Final analytic sample: **n = {result.n:,}**. Covariates: "
        f"{', '.join(result.covariates)}. Gender coding: `{sample.gender_scheme}` "
        f"({gender_note}).",
        "",
        "## 2. Descriptive statistics",
        "",
        describe(df).to_markdown(index=False),
        "",
        "Gender:" if "Gender" in df else "",
        "",
        category_counts(df, "Gender").to_markdown(index=False) if "Gender" in df else "",
        "",
        "Stress level:" if "Stress_Level" in df else "",
        "",
        stress_counts(df).to_markdown(index=False) if "Stress_Level" in df else "",
        "",
        "## 3. Bivariate correlations",
        "",
        key_correlations(df).to_markdown(index=False),
        "",
        "## 4. Hypothesis tests",
        "",
        "Estimates and intervals are per 100 mg of caffeine per day. H1 and H2 "
        "use the HC3 model-based interval; H3 uses the percentile-bootstrap "
        "interval for the indirect effect.",
        "",
        hyp_view.to_markdown(index=False),
        "",
        "## 5. Path estimates (manuscript Table 2)",
        "",
        f"Standard errors are heteroskedasticity-consistent (HC3). Rescaled "
        f"coefficients for the exposure paths are {CAFFEINE_SCALE_LABEL}; path b is "
        f"per one point of the stress scale.",
        "",
        path_view.to_markdown(index=False),
        "",
        "Classical (non-robust) intervals, for comparison:",
        "",
        paths[["Path", "SE (classical)", "95% CI low (classical)",
               "95% CI high (classical)"]].to_markdown(index=False),
        "",
        "Percentile-bootstrap intervals for every path:",
        "",
        paths[["Path", "95% CI low (bootstrap)",
               "95% CI high (bootstrap)"]].to_markdown(index=False),
        "",
        "## 6. Indirect, direct and total effects",
        "",
        eff_view.to_markdown(index=False),
        "",
        f"Primary inference: the percentile-bootstrap interval for a*b "
        f"({result.n_boot:,} resamples, seed {result.seed}). The Sobel test "
        f"(z = {result.effect_sizes['sobel_z']:.3f}, p = "
        f"{_fmt_p(result.effect_sizes['sobel_p'])}) is reported only as a "
        f"normal-theory comparison.",
        "",
        f"Cohen's f2 = {result.effect_sizes['f2_mediator_increment']:.4f} is the "
        f"increment in explained variance when the mediator enters the outcome "
        f"model (delta R2 = {result.model_fit['delta_r2']:.4f}). It is **not** a "
        f"mediation effect size and is not interpreted as one; the mediation "
        f"effect sizes are the standardised indirect effect "
        f"({result.effect_sizes['ab_std']:.4f}) and the proportion mediated "
        f"({result.effect_sizes['prop']:.1f}%, 95% CI "
        f"[{result.effect_sizes['prop_ci'][0]:.1f}%, "
        f"{result.effect_sizes['prop_ci'][1]:.1f}%]).",
        "",
        "## 7. Baron-Kenny conditions",
        "",
        pd.DataFrame({
            "Condition": list(result.conditions.keys()),
            "Met": ["yes" if v else "no" for v in result.conditions.values()],
        }).to_markdown(index=False),
        "",
        f"Conclusion: **{result.mediation_type}** (statistical indirect "
        f"association under the assumed ordering, not an identified causal effect).",
        "",
        "## 8. Assumption diagnostics",
        "",
        vif.to_markdown(index=False),
        "",
        pd.DataFrame([
            {"Model": "Total effect (Y ~ X + C)", **total_d},
            {"Model": "Mediated (Y ~ X + M + C)", **med_d},
            {"Model": "Mediator (M ~ X + C)", **diagnostics["mediator_model"]},
        ]).to_markdown(index=False),
        "",
        se_note,
        "",
    ]
    return "\n".join(line for line in lines if line is not None)


def write_analysis_outputs(sample, result, reports_dir=None, diagnostics=None,
                           markdown_copy_dir=OUTPUTS_DIR) -> dict:
    """Write every CSV/markdown artefact for one analysis; return the paths.

    `Reports/` is regenerated on every run and is git-ignored, so a copy of the
    human-readable report is also placed in `markdown_copy_dir` (by default
    `outputs/`, which is versioned) unless that is set to None.
    """
    reports_dir = Path(reports_dir or REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)
    df = sample.df
    diagnostics = diagnostics or run_diagnostics(sample)

    written = {}

    def _write(name: str, frame: pd.DataFrame, index: bool = False):
        path = reports_dir / name
        frame.to_csv(path, index=index)
        written[name] = path

    _write("descriptive_statistics.csv", describe(df))
    _write("correlation_matrix.csv", correlation_matrix(df), index=True)
    _write("key_correlations.csv", key_correlations(df))
    _write("sample_exclusions.csv", sample.exclusion_table())
    _write("mediation_results.csv", pd.DataFrame([result.to_dict()]))
    _write("mediation_paths_table.csv", result.paths_table())
    _write("mediation_effects_table.csv", result.effects_table())
    _write("hypothesis_tests.csv", evaluate_hypotheses(result))
    _write("assumption_diagnostics.csv", diagnostics["vif"])
    _write("residual_diagnostics.csv", pd.DataFrame([
        {"Model": "total", **diagnostics["total_model"]},
        {"Model": "mediated", **diagnostics["mediated_model"]},
        {"Model": "mediator", **diagnostics["mediator_model"]},
    ]))

    markdown = analysis_markdown(sample, result, diagnostics)
    report_path = reports_dir / "analysis_report.md"
    report_path.write_text(markdown, encoding="utf-8")
    written["analysis_report.md"] = report_path

    if markdown_copy_dir is not None:
        copy_dir = Path(markdown_copy_dir)
        copy_dir.mkdir(parents=True, exist_ok=True)
        copy_path = copy_dir / "analysis_report.md"
        if copy_path != report_path:
            copy_path.write_text(markdown, encoding="utf-8")
            written["outputs/analysis_report.md"] = copy_path
        result.paths_table().to_csv(copy_dir / "mediation_paths_table.csv", index=False)
        evaluate_hypotheses(result).to_csv(copy_dir / "hypothesis_tests.csv", index=False)
    return written
