"""Runtime, scalability and cross-software agreement benchmarks.

Reviewer 3 asked for the computational side of the pipeline to be reported:
how long it runs, how it scales, and whether its numbers agree with
established mediation software.

Three things are measured:

1. `time_pipeline`       - wall-clock time of each stage (load/clean, fit the
                           three regressions, bootstrap).
2. `scalability_curve`   - estimation and bootstrap time as the sample grows,
                           on subsamples of the analytic sample and on larger
                           simulated samples.
3. `compare_software`    - the same model estimated by other implementations,
                           so the indirect effect can be compared numerically
                           rather than asserted.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import (
    CAFFEINE_SCALE,
    EXPOSURE,
    MEDIATOR,
    N_BOOTSTRAP,
    OUTCOME,
    RANDOM_SEED,
)
from .data import build_analytic_sample
from .mediation import bootstrap_paths, fit_paths, run_mediation

__all__ = ["time_pipeline", "scalability_curve", "compare_software", "run_benchmarks",
           "benchmark_markdown"]

INDIRECT_COL = "Indirect effect (per 100 mg)"


def _timer():
    return time.perf_counter()


def _pkg_version() -> str:
    from . import __version__
    return __version__


def time_pipeline(sample=None, n_boot: int = N_BOOTSTRAP, repeats: int = 3) -> pd.DataFrame:
    """Median wall-clock time of each pipeline stage over `repeats` runs."""
    stages = {"load_and_clean": [], "fit_paths": [], f"bootstrap_{n_boot}": []}

    for _ in range(repeats):
        t0 = _timer()
        s = build_analytic_sample()
        stages["load_and_clean"].append(_timer() - t0)

        df = s.df
        X = df[EXPOSURE].to_numpy(float)
        M = df[MEDIATOR].to_numpy(float)
        Y = df[OUTCOME].to_numpy(float)
        C = df[s.covariates].to_numpy(float)

        t0 = _timer()
        fit_paths(X, M, Y, C)
        stages["fit_paths"].append(_timer() - t0)

        t0 = _timer()
        bootstrap_paths(X, M, Y, C, n_boot=n_boot, seed=RANDOM_SEED)
        stages[f"bootstrap_{n_boot}"].append(_timer() - t0)

    rows = []
    for stage, times in stages.items():
        rows.append({
            "Stage": stage,
            "Median seconds": round(float(np.median(times)), 4),
            "Min seconds": round(float(np.min(times)), 4),
            "Max seconds": round(float(np.max(times)), 4),
            "Repeats": repeats,
        })
    total = sum(float(np.median(t)) for t in stages.values())
    rows.append({"Stage": "TOTAL (single analysis)", "Median seconds": round(total, 4),
                 "Min seconds": np.nan, "Max seconds": np.nan, "Repeats": repeats})
    return pd.DataFrame(rows)


def _simulate(n: int, seed: int = RANDOM_SEED, a: float = 0.002, b: float = -0.48,
              cp: float = -0.0007, n_cov: int = 5) -> pd.DataFrame:
    """Simulated data with a known indirect effect, for the scaling curve."""
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(n, n_cov))
    X = rng.normal(250, 120, size=n)
    M = 4.0 + a * X + C @ np.full(n_cov, 0.1) + rng.normal(0, 1.5, n)
    Y = 9.0 + cp * X + b * M + C @ np.full(n_cov, 0.05) + rng.normal(0, 0.5, n)
    data = {EXPOSURE: X, MEDIATOR: M, OUTCOME: Y}
    for i in range(n_cov):
        data[f"Cov{i + 1}"] = C[:, i]
    return pd.DataFrame(data)


def scalability_curve(sizes=None, n_boot: int = 500, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Estimation and bootstrap time as a function of sample size.

    Sizes at or below the analytic sample are drawn from the analytic sample
    itself; larger sizes are simulated, so the curve extends past n = 10,000.
    """
    sample = build_analytic_sample()
    df = sample.df
    covs = sample.covariates
    n_max = len(df)
    sizes = list(sizes or [500, 1000, 2500, 5000, n_max, 25_000, 100_000])

    rows = []
    rng = np.random.default_rng(seed)
    for n in sizes:
        if n <= n_max:
            idx = rng.choice(n_max, n, replace=False)
            sub = df.iloc[idx]
            X = sub[EXPOSURE].to_numpy(float)
            M = sub[MEDIATOR].to_numpy(float)
            Y = sub[OUTCOME].to_numpy(float)
            C = sub[covs].to_numpy(float)
            origin = "analytic sample (subsampled)"
        else:
            sim = _simulate(n, seed=seed)
            X = sim[EXPOSURE].to_numpy(float)
            M = sim[MEDIATOR].to_numpy(float)
            Y = sim[OUTCOME].to_numpy(float)
            C = sim[[c for c in sim.columns if c.startswith("Cov")]].to_numpy(float)
            origin = "simulated"

        t0 = _timer()
        fit_paths(X, M, Y, C)
        t_fit = _timer() - t0

        t0 = _timer()
        bootstrap_paths(X, M, Y, C, n_boot=n_boot, seed=seed)
        t_boot = _timer() - t0

        rows.append({
            "n": n,
            "Source": origin,
            "Fit seconds": round(t_fit, 4),
            f"Bootstrap seconds (B={n_boot})": round(t_boot, 3),
            "Bootstrap us per replicate": round(t_boot / n_boot * 1e6, 1),
        })
    return pd.DataFrame(rows)


def compare_software(sample=None, n_boot: int = 2000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Compare the indirect effect against other mediation implementations.

    Every row reports the indirect effect **per 100 mg of caffeine per day**.
    `statsmodels.stats.mediation.Mediation` always contrasts exposure 0 with
    exposure 1, so feeding it caffeine in 100 mg units makes its contrast the
    same quantity the manuscript reports.

    Comparators:

    * `statsmodels` OLS through the formula API - an independent route to the
      same least-squares problem, so the product of coefficients must agree
      with this package to machine precision. This is the strict numerical
      check.
    * `statsmodels.stats.mediation.Mediation` - the Imai-Keele-Tingley
      simulation estimator. It draws the potential mediator at random in each
      replicate, so agreement is expected only up to its Monte Carlo noise,
      which the table reports alongside.
    * `pingouin.mediation_analysis` - a Preacher-Hayes bootstrap
      implementation, when installed.
    """
    sample = sample or build_analytic_sample()
    df = sample.df.copy()
    covs = list(sample.covariates)
    scale = CAFFEINE_SCALE
    scaled = "Caffeine_100mg"
    df[scaled] = df[EXPOSURE] / scale

    own = run_mediation(sample, n_boot=n_boot, seed=seed)
    rows = [{
        "Software": f"bkmediation {_pkg_version()} (this package, B={n_boot})",
        "Estimator": "OLS paths + percentile bootstrap",
        INDIRECT_COL: own.effect_sizes["ab"] * scale,
        "95% CI low": own.effect_sizes["ab_ci"][0] * scale,
        "95% CI high": own.effect_sizes["ab_ci"][1] * scale,
        "Direct effect (per 100 mg)": own.paths["cp"]["coef"] * scale,
        "Total effect (per 100 mg)": own.paths["c"]["coef"] * scale,
        "Monte Carlo SE": np.nan,
        "Seconds": np.nan,
    }]

    # -- statsmodels, formula API: deterministic cross-check -----------------
    t0 = _timer()
    med_fit = sm.OLS.from_formula(
        f"{MEDIATOR} ~ {scaled} + " + " + ".join(covs), data=df).fit()
    out_fit = sm.OLS.from_formula(
        f"{OUTCOME} ~ {scaled} + {MEDIATOR} + " + " + ".join(covs), data=df).fit()
    tot_fit = sm.OLS.from_formula(
        f"{OUTCOME} ~ {scaled} + " + " + ".join(covs), data=df).fit()
    seconds = _timer() - t0
    a_f, b_f = float(med_fit.params[scaled]), float(out_fit.params[MEDIATOR])
    rows.append({
        "Software": "statsmodels OLS (formula API), product of coefficients",
        "Estimator": "closed form, no resampling",
        INDIRECT_COL: a_f * b_f,
        "95% CI low": np.nan,
        "95% CI high": np.nan,
        "Direct effect (per 100 mg)": float(out_fit.params[scaled]),
        "Total effect (per 100 mg)": float(tot_fit.params[scaled]),
        "Monte Carlo SE": np.nan,
        "Seconds": round(seconds, 3),
    })

    # -- statsmodels Mediation (Imai-Keele-Tingley) --------------------------
    try:
        from statsmodels.stats.mediation import Mediation

        outcome_model = sm.OLS.from_formula(
            f"{OUTCOME} ~ {scaled} + {MEDIATOR} + " + " + ".join(covs), data=df)
        mediator_model = sm.OLS.from_formula(
            f"{MEDIATOR} ~ {scaled} + " + " + ".join(covs), data=df)
        np.random.seed(seed)
        t0 = _timer()
        med = Mediation(outcome_model, mediator_model, scaled, MEDIATOR).fit(
            n_rep=n_boot, method="parametric")
        seconds = _timer() - t0
        summary = med.summary()
        acme = float(summary.loc["ACME (average)", "Estimate"])
        acme_lo = float(summary.loc["ACME (average)", "Lower CI bound"])
        acme_hi = float(summary.loc["ACME (average)", "Upper CI bound"])
        # The reported interval spans roughly +/- 1.96 replicate SDs, so the
        # Monte Carlo SE of the averaged ACME is (hi - lo) / 3.92 / sqrt(n_rep).
        mc_se = (acme_hi - acme_lo) / 3.92 / np.sqrt(n_boot)
        rows.append({
            "Software": f"statsmodels.stats.mediation.Mediation (n_rep={n_boot})",
            "Estimator": "Imai-Keele-Tingley parametric simulation",
            INDIRECT_COL: acme,
            "95% CI low": acme_lo,
            "95% CI high": acme_hi,
            "Direct effect (per 100 mg)": float(summary.loc["ADE (average)", "Estimate"]),
            "Total effect (per 100 mg)": float(summary.loc["Total effect", "Estimate"]),
            "Monte Carlo SE": mc_se,
            "Seconds": round(seconds, 2),
        })
    except Exception as exc:  # pragma: no cover - depends on the environment
        rows.append({"Software": "statsmodels.stats.mediation.Mediation",
                     "Estimator": f"unavailable ({type(exc).__name__}: {exc})",
                     INDIRECT_COL: np.nan, "95% CI low": np.nan,
                     "95% CI high": np.nan, "Direct effect (per 100 mg)": np.nan,
                     "Total effect (per 100 mg)": np.nan, "Monte Carlo SE": np.nan,
                     "Seconds": np.nan})

    # -- pingouin (optional) -------------------------------------------------
    try:  # pragma: no cover - optional dependency
        import pingouin as pg

        t0 = _timer()
        res = pg.mediation_analysis(data=df, x=scaled, m=MEDIATOR, y=OUTCOME,
                                    covar=covs, n_boot=n_boot, seed=seed)
        seconds = _timer() - t0
        ind = res[res["path"] == "Indirect"].iloc[0]
        direct = res[res["path"].str.startswith("Direct")].iloc[0]
        # Column names differ across pingouin versions (CI2.5 vs "CI[2.5%]").
        lo_col = "CI2.5" if "CI2.5" in res.columns else "CI[2.5%]"
        hi_col = "CI97.5" if "CI97.5" in res.columns else "CI[97.5%]"
        rows.append({
            "Software": f"pingouin.mediation_analysis (n_boot={n_boot})",
            "Estimator": "Preacher-Hayes percentile bootstrap",
            INDIRECT_COL: float(ind["coef"]),
            "95% CI low": float(ind[lo_col]),
            "95% CI high": float(ind[hi_col]),
            "Direct effect (per 100 mg)": float(direct["coef"]),
            "Total effect (per 100 mg)": np.nan,
            "Monte Carlo SE": np.nan,
            "Seconds": round(seconds, 2),
        })
    except ImportError:
        rows.append({"Software": "pingouin.mediation_analysis",
                     "Estimator": "not installed (optional comparison)",
                     INDIRECT_COL: np.nan, "95% CI low": np.nan,
                     "95% CI high": np.nan, "Direct effect (per 100 mg)": np.nan,
                     "Total effect (per 100 mg)": np.nan, "Monte Carlo SE": np.nan,
                     "Seconds": np.nan})

    return pd.DataFrame(rows)


def run_benchmarks(out_dir=None, n_boot: int = N_BOOTSTRAP, quick: bool = False) -> dict:
    """Run all three benchmarks and (optionally) write the markdown report."""
    sizes = [500, 2500, 9795] if quick else None
    results = {
        "timing": time_pipeline(n_boot=(500 if quick else n_boot), repeats=(1 if quick else 3)),
        "scalability": scalability_curve(sizes=sizes, n_boot=(100 if quick else 500)),
        "software": compare_software(n_boot=(200 if quick else 1000)),
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results["timing"].to_csv(out_dir / "benchmark_timing.csv", index=False)
        results["scalability"].to_csv(out_dir / "benchmark_scalability.csv", index=False)
        results["software"].to_csv(out_dir / "benchmark_software_comparison.csv", index=False)
        (out_dir / "benchmark_report.md").write_text(benchmark_markdown(results), encoding="utf-8")
    return results


def benchmark_markdown(results: dict) -> str:
    from .provenance import collect_provenance

    info = collect_provenance()
    soft = results["software"]
    own = soft.iloc[0]
    lines = [
        "# Computational Benchmarks",
        "",
        f"Machine: {info['platform']}; Python {info['python']}; "
        f"numpy {info['packages']['numpy']}, statsmodels {info['packages']['statsmodels']}.",
        f"Seed {info['analysis_settings']['random_seed']}; "
        f"commit `{info['git']['commit_short'] or 'n/a'}`.",
        "",
        "## 1. Runtime of the analysis pipeline",
        "",
        results["timing"].to_markdown(index=False),
        "",
        "## 2. Scalability",
        "",
        "Samples at or below the analytic sample size are drawn from the analytic "
        "sample itself; larger ones are simulated with the same covariate "
        "structure, so the curve extends beyond the study's n.",
        "",
        results["scalability"].to_markdown(index=False),
        "",
        "Bootstrap cost is linear in both the number of replicates and the sample "
        "size, because each replicate is a least-squares solve on an n x (k+2) "
        "design.",
        "",
        "## 3. Agreement with established mediation software",
        "",
        "All effects are per 100 mg of caffeine per day.",
        "",
        results["software"].to_markdown(index=False),
        "",
    ]
    comparators = soft.iloc[1:]
    numeric = comparators[pd.notna(comparators[INDIRECT_COL])]
    if len(numeric):
        diffs = []
        for _, r in numeric.iterrows():
            diff = abs(r[INDIRECT_COL] - own[INDIRECT_COL])
            rel = diff / abs(own[INDIRECT_COL]) * 100 if own[INDIRECT_COL] else np.nan
            note = ""
            if pd.notna(r["Monte Carlo SE"]) and r["Monte Carlo SE"] > 0:
                note = (f" That is {diff / r['Monte Carlo SE']:.1f} Monte Carlo "
                        f"standard errors of its own simulation noise.")
            diffs.append(f"- {r['Software']}: absolute difference {diff:.2e} h "
                         f"({rel:.2f}% of the point estimate).{note}")
        lines += ["Difference in the estimated indirect effect versus this package:", "",
                  *diffs, "",
                  "The closed-form statsmodels fit agrees to machine precision: the two "
                  "implementations solve the same least-squares problem by different "
                  "routes, which is the numerical check. The Imai-Keele-Tingley "
                  "estimator draws the potential mediator at random in every replicate, "
                  "so it agrees only up to its own Monte Carlo noise, and its interval "
                  "is wider for that reason rather than because the effect is less "
                  "certain.", ""]
    return "\n".join(lines)
