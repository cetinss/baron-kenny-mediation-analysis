"""Baron-Kenny mediation estimation with bootstrap inference.

What this module reports, and why
---------------------------------
* Every path (a, b, c, c') is reported with a standard error, a t statistic,
  a p value AND a 95% confidence interval - both a model-based interval and a
  percentile-bootstrap interval.
* Standard errors are heteroskedasticity-consistent (HC3) by default; the
  classical OLS interval is reported alongside so the two can be compared.
* Coefficients on the exposure are additionally expressed per 100 mg of
  caffeine per day, which is the scale the manuscript reports.
* The primary inferential statement is the percentile-bootstrap interval for
  the indirect effect a*b. The Sobel test is retained only as a secondary,
  normal-theory comparison.
* The proportion mediated is reported WITH a bootstrap interval, because the
  ratio a*b/c is a highly unstable quantity.
* Cohen's f2 is computed for the increment in explained variance when the
  mediator enters the outcome model, and is labelled as exactly that. It is
  NOT a mediation effect size and must not be interpreted as one; the
  mediation effect sizes are the (standardised) indirect effect and the
  proportion mediated.

Nothing in this module identifies a causal effect. With cross-sectional data
and a mediator that is measured (or, in the synthetic dataset, generated) at
the same time as the outcome, a*b is a statistical indirect association under
an assumed ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .config import (
    ALPHA,
    CAFFEINE_SCALE,
    DEFAULT_COV_TYPE,
    EXPOSURE,
    MEDIATOR,
    N_BOOTSTRAP,
    OUTCOME,
    RANDOM_SEED,
)
from .data import AnalyticSample

__all__ = ["MediationResult", "run_mediation", "fit_paths", "bootstrap_paths"]

PATH_LABELS = {
    "a": "a  (Caffeine -> Stress)",
    "b": "b  (Stress -> Sleep | Caffeine)",
    "c": "c  (Caffeine -> Sleep, total)",
    "cp": "c' (Caffeine -> Sleep, direct)",
}
# Paths carried by the exposure: rescaling caffeine by 100 mg multiplies
# these coefficients by 100. Path b is per one stress-scale point and is
# unaffected.
EXPOSURE_PATHS = ("a", "c", "cp")


def _ols(y, X, cov_type="nonrobust"):
    model = sm.OLS(np.asarray(y, float), sm.add_constant(np.asarray(X, float)))
    return model.fit(cov_type=cov_type) if cov_type != "nonrobust" else model.fit()


def fit_paths(X, M, Y, C, cov_type: str = DEFAULT_COV_TYPE, alpha: float = ALPHA):
    """Fit the three Baron-Kenny regressions and extract the four paths.

    Returns (paths, models) where `paths` maps 'a'/'b'/'c'/'cp' to a dict with
    classical and robust standard errors, t, p and confidence limits.

        Step 1  Y = i1 + c*X  + g'C      (total effect)
        Step 2  M = i2 + a*X  + d'C      (path a)
        Step 3  Y = i3 + c'*X + b*M + f'C   (path b and direct effect)
    """
    X = np.asarray(X, float).reshape(-1, 1)
    M = np.asarray(M, float).reshape(-1, 1)
    Y = np.asarray(Y, float)
    C = np.asarray(C, float)

    design_total = np.column_stack([X, C])
    design_med = np.column_stack([X, M, C])

    fits = {}
    for key, (y, design, cov) in {
        "total": (Y, design_total, cov_type),
        "a": (M.ravel(), design_total, cov_type),
        "med": (Y, design_med, cov_type),
    }.items():
        fits[key] = {
            "robust": _ols(y, design, cov),
            "classical": _ols(y, design, "nonrobust"),
        }

    spec = {
        "a": ("a", 1),
        "c": ("total", 1),
        "cp": ("med", 1),
        "b": ("med", 2),
    }
    paths = {}
    for name, (fit_key, idx) in spec.items():
        rob, cls = fits[fit_key]["robust"], fits[fit_key]["classical"]
        ci_r = rob.conf_int(alpha=alpha)[idx]
        ci_c = cls.conf_int(alpha=alpha)[idx]
        paths[name] = {
            "label": PATH_LABELS[name],
            "coef": float(rob.params[idx]),
            "se": float(rob.bse[idx]),
            "t": float(rob.tvalues[idx]),
            "p": float(rob.pvalues[idx]),
            "ci_lo": float(ci_r[0]),
            "ci_hi": float(ci_r[1]),
            "se_classical": float(cls.bse[idx]),
            "t_classical": float(cls.tvalues[idx]),
            "p_classical": float(cls.pvalues[idx]),
            "ci_lo_classical": float(ci_c[0]),
            "ci_hi_classical": float(ci_c[1]),
        }

    models = {
        "total": fits["total"]["robust"],
        "a": fits["a"]["robust"],
        "med": fits["med"]["robust"],
        "total_classical": fits["total"]["classical"],
        "a_classical": fits["a"]["classical"],
        "med_classical": fits["med"]["classical"],
    }
    return paths, models


def bootstrap_paths(X, M, Y, C, n_boot: int = N_BOOTSTRAP, seed: int = RANDOM_SEED):
    """Nonparametric (case-resampling) bootstrap of all four paths.

    Each replicate refits the two regressions by least squares on the
    resampled cases and stores a, b, c, c', a*b and the proportion mediated.
    Least squares is solved with `numpy.linalg.lstsq` rather than a full
    statsmodels fit: the point estimates are identical to machine precision
    but the loop is roughly an order of magnitude faster, which is what makes
    5,000 replicates cheap (see `benchmark.py`).

    The random stream is `numpy.random.RandomState(seed)`, which reproduces
    the published intervals exactly.
    """
    X = np.asarray(X, float).ravel()
    M = np.asarray(M, float).ravel()
    Y = np.asarray(Y, float)
    C = np.asarray(C, float)
    n = len(Y)
    ones = np.ones(n)

    rs = np.random.RandomState(seed)
    out = np.empty((n_boot, 6))
    kept = 0
    for _ in range(n_boot):
        idx = rs.choice(n, n, replace=True)
        Xb, Mb, Yb, Cb = X[idx], M[idx], Y[idx], C[idx]
        d_total = np.column_stack([ones, Xb, Cb])
        d_med = np.column_stack([ones, Xb, Mb, Cb])
        try:
            beta_a = np.linalg.lstsq(d_total, Mb, rcond=None)[0]
            beta_c = np.linalg.lstsq(d_total, Yb, rcond=None)[0]
            beta_m = np.linalg.lstsq(d_med, Yb, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        a_b, c_b = beta_a[1], beta_c[1]
        cp_b, b_b = beta_m[1], beta_m[2]
        ab_b = a_b * b_b
        out[kept] = (a_b, b_b, c_b, cp_b, ab_b, ab_b / c_b * 100 if c_b != 0 else np.nan)
        kept += 1
    out = out[:kept]
    return {
        "a": out[:, 0],
        "b": out[:, 1],
        "c": out[:, 2],
        "cp": out[:, 3],
        "ab": out[:, 4],
        "prop": out[:, 5],
        "n_valid": kept,
    }


def _pct_ci(draws, alpha: float = ALPHA):
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


@dataclass
class MediationResult:
    """Everything the manuscript reports about one mediation model."""

    n: int
    paths: dict
    boot: dict
    effect_sizes: dict
    model_fit: dict
    conditions: dict
    mediation_type: str
    covariates: list
    cov_type: str
    n_boot: int
    seed: int
    scale: float = CAFFEINE_SCALE
    is_synthetic: bool = True
    labels: dict = field(default_factory=lambda: {"X": EXPOSURE, "M": MEDIATOR, "Y": OUTCOME})
    models: dict = field(default_factory=dict, repr=False)

    # -- tables --------------------------------------------------------------
    def paths_table(self) -> pd.DataFrame:
        """Table 2 of the manuscript: paths with SEs, p values and 95% CIs.

        `Coef (per 100 mg)` rescales the three exposure paths; path b is per
        one point of the stress scale and is reported unchanged.
        """
        rows = []
        for name in ("a", "b", "c", "cp"):
            p = self.paths[name]
            k = self.scale if name in EXPOSURE_PATHS else 1.0
            blo, bhi = _pct_ci(self.boot[name])
            rows.append({
                "Path": name if name != "cp" else "c'",
                "Description": p["label"],
                "Coef": p["coef"],
                "SE (HC3)": p["se"],
                "t": p["t"],
                "p": p["p"],
                "95% CI low": p["ci_lo"],
                "95% CI high": p["ci_hi"],
                "SE (classical)": p["se_classical"],
                "95% CI low (classical)": p["ci_lo_classical"],
                "95% CI high (classical)": p["ci_hi_classical"],
                "95% CI low (bootstrap)": blo,
                "95% CI high (bootstrap)": bhi,
                "Scale": "per 100 mg/day" if name in EXPOSURE_PATHS else "per stress point",
                "Coef (rescaled)": p["coef"] * k,
                "CI low (rescaled)": p["ci_lo"] * k,
                "CI high (rescaled)": p["ci_hi"] * k,
            })
        return pd.DataFrame(rows)

    def effects_table(self) -> pd.DataFrame:
        """Indirect / direct / total effects with bootstrap intervals."""
        e = self.effect_sizes
        rows = [
            ("Indirect (a*b)", e["ab"], e["ab_ci"][0], e["ab_ci"][1],
             "percentile bootstrap"),
            ("Direct (c')", self.paths["cp"]["coef"],
             _pct_ci(self.boot["cp"])[0], _pct_ci(self.boot["cp"])[1],
             "percentile bootstrap"),
            ("Total (c)", self.paths["c"]["coef"],
             _pct_ci(self.boot["c"])[0], _pct_ci(self.boot["c"])[1],
             "percentile bootstrap"),
            ("Indirect, per 100 mg", e["ab"] * self.scale,
             e["ab_ci"][0] * self.scale, e["ab_ci"][1] * self.scale,
             "percentile bootstrap"),
            ("Indirect, completely standardised", e["ab_std"],
             np.nan, np.nan, "point estimate"),
            ("Proportion mediated (%)", e["prop"], e["prop_ci"][0], e["prop_ci"][1],
             "percentile bootstrap"),
        ]
        return pd.DataFrame(rows, columns=["Effect", "Estimate", "95% CI low",
                                           "95% CI high", "CI method"])

    # -- convenience ---------------------------------------------------------
    @property
    def ab(self) -> float:
        return self.effect_sizes["ab"]

    @property
    def ab_ci(self):
        return self.effect_sizes["ab_ci"]

    @property
    def indirect_is_significant(self) -> bool:
        lo, hi = self.ab_ci
        return not (lo <= 0 <= hi)

    def to_dict(self) -> dict:
        """Flat dictionary for CSV export."""
        row = {"n": self.n, "cov_type": self.cov_type, "n_boot": self.n_boot,
               "seed": self.seed, "gender_covariates": ";".join(self.covariates),
               "mediation_type": self.mediation_type,
               "is_synthetic": self.is_synthetic}
        for name, p in self.paths.items():
            for k, v in p.items():
                if k == "label":
                    continue
                row[f"{name}_{k}"] = v
            lo, hi = _pct_ci(self.boot[name])
            row[f"{name}_boot_lo"], row[f"{name}_boot_hi"] = lo, hi
        for k, v in self.effect_sizes.items():
            if isinstance(v, tuple):
                row[f"{k}_lo"], row[f"{k}_hi"] = v
            else:
                row[k] = v
        row.update(self.model_fit)
        row.update({f"cond_{k}": v for k, v in self.conditions.items()})
        return row

    def legacy_dict(self) -> dict:
        """Keys used by the figure code in `src/main.py` (kept stable)."""
        p = self.paths
        e = self.effect_sizes
        return {
            "n": self.n,
            "a": p["a"]["coef"], "a_se": p["a"]["se"], "a_t": p["a"]["t"],
            "a_p": p["a"]["p"], "a_std": e["a_std"],
            "b": p["b"]["coef"], "b_se": p["b"]["se"], "b_t": p["b"]["t"],
            "b_p": p["b"]["p"], "b_std": e["b_std"],
            "c": p["c"]["coef"], "c_se": p["c"]["se"], "c_t": p["c"]["t"],
            "c_p": p["c"]["p"], "c_std": e["c_std"],
            "cp": p["cp"]["coef"], "cp_se": p["cp"]["se"], "cp_t": p["cp"]["t"],
            "cp_p": p["cp"]["p"], "cp_std": e["cp_std"],
            "ab": e["ab"], "ab_std": e["ab_std"],
            "ci_lo": e["ab_ci"][0], "ci_hi": e["ab_ci"][1],
            "ci_sig": self.indirect_is_significant,
            "sob_z": e["sobel_z"], "sob_p": e["sobel_p"],
            "r2_total": self.model_fit["r2_total"],
            "r2_total_adj": self.model_fit["r2_total_adj"],
            "f_total": self.model_fit["f_total"],
            "f_total_p": self.model_fit["f_total_p"],
            "r2_a": self.model_fit["r2_a"],
            "r2_med": self.model_fit["r2_med"],
            "r2_med_adj": self.model_fit["r2_med_adj"],
            "f_med": self.model_fit["f_med"],
            "f_med_p": self.model_fit["f_med_p"],
            "f2": e["f2_mediator_increment"],
            "prop": e["prop"],
            "prop_ci": e["prop_ci"],
            "cond": list(self.conditions.values()),
            "all_ok": all(self.conditions.values()),
            "med_type": self.mediation_type,
            "boot": self.boot["ab"],
            "model1": self.models.get("total"),
            "model2": self.models.get("a"),
            "model3": self.models.get("med"),
        }

    def summary(self) -> str:
        t = self.paths_table()
        lines = [
            f"Baron-Kenny mediation  (n = {self.n:,}, {self.cov_type} SEs, "
            f"B = {self.n_boot:,}, seed = {self.seed})",
            "-" * 78,
        ]
        for _, r in t.iterrows():
            lines.append(
                f"  {r['Path']:<3} {r['Coef']:>12.6f}  SE {r['SE (HC3)']:.6f}  "
                f"p {r['p']:.2e}  95% CI [{r['95% CI low']:.6f}, {r['95% CI high']:.6f}]"
                f"   [{r['Coef (rescaled)']:+.4f} {r['Scale']}]"
            )
        e = self.effect_sizes
        lines += [
            "-" * 78,
            f"  Indirect effect a*b = {e['ab']:.6f}  "
            f"95% bootstrap CI [{e['ab_ci'][0]:.6f}, {e['ab_ci'][1]:.6f}]",
            f"  Indirect effect per 100 mg = {e['ab'] * self.scale:+.4f} h  "
            f"[{e['ab_ci'][0] * self.scale:+.4f}, {e['ab_ci'][1] * self.scale:+.4f}]",
            f"  Completely standardised indirect effect = {e['ab_std']:.4f}",
            f"  Proportion mediated = {e['prop']:.1f}% "
            f"[{e['prop_ci'][0]:.1f}%, {e['prop_ci'][1]:.1f}%]",
            f"  Sobel z = {e['sobel_z']:.3f} (secondary), p = {e['sobel_p']:.2e}",
            f"  f2 for the mediator increment = {e['f2_mediator_increment']:.4f} "
            f"(variance-explained increment, NOT a mediation effect size)",
            f"  Conclusion: {self.mediation_type}",
        ]
        if self.is_synthetic:
            lines.append("  NOTE: synthetic data - statistical demonstration, not causal evidence.")
        return "\n".join(lines)


def run_mediation(
    sample,
    covariates=None,
    exposure: str = EXPOSURE,
    mediator: str = MEDIATOR,
    outcome: str = OUTCOME,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    cov_type: str = DEFAULT_COV_TYPE,
    alpha: float = ALPHA,
    scale: float = CAFFEINE_SCALE,
    verbose: bool = False,
) -> MediationResult:
    """Estimate the Baron-Kenny model on an :class:`AnalyticSample` or DataFrame.

    Parameters
    ----------
    sample
        An :class:`~bkmediation.data.AnalyticSample`, or a plain DataFrame (in
        which case `covariates` must be given or the package default is used).
    covariates
        Covariate column names. Defaults to the sample's own covariate list.
    cov_type
        Covariance estimator for the reported SEs; "HC3" (default) or
        "nonrobust". Point estimates do not depend on this choice.
    """
    from .config import DEFAULT_COVARIATES

    if isinstance(sample, AnalyticSample):
        df = sample.df
        covariates = list(covariates or sample.covariates)
        is_synthetic = sample.is_synthetic
    else:
        df = sample
        covariates = list(covariates or DEFAULT_COVARIATES)
        missing = [c for c in covariates if c not in df.columns]
        if missing:
            raise KeyError(f"Covariates missing from the data: {missing}")
        is_synthetic = True

    X = df[exposure].to_numpy(float)
    M = df[mediator].to_numpy(float)
    Y = df[outcome].to_numpy(float)
    C = df[covariates].to_numpy(float)
    n = len(df)

    paths, models = fit_paths(X, M, Y, C, cov_type=cov_type, alpha=alpha)
    boot = bootstrap_paths(X, M, Y, C, n_boot=n_boot, seed=seed)

    a, b = paths["a"]["coef"], paths["b"]["coef"]
    c, cp = paths["c"]["coef"], paths["cp"]["coef"]
    ab = a * b
    ab_ci = _pct_ci(boot["ab"], alpha)
    prop = ab / c * 100 if c != 0 else np.nan
    prop_ci = _pct_ci(boot["prop"], alpha)

    sobel_se = float(np.sqrt(a ** 2 * paths["b"]["se"] ** 2 + b ** 2 * paths["a"]["se"] ** 2))
    sobel_z = ab / sobel_se if sobel_se > 0 else np.nan
    sobel_p = float(2 * (1 - stats.norm.cdf(abs(sobel_z)))) if np.isfinite(sobel_z) else np.nan

    sx, sm_, sy = X.std(ddof=1), M.std(ddof=1), Y.std(ddof=1)
    a_std, b_std = a * sx / sm_, b * sm_ / sy
    c_std, cp_std = c * sx / sy, cp * sx / sy

    m_total, m_a, m_med = models["total_classical"], models["a_classical"], models["med_classical"]
    r2_total, r2_med = float(m_total.rsquared), float(m_med.rsquared)
    f2 = (r2_med - r2_total) / (1 - r2_med) if r2_med < 1 else np.nan

    model_fit = {
        "r2_total": r2_total,
        "r2_total_adj": float(m_total.rsquared_adj),
        "f_total": float(m_total.fvalue),
        "f_total_p": float(m_total.f_pvalue),
        "r2_a": float(m_a.rsquared),
        "r2_med": r2_med,
        "r2_med_adj": float(m_med.rsquared_adj),
        "f_med": float(m_med.fvalue),
        "f_med_p": float(m_med.f_pvalue),
        "delta_r2": r2_med - r2_total,
    }

    effect_sizes = {
        "ab": ab,
        "ab_ci": ab_ci,
        "ab_per_scale": ab * scale,
        "ab_std": a_std * b_std,
        "ab_partially_std": ab / sy,
        "a_std": a_std, "b_std": b_std, "c_std": c_std, "cp_std": cp_std,
        "prop": prop,
        "prop_ci": prop_ci,
        "sobel_z": sobel_z,
        "sobel_p": sobel_p,
        # Reported for completeness only; see the module docstring.
        "f2_mediator_increment": f2,
    }

    conditions = {
        "c_significant": paths["c"]["p"] < alpha,
        "a_significant": paths["a"]["p"] < alpha,
        "b_significant": paths["b"]["p"] < alpha,
        "direct_effect_reduced": abs(cp) < abs(c),
    }
    ci_sig = not (ab_ci[0] <= 0 <= ab_ci[1])
    if all(conditions.values()) and ci_sig:
        mediation_type = "FULL MEDIATION" if paths["cp"]["p"] >= alpha else "PARTIAL MEDIATION"
    else:
        mediation_type = "NO MEDIATION"

    result = MediationResult(
        n=n,
        paths=paths,
        boot=boot,
        effect_sizes=effect_sizes,
        model_fit=model_fit,
        conditions=conditions,
        mediation_type=mediation_type,
        covariates=covariates,
        cov_type=cov_type,
        n_boot=n_boot,
        seed=seed,
        scale=scale,
        is_synthetic=is_synthetic,
        labels={"X": exposure, "M": mediator, "Y": outcome},
        models=models,
    )
    if verbose:
        print(result.summary())
    return result
