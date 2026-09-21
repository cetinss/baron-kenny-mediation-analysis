"""
================================================================================
LEMURS External Validation of the Baron-Kenny Mediation Analysis
Caffeine Exposure -> Perceived Stress (PSS-10) -> Sleep Duration

Purpose
-------
Independent-data check of the mediation pattern found on the Global Coffee
Health synthetic dataset (src/main.py), using LEMURS survey_syn_5.csv as an
external reference dataset.

Dataset status
--------------
LEMURS (survey_syn_5.csv) is also technically synthetic, but of a different
kind than the Global Coffee Health data: it is a Differentially Private (DP)
synthetic release (AIM algorithm, epsilon = 5) derived from a real study of
~600 university students (Oura Ring + weekly surveys). The real underlying
data cannot be shared for re-identification reasons, but the generation
method and its statistical fidelity to the real data were validated in a
peer-reviewed publication (Ghasemizade et al., 2025, JAMIA Open). It is
treated here as an independently-generated, formally privacy-protected
synthetic reference dataset -- not as "real data".

Two methodological deviations from the original pipeline, decided with the
user after the feature-matching investigation (see
outputs/feature_matching_report.md):

  1. No `record_id` column exists in survey_syn_5.csv (despite the LEMURS
     README documenting one). Person-level averaging / cluster-robust SE
     by participant is therefore not possible. Each row (person-week) is
     treated as an independent cross-sectional observation instead.
  2. The IPAQ activity duration columns (F1_Activity_hoursminutes1/4) have
     an ambiguous category (vigorous vs. moderate cannot be determined from
     the codebook). The primary covariate uses the unambiguous "total
     active days/week" count; the duration-based version is reported only
     as a labeled sensitivity check.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
N_BOOTSTRAP = 5000
ALPHA = 0.05

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "lemurs"
OUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)

plt.style.use("default")
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.3,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "axes.linewidth": 1.2, "grid.alpha": 0.3,
})

COLORS = {
    "caffeine": "#8B4513", "stress": "#E67E22", "sleep": "#1F4788",
    "positive": "#27AE60", "negative": "#E74C3C", "neutral": "#7F8C8D",
    "bg": "#F8F9FA", "dark": "#2C3E50",
}
_sig = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""

CAFFEINE_FLAGS = ["F1_Caffeine_coffee", "F1_Caffeine_tea", "F1_Caffeine_soda",
                  "F1_Caffeine_energydrink", "F1_Caffeine_workoutdrink",
                  "F1_Caffeine_otherdrink"]
PSS_ITEMS = ["F1_PSS_stressupset", "F1_PSS_stresscontrol", "F1_PSS_stressnervous",
             "F1_PSS_stressconfident", "F1_PSS_stressthings", "F1_PSS_stresscope",
             "F1_PSS_stressirritations", "F1_PSS_stressontop",
             "F1_PSS_stressangered", "F1_PSS_stressdifficulties"]
PSS_REVERSE = ["F1_PSS_stressconfident", "F1_PSS_stressthings",
               "F1_PSS_stressirritations", "F1_PSS_stressontop"]
ACTIVITY_DAY_COLS = ["F1_Activity_numvigorous", "F1_Activity_nummoderate",
                     "F1_Activity_numwalking"]


# =============================================================================
# 1. DATA PIPELINE
# =============================================================================

def load_and_build():
    print("=" * 70)
    print("STEP 1: LOAD LEMURS DATA AND BUILD ANALYSIS VARIABLES")
    print("=" * 70)

    raw = pd.read_csv(DATA_DIR / "survey_syn_5.csv")
    print(f"Raw data loaded: {len(raw):,} person-week rows, {len(raw.columns)} columns")

    df = raw.copy()

    # Y: sleep duration -- direct match (F1_Sleep_sleephours)
    df["sleep_hours"] = df["F1_Sleep_sleephours"]

    # M: PSS-10 total score. Reverse-code the 4 positively-worded items
    # (4 - x on the 0-4 scale; confirmed against codebook response coding).
    for col in PSS_REVERSE:
        df[col + "_r"] = 4 - df[col]
    pss_cols_scored = [c + "_r" if c in PSS_REVERSE else c for c in PSS_ITEMS]
    df["pss_score"] = df[pss_cols_scored].sum(axis=1, skipna=False)

    # X: caffeine exposure score (0-6) = count of caffeinated drink types
    # consumed this week. NOT a dose (no mg data exists in LEMURS).
    df["caffeine_exposure_score"] = df[CAFFEINE_FLAGS].sum(axis=1, skipna=False)

    # Covariate (primary): total active days/week, unambiguous IPAQ proxy.
    df["activity_days"] = df[ACTIVITY_DAY_COLS].sum(axis=1, skipna=False)

    # Covariate (sensitivity only): duration-based proxy with an ambiguous
    # activity category (see module docstring / limitations.md).
    df["activity_hours_sensitivity"] = (
        df["F1_Activity_numwalking"] * (df["F1_Activity_hoursminutes2"] + df["F1_Activity_hoursminutes3"] / 60)
        + df["F1_Activity_numvigorous"] * (df["F1_Activity_hoursminutes1"] + df["F1_Activity_hoursminutes4"] / 60)
    )

    key_vars = ["sleep_hours", "pss_score", "caffeine_exposure_score", "activity_days"]
    n_before = len(df)
    df_clean = df.dropna(subset=key_vars).copy()
    print(f"After missing-value removal on [{', '.join(key_vars)}]: "
          f"{len(df_clean):,} (dropped {n_before - len(df_clean):,})")

    # Outlier removal (IQR x 1.5) on the continuous variables.
    # caffeine_exposure_score is a bounded 0-6 count (like Stress_Score in
    # the original pipeline) and is not IQR-filtered.
    continuous = ["sleep_hours", "pss_score", "activity_days"]
    df_clean = _remove_outliers_iqr(df_clean, continuous)

    print(f"\nFinal analytic sample: n = {len(df_clean):,}")
    print(f"Total rows removed: {len(raw) - len(df_clean):,} "
          f"({(len(raw) - len(df_clean)) / len(raw) * 100:.1f}%)")
    return df_clean


def _remove_outliers_iqr(df, columns, k=1.5):
    out = df.copy()
    for col in columns:
        Q1, Q3 = out[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lo, hi = Q1 - k * IQR, Q3 + k * IQR
        before = len(out)
        out = out[(out[col] >= lo) & (out[col] <= hi)]
        rm = before - len(out)
        if rm > 0:
            print(f"  {col}: {rm} outliers removed (range: {lo:.2f} - {hi:.2f})")
    return out


# =============================================================================
# 2. DESCRIPTIVES
# =============================================================================

def compute_descriptives(df):
    print("\n" + "=" * 70)
    print("STEP 2: DESCRIPTIVE STATISTICS")
    print("=" * 70)
    info = {
        "caffeine_exposure_score": "Caffeine Exposure Score (0-6 drink types)",
        "pss_score":               "PSS-10 Total Score (0-40)",
        "sleep_hours":             "Sleep Duration (hours/night)",
        "activity_days":           "Active Days/Week (0-21)",
    }
    rows = []
    for var, label in info.items():
        s = df[var]
        rows.append({
            "Variable": label, "n": int(s.count()),
            "Mean": round(s.mean(), 3), "SD": round(s.std(), 3),
            "Median": round(s.median(), 2),
            "Min": round(s.min(), 2), "Max": round(s.max(), 2),
        })
    desc = pd.DataFrame(rows)
    print(desc.to_string(index=False))
    desc.to_csv(OUT_DIR / "lemurs_descriptive_statistics.csv", index=False)
    return desc


# =============================================================================
# 3. MEDIATION PIPELINE (mirrors src/main.py Baron-Kenny logic)
# =============================================================================

def run_mediation(df, cov_col="activity_days", label="PRIMARY"):
    print("\n" + "=" * 70)
    print(f"STEP 3: BARON-KENNY MEDIATION ANALYSIS -- {label} (covariate = {cov_col})")
    print("=" * 70)

    X = df["caffeine_exposure_score"].values.astype(float)
    M = df["pss_score"].values.astype(float)
    Y = df["sleep_hours"].values.astype(float)
    C = df[[cov_col]].values.astype(float)
    n = len(df)

    # Mean-center continuous predictors. In a model with no interaction or
    # product terms, centring shifts only the intercepts: every slope (a, b,
    # c, c') and therefore the indirect effect are numerically identical to
    # the uncentred main analysis in src/main.py. Centring is kept here for
    # interpretability of the intercepts, and tests/test_mediation.py asserts
    # the invariance, so that the text and the code agree on this point.
    Xc = X - X.mean()
    Mc = M - M.mean()
    Cc = C - C.mean(axis=0)

    # Step 1: Total effect
    X1 = sm.add_constant(np.column_stack([Xc, Cc]))
    m1 = sm.OLS(Y, X1).fit()
    c, c_se, c_t, c_p = m1.params[1], m1.bse[1], m1.tvalues[1], m1.pvalues[1]
    r2_1, r2_1a = m1.rsquared, m1.rsquared_adj
    f1, f1_p = m1.fvalue, m1.f_pvalue

    # Step 2: Path a
    X2 = sm.add_constant(np.column_stack([Xc, Cc]))
    m2 = sm.OLS(Mc, X2).fit()
    a, a_se, a_t, a_p = m2.params[1], m2.bse[1], m2.tvalues[1], m2.pvalues[1]
    r2_2 = m2.rsquared

    # Step 3-4: Path b + direct effect
    X3 = sm.add_constant(np.column_stack([Xc, Mc, Cc]))
    m3 = sm.OLS(Y, X3).fit()
    cp, cp_se, cp_t, cp_p = m3.params[1], m3.bse[1], m3.tvalues[1], m3.pvalues[1]
    b, b_se, b_t, b_p = m3.params[2], m3.bse[2], m3.tvalues[2], m3.pvalues[2]
    r2_3, r2_3a = m3.rsquared, m3.rsquared_adj
    f3, f3_p = m3.fvalue, m3.f_pvalue

    ab = a * b
    sob_se = np.sqrt(a**2 * b_se**2 + b**2 * a_se**2)
    sob_z = ab / sob_se if sob_se > 0 else 0
    sob_p = 2 * (1 - stats.norm.cdf(abs(sob_z)))

    print(f"\n  Bootstrap ({N_BOOTSTRAP:,} resamples) ...")
    boot = np.empty(N_BOOTSTRAP)
    ok = 0
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(n, n, replace=True)
        Xb, Mb, Yb, Cb = Xc[idx], Mc[idx], Y[idx], Cc[idx]
        try:
            a_b = sm.OLS(Mb, sm.add_constant(np.column_stack([Xb, Cb]))).fit().params[1]
            b_b = sm.OLS(Yb, sm.add_constant(np.column_stack([Xb, Mb, Cb]))).fit().params[2]
            boot[ok] = a_b * b_b
            ok += 1
        except Exception:
            continue
    boot = boot[:ok]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    ci_sig = not (ci_lo <= 0 <= ci_hi)

    cond = [c_p < ALPHA, a_p < ALPHA, b_p < ALPHA, abs(cp) < abs(c)]
    all_ok = all(cond)
    if all_ok and ci_sig:
        med_type = "FULL MEDIATION" if cp_p >= ALPHA else "PARTIAL MEDIATION"
    else:
        med_type = "NO MEDIATION"
    prop = (ab / c * 100) if c != 0 else 0

    sx, sm_, sy = X.std(), M.std(), Y.std()
    a_std = a * sx / sm_
    b_std = b * sm_ / sy
    c_std = c * sx / sy
    cp_std = cp * sx / sy
    ab_std = a_std * b_std
    f2 = (r2_3 - r2_1) / (1 - r2_3) if r2_3 < 1 else np.nan

    predictors = np.column_stack([Xc, Mc, Cc])
    Xvif = sm.add_constant(predictors)
    vif_vals = [variance_inflation_factor(Xvif, i) for i in range(1, Xvif.shape[1])]
    vif_names = ["caffeine_exposure_score", "pss_score", cov_col]
    dw = durbin_watson(m3.resid)

    print(f"  c  (total)  = {c:.6f}  p = {c_p:.3e} {_sig(c_p)}")
    print(f"  a  (X->M)   = {a:.6f}  p = {a_p:.3e} {_sig(a_p)}")
    print(f"  b  (M->Y|X) = {b:.6f}  p = {b_p:.3e} {_sig(b_p)}")
    print(f"  c' (direct) = {cp:.6f}  p = {cp_p:.3e} {_sig(cp_p)}")
    print(f"  Indirect (a*b) = {ab:.6f}  95% CI [{ci_lo:.6f}, {ci_hi:.6f}]  sig={ci_sig}")
    print(f"  Proportion mediated = {prop:.1f}%")
    print(f"  R2 total={r2_1:.4f}  R2 mediated={r2_3:.4f}  Cohen f2={f2:.4f}")
    print(f"  Max VIF = {max(vif_vals):.2f}   Durbin-Watson = {dw:.3f}")
    print(f"  => {med_type}")

    R = dict(
        n=n, cov_col=cov_col, label=label,
        a=a, a_se=a_se, a_t=a_t, a_p=a_p, a_std=a_std,
        b=b, b_se=b_se, b_t=b_t, b_p=b_p, b_std=b_std,
        c=c, c_se=c_se, c_t=c_t, c_p=c_p, c_std=c_std,
        cp=cp, cp_se=cp_se, cp_t=cp_t, cp_p=cp_p, cp_std=cp_std,
        ab=ab, ab_std=ab_std,
        ci_lo=ci_lo, ci_hi=ci_hi, ci_sig=ci_sig,
        sob_z=sob_z, sob_p=sob_p,
        r2_total=r2_1, r2_total_adj=r2_1a, f_total=f1, f_total_p=f1_p,
        r2_a=r2_2,
        r2_med=r2_3, r2_med_adj=r2_3a, f_med=f3, f_med_p=f3_p,
        f2=f2, prop=prop,
        vif=dict(zip(vif_names, vif_vals)), dw=dw,
        cond=cond, all_ok=all_ok, med_type=med_type,
        boot=boot, model3=m3,
    )
    return R


# =============================================================================
# 4. FIGURES
# =============================================================================

def fig_distributions(df):
    print("\nGenerating Figure: Variable Distributions ...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle("LEMURS Validation Sample - Key Variable Distributions",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    counts = df["caffeine_exposure_score"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=COLORS["caffeine"], edgecolor="black")
    ax.set_xlabel("Caffeine Exposure Score (0-6 drink types)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("A. Caffeine Exposure (X, proxy)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    ax.hist(df["pss_score"], bins=30, color=COLORS["stress"], alpha=0.8, edgecolor="black")
    ax.axvline(df["pss_score"].mean(), color="red", ls="--", lw=2,
               label=f'Mean = {df["pss_score"].mean():.1f}')
    ax.set_xlabel("PSS-10 Total Score (0-40)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("B. Perceived Stress (M, PSS-10)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    ax.hist(df["sleep_hours"], bins=30, color=COLORS["sleep"], alpha=0.8, edgecolor="black")
    ax.axvline(df["sleep_hours"].mean(), color="red", ls="--", lw=2,
               label=f'Mean = {df["sleep_hours"].mean():.2f} h')
    ax.set_xlabel("Sleep Duration (hours/night)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("C. Sleep Duration (Y)", fontweight="bold", color=COLORS["sleep"])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    counts = df["activity_days"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=COLORS["positive"], edgecolor="black")
    ax.set_xlabel("Active Days/Week (0-21, covariate)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("D. Physical Activity (covariate)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "LEMURS_Fig1_Distributions.png", facecolor="white")
    plt.close()
    print("  Saved -> LEMURS_Fig1_Distributions.png")


def fig_path_diagram(R):
    print("Generating Figure: Mediation Path Diagram ...")
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    fig.suptitle("LEMURS Validation - Baron-Kenny Mediation Model\n"
                 "Caffeine Exposure -> Perceived Stress (PSS-10) -> Sleep Duration",
                 fontsize=14, fontweight="bold", y=0.98)

    for xy, label, sub, color in [
        ((0.3, 4.0), "CAFFEINE\nEXPOSURE (0-6)", "(Proxy IV)", COLORS["caffeine"]),
        ((3.9, 6.8), "PSS-10\nSTRESS SCORE", "(Mediator)", COLORS["stress"]),
        ((7.5, 4.0), "SLEEP\nDURATION (h)", "(Outcome)", COLORS["sleep"]),
    ]:
        box = FancyBboxPatch(xy, 2.2, 1.8, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="black", lw=2.5)
        ax.add_patch(box)
        cx, cy = xy[0] + 1.1, xy[1] + 1.05
        ax.text(cx, cy, label, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="white")
        ax.text(cx, cy - 0.45, sub, ha="center", va="center", fontsize=8,
                color="white", style="italic")

    arr_a = FancyArrowPatch((2.5, 5.3), (3.9, 7.3), arrowstyle="->",
                            mutation_scale=30, lw=3, color="#16A085", zorder=5)
    ax.add_patch(arr_a)
    ax.text(2.7, 6.6, f"a = {R['a']:.4f}\np = {R['a_p']:.3e}{_sig(R['a_p'])}",
            ha="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#ECFDF5", ec="#16A085", lw=1.5))

    arr_b = FancyArrowPatch((6.1, 7.3), (7.5, 5.3), arrowstyle="->",
                            mutation_scale=30, lw=3, color="#C0392B", zorder=5)
    ax.add_patch(arr_b)
    ax.text(7.3, 6.6, f"b = {R['b']:.4f}\np = {R['b_p']:.3e}{_sig(R['b_p'])}",
            ha="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#FADBD8", ec="#C0392B", lw=1.5))

    arr_c = FancyArrowPatch((2.5, 5.5), (7.5, 5.5), arrowstyle="->",
                            mutation_scale=25, lw=2, color="gray", ls="--", alpha=0.7)
    ax.add_patch(arr_c)
    ax.text(5.0, 5.85, f"c = {R['c']:.4f} (Total Effect)", ha="center", fontsize=9,
            style="italic", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray"))

    arr_cp = FancyArrowPatch((2.5, 4.5), (7.5, 4.5), arrowstyle="->",
                             mutation_scale=30, lw=3, color="#8E44AD", zorder=4)
    ax.add_patch(arr_cp)
    ax.text(5.0, 3.85, f"c' = {R['cp']:.4f} (Direct Effect)", ha="center", fontsize=10,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", fc="#F4ECF7", ec="#8E44AD", lw=2))

    rbox = FancyBboxPatch((0.3, 0.5), 9.4, 1.6, boxstyle="round,pad=0.15",
                          fc=COLORS["bg"], ec=COLORS["dark"], lw=2)
    ax.add_patch(rbox)
    rtxt = (f"Indirect Effect (a x b) = {R['ab']:.4f}   |   "
            f"95% Bootstrap CI = [{R['ci_lo']:.4f}, {R['ci_hi']:.4f}]   |   "
            f"Significant: {'YES' if R['ci_sig'] else 'NO'}\n"
            f"Proportion Mediated = {R['prop']:.1f}%   |   "
            f"Conclusion: {R['med_type']}")
    ax.text(5.0, 1.3, rtxt, ha="center", va="center", fontsize=10, fontweight="bold",
            color=COLORS["dark"], family="monospace")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "LEMURS_Fig2_Path_Diagram.png", facecolor="white")
    plt.close()
    print("  Saved -> LEMURS_Fig2_Path_Diagram.png")


def fig_bootstrap(R):
    print("Generating Figure: Bootstrap Distribution ...")
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.hist(R["boot"], bins=60, color=COLORS["sleep"], alpha=0.75,
            edgecolor="black", lw=0.4, density=True)
    ax.axvline(R["ab"], color="red", lw=2.5, label=f'Point est = {R["ab"]:.4f}')
    ax.axvline(R["ci_lo"], color="orange", ls="--", lw=2, label=f'2.5th = {R["ci_lo"]:.4f}')
    ax.axvline(R["ci_hi"], color="orange", ls="--", lw=2, label=f'97.5th = {R["ci_hi"]:.4f}')
    ax.axvline(0, color="black", ls=":", lw=1.5, label="Null (zero)")
    ax.set_xlabel("Indirect Effect (a x b)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title(f"LEMURS Validation - Bootstrap Distribution of Indirect Effect\n"
                 f"(B = {N_BOOTSTRAP:,} resamples)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "LEMURS_Fig3_Bootstrap.png", facecolor="white")
    plt.close()
    print("  Saved -> LEMURS_Fig3_Bootstrap.png")


def fig_diagnostics(R):
    print("Generating Figure: Residual Diagnostics ...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor("white")
    m3 = R["model3"]
    fitted, resid = m3.fittedvalues, m3.resid

    ax = axes[0]
    ax.scatter(fitted, resid, alpha=0.25, s=10, color=COLORS["sleep"])
    ax.axhline(0, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Fitted Values", fontweight="bold")
    ax.set_ylabel("Residuals", fontweight="bold")
    ax.set_title("A. Residuals vs. Fitted", fontweight="bold")
    ax.grid(alpha=0.3)

    ax = axes[1]
    sm.qqplot(resid, line="45", fit=True, ax=ax, markerfacecolor=COLORS["caffeine"],
              markeredgecolor="black", alpha=0.4, markersize=3)
    ax.set_title("B. Normal Q-Q Plot", fontweight="bold")
    ax.grid(alpha=0.3)

    fig.suptitle("LEMURS Validation - Regression Diagnostics (Mediated Model)",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "LEMURS_Fig4_Diagnostics.png", facecolor="white")
    plt.close()
    print("  Saved -> LEMURS_Fig4_Diagnostics.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("LEMURS EXTERNAL VALIDATION")
    print("Caffeine Exposure -> PSS-10 Stress -> Sleep Duration")
    print("=" * 70 + "\n")

    df = load_and_build()
    desc = compute_descriptives(df)
    R = run_mediation(df, cov_col="activity_days", label="PRIMARY (active days/week)")

    # Sensitivity: duration-based activity proxy (ambiguous category, see docstring)
    sens_key_vars = ["sleep_hours", "pss_score", "caffeine_exposure_score",
                      "activity_hours_sensitivity"]
    df_sens = df.dropna(subset=["activity_hours_sensitivity"]).copy()
    R_sens = run_mediation(df_sens, cov_col="activity_hours_sensitivity",
                           label="SENSITIVITY (duration-based, ambiguous category)")

    fig_distributions(df)
    fig_path_diagram(R)
    fig_bootstrap(R)
    fig_diagnostics(R)

    print("\n" + "=" * 70)
    print("LEMURS VALIDATION COMPLETE")
    print(f"  Primary sample n = {len(df):,}   Sensitivity sample n = {len(df_sens):,}")
    print(f"  Primary result: {R['med_type']}  ({R['prop']:.1f}% mediated)")
    print("=" * 70 + "\n")

    return df, desc, R, R_sens


if __name__ == "__main__":
    df, desc, R, R_sens = main()
