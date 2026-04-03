"""
================================================================================
Baron-Kenny Mediation Analysis
Caffeine Intake -> Perceived Stress -> Sleep Duration

Research Question
-----------------
Does perceived stress mediate the relationship between daily caffeine
intake and sleep duration?

Design       : Cross-sectional, observational (synthetic data)
Dataset      : Global Coffee Health - Synthetic (Kaggle, n = 10 000)
Framework    : Baron & Kenny (1986) four-step approach
CI method    : Percentile bootstrap (5 000 resamples)
Outcome      : Sleep Duration (hours)  [single DV]

Authors      : Sena Cetin & Elif Beyza Oztoprak
Affiliation  : Turkish-German University
================================================================================
"""

# ---- IMPORTS ----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ---- CONFIGURATION ----------------------------------------------------------
np.random.seed(42)
N_BOOTSTRAP = 5000
ALPHA = 0.05

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "Reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Plot styling
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


# =============================================================================
# 1.  DATA PIPELINE
# =============================================================================

def load_and_clean_data():
    """
    Load the Global Coffee Health dataset and prepare an analytic sample.

    KEY FIX (vs. previous version)
    ------------------------------
    The earlier pipeline mapped Sleep_Quality -> numeric (Poor/Fair/Good)
    but the raw data also contains the 'Excellent' category, which had no
    mapping, producing NaN.  Because Sleep_Quality_Score was among the
    mandatory key_vars, those ~1 500 rows were silently dropped.

    Since our sole DV is Sleep Duration (hours), we no longer create or
    require Sleep_Quality_Score.  This retains the full sample after the
    standard age and outlier filters.
    """
    print("=" * 70)
    print("STEP 1: DATA LOADING AND CLEANING")
    print("=" * 70)

    raw = pd.read_csv(DATA_DIR / "synthetic_coffee_health_10000.csv")
    print(f"Raw data loaded: {len(raw):,} records, {len(raw.columns)} variables")

    df = raw.copy()

    # Age filter (adults 18-65)
    df = df[(df["Age"] >= 18) & (df["Age"] <= 65)]
    print(f"After age filter (18-65): {len(df):,} records")

    # Encode categoricals
    stress_map = {"Low": 2, "Medium": 5, "High": 8}
    df["Stress_Score"] = df["Stress_Level"].map(stress_map)
    df["Gender_Num"] = (df["Gender"] == "Male").astype(int)

    # Drop rows missing on analysis variables
    # NOTE: Sleep_Quality_Score NOT included -> no data loss from "Excellent"
    key_vars = [
        "Caffeine_mg", "Stress_Score", "Sleep_Hours",
        "Age", "Gender_Num", "BMI",
        "Physical_Activity_Hours", "Heart_Rate",
    ]
    n_before = len(df)
    df = df.dropna(subset=key_vars)
    print(f"After missing-value removal: {len(df):,} (dropped {n_before - len(df)})")

    # Outlier removal (IQR x 1.5)
    continuous = ["Caffeine_mg", "Sleep_Hours", "BMI",
                  "Heart_Rate", "Physical_Activity_Hours"]
    df = _remove_outliers_iqr(df, continuous)

    print(f"\nFinal analytic sample: n = {len(df):,}")
    print(f"Total records removed: {len(raw) - len(df):,} "
          f"({(len(raw) - len(df)) / len(raw) * 100:.1f}%)")
    return df


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
            print(f"  {col}: {rm} outliers removed (range: {lo:.1f} - {hi:.1f})")
    return out


# =============================================================================
# 2.  DESCRIPTIVE STATISTICS
# =============================================================================

def compute_descriptives(df):
    print("\n" + "=" * 70)
    print("STEP 2: DESCRIPTIVE STATISTICS")
    print("=" * 70)

    info = {
        "Caffeine_mg":             "Daily Caffeine Intake (mg)",
        "Stress_Score":            "Stress Score (2-8)",
        "Sleep_Hours":             "Sleep Duration (hours)",
        "Age":                     "Age (years)",
        "BMI":                     "BMI (kg/m2)",
        "Physical_Activity_Hours": "Physical Activity (h/week)",
        "Heart_Rate":              "Resting Heart Rate (bpm)",
    }
    rows = []
    for var, label in info.items():
        s = df[var]
        rows.append({
            "Variable": label, "n": int(s.count()),
            "Mean": round(s.mean(), 2), "SD": round(s.std(), 2),
            "Median": round(s.median(), 2),
            "Min": round(s.min(), 2), "Max": round(s.max(), 2),
            "Skewness": round(s.skew(), 3), "Kurtosis": round(s.kurtosis(), 3),
        })

    desc = pd.DataFrame(rows)
    print("\nDescriptive Statistics:")
    print("-" * 70)
    for _, r in desc.iterrows():
        print(f"  {r['Variable']:35s}  M={r['Mean']:8.2f}  SD={r['SD']:7.2f}  "
              f"[{r['Min']}, {r['Max']}]")

    gc = df["Gender"].value_counts()
    print(f"\nGender: Male {gc.get('Male',0)} ({gc.get('Male',0)/len(df)*100:.1f}%), "
          f"Female {gc.get('Female',0)} ({gc.get('Female',0)/len(df)*100:.1f}%)")

    sc = df["Stress_Level"].value_counts()
    for lev in ["Low", "Medium", "High"]:
        c = sc.get(lev, 0)
        print(f"  Stress {lev}: {c} ({c / len(df) * 100:.1f}%)")

    desc.to_csv(REPORTS_DIR / "descriptive_statistics.csv", index=False)
    print(f"\nSaved -> descriptive_statistics.csv")
    return desc


# =============================================================================
# 3.  CORRELATION ANALYSIS
# =============================================================================

def compute_correlations(df):
    print("\n" + "=" * 70)
    print("STEP 3: CORRELATION ANALYSIS")
    print("=" * 70)

    vars_ = ["Caffeine_mg", "Stress_Score", "Sleep_Hours",
             "Age", "BMI", "Physical_Activity_Hours", "Heart_Rate"]
    corr = df[vars_].corr(method="pearson")

    pairs = [
        ("Caffeine_mg", "Stress_Score", "Caffeine <-> Stress"),
        ("Caffeine_mg", "Sleep_Hours",  "Caffeine <-> Sleep Duration"),
        ("Stress_Score", "Sleep_Hours", "Stress   <-> Sleep Duration"),
    ]
    print("\nKey Bivariate Correlations (Pearson r):")
    for v1, v2, label in pairs:
        r, p = stats.pearsonr(df[v1], df[v2])
        print(f"  {label:35s}  r = {r:+.4f}  p = {p:.2e} {_sig(p)}")

    corr.to_csv(REPORTS_DIR / "correlation_matrix.csv")
    print(f"\nSaved -> correlation_matrix.csv")
    return corr


# =============================================================================
# 4.  REGRESSION ASSUMPTION DIAGNOSTICS
# =============================================================================

def check_assumptions(df):
    print("\n" + "=" * 70)
    print("STEP 4: REGRESSION DIAGNOSTICS")
    print("=" * 70)

    predictors = ["Caffeine_mg", "Stress_Score", "Age",
                  "Gender_Num", "BMI", "Physical_Activity_Hours", "Heart_Rate"]
    X = sm.add_constant(df[predictors])

    print("\nVariance Inflation Factors:")
    vif_rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        v = variance_inflation_factor(X.values, i)
        vif_rows.append({"Variable": col, "VIF": round(v, 2)})
        print(f"  {col:30s}  VIF = {v:.2f}")
    vif_df = pd.DataFrame(vif_rows)
    mx = vif_df["VIF"].max()
    print(f"\n  Max VIF = {mx:.2f} {'(< 10: OK)' if mx < 10 else '(WARNING)'}")

    Y = df["Sleep_Hours"].values
    X_ols = sm.add_constant(np.column_stack([
        df["Caffeine_mg"].values, df[predictors[2:]].values]))
    resid = sm.OLS(Y, X_ols).fit().resid
    dw = durbin_watson(resid)
    print(f"\n  Durbin-Watson (total-effect model): {dw:.3f}")

    vif_df.to_csv(REPORTS_DIR / "assumption_diagnostics.csv", index=False)
    print(f"\nSaved -> assumption_diagnostics.csv")
    return vif_df


# =============================================================================
# 5.  BARON-KENNY MEDIATION ANALYSIS
# =============================================================================

def run_mediation(df):
    """
    Baron-Kenny (1986) mediation - single outcome: Sleep Duration.

    X = Caffeine_mg   (independent)
    M = Stress_Score   (mediator)
    Y = Sleep_Hours    (dependent)
    C = covariates

    Steps:
      1. Total effect:  Y = b0 + c*X  + g'C + e
      2. Path a:        M = a0 + a*X  + d'C + v
      3-4. b + direct:  Y = t0 + c'*X + b*M + f'C + h

    Indirect effect = a * b
    95% CI = percentile bootstrap (N_BOOTSTRAP resamples)
    """
    print("\n" + "=" * 70)
    print("STEP 5: BARON-KENNY MEDIATION ANALYSIS - Sleep Duration")
    print("=" * 70)

    covs = ["Age", "Gender_Num", "BMI", "Physical_Activity_Hours", "Heart_Rate"]
    X = df["Caffeine_mg"].values
    M = df["Stress_Score"].values
    Y = df["Sleep_Hours"].values
    C = df[covs].values
    n = len(df)

    # Step 1: Total effect
    X1 = sm.add_constant(np.column_stack([X, C]))
    m1 = sm.OLS(Y, X1).fit()
    c, c_se, c_t, c_p = m1.params[1], m1.bse[1], m1.tvalues[1], m1.pvalues[1]
    r2_1, r2_1a = m1.rsquared, m1.rsquared_adj
    f1, f1_p = m1.fvalue, m1.f_pvalue

    print(f"\n  Step 1 - Total Effect  Y = c*X + C")
    print(f"    c  = {c:.6f}   SE = {c_se:.6f}   t = {c_t:.3f}   p = {c_p:.2e} {_sig(c_p)}")
    print(f"    R2 = {r2_1:.4f}   Adj R2 = {r2_1a:.4f}   F = {f1:.2f}   p(F) = {f1_p:.2e}")

    # Step 2: Path a (X -> M)
    X2 = sm.add_constant(np.column_stack([X, C]))
    m2 = sm.OLS(M, X2).fit()
    a, a_se, a_t, a_p = m2.params[1], m2.bse[1], m2.tvalues[1], m2.pvalues[1]
    r2_2 = m2.rsquared

    print(f"\n  Step 2 - Path a  M = a*X + C")
    print(f"    a  = {a:.6f}   SE = {a_se:.6f}   t = {a_t:.3f}   p = {a_p:.2e} {_sig(a_p)}")
    print(f"    R2 = {r2_2:.4f}")

    # Step 3-4: Path b + Direct effect
    X3 = sm.add_constant(np.column_stack([X, M, C]))
    m3 = sm.OLS(Y, X3).fit()
    cp, cp_se, cp_t, cp_p = m3.params[1], m3.bse[1], m3.tvalues[1], m3.pvalues[1]
    b, b_se, b_t, b_p = m3.params[2], m3.bse[2], m3.tvalues[2], m3.pvalues[2]
    r2_3, r2_3a = m3.rsquared, m3.rsquared_adj
    f3, f3_p = m3.fvalue, m3.f_pvalue

    print(f"\n  Step 3-4 - Direct effect + Path b   Y = c'*X + b*M + C")
    print(f"    c' = {cp:.6f}   SE = {cp_se:.6f}   t = {cp_t:.3f}   p = {cp_p:.2e} {_sig(cp_p)}")
    print(f"    b  = {b:.6f}    SE = {b_se:.6f}   t = {b_t:.3f}   p = {b_p:.2e} {_sig(b_p)}")
    print(f"    R2 = {r2_3:.4f}   Adj R2 = {r2_3a:.4f}   F = {f3:.2f}   p(F) = {f3_p:.2e}")

    # Indirect effect & Sobel
    ab = a * b
    sob_se = np.sqrt(a**2 * b_se**2 + b**2 * a_se**2)
    sob_z = ab / sob_se if sob_se > 0 else 0
    sob_p = 2 * (1 - stats.norm.cdf(abs(sob_z)))

    # Percentile bootstrap CI
    print(f"\n  Bootstrap ({N_BOOTSTRAP:,} resamples) ...")
    boot = np.empty(N_BOOTSTRAP)
    ok = 0
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(n, n, replace=True)
        Xb, Mb, Yb, Cb = X[idx], M[idx], Y[idx], C[idx]
        try:
            a_b = sm.OLS(Mb, sm.add_constant(np.column_stack([Xb, Cb]))).fit().params[1]
            b_b = sm.OLS(Yb, sm.add_constant(np.column_stack([Xb, Mb, Cb]))).fit().params[2]
            boot[ok] = a_b * b_b
            ok += 1
        except Exception:
            continue
    boot = boot[:ok]
    ci_lo = np.percentile(boot, 2.5)
    ci_hi = np.percentile(boot, 97.5)
    ci_sig = not (ci_lo <= 0 <= ci_hi)

    # Baron-Kenny conditions
    cond = [c_p < ALPHA, a_p < ALPHA, b_p < ALPHA, abs(cp) < abs(c)]
    all_ok = all(cond)
    if all_ok and ci_sig:
        med_type = "FULL MEDIATION" if cp_p >= ALPHA else "PARTIAL MEDIATION"
    else:
        med_type = "NO MEDIATION"
    prop = (ab / c * 100) if c != 0 else 0

    # Standardised coefficients
    sx, sm_, sy = X.std(), M.std(), Y.std()
    a_std = a * sx / sm_
    b_std = b * sm_ / sy
    c_std = c * sx / sy
    cp_std = cp * sx / sy
    ab_std = a_std * b_std
    f2 = (r2_3 - r2_1) / (1 - r2_3) if r2_3 < 1 else np.nan

    # Print summary
    print(f"\n  {'=' * 60}")
    print(f"  INDIRECT EFFECT  (a x b) = {ab:.6f}")
    print(f"  95% Percentile Bootstrap CI: [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"  CI excludes zero: {'Yes' if ci_sig else 'No'}")
    print(f"  Sobel Z = {sob_z:.3f}  p = {sob_p:.2e}")
    print(f"  Proportion mediated = {prop:.1f}%")
    print(f"  {'=' * 60}")
    print(f"  Standardised:  a={a_std:.4f}  b={b_std:.4f}  "
          f"c={c_std:.4f}  c'={cp_std:.4f}  ab={ab_std:.4f}")
    print(f"  Cohen's f2 (mediator addition) = {f2:.4f}")
    print(f"  {'=' * 60}")
    print(f"  Baron-Kenny Conditions:")
    labels = ["c sig (X->Y)", "a sig (X->M)", "b sig (M->Y|X)", "|c'|<|c|"]
    pvals = [c_p, a_p, b_p, None]
    for i, (lb, cd) in enumerate(zip(labels, cond)):
        pstr = f"  p = {pvals[i]:.2e}" if pvals[i] is not None else ""
        print(f"    {i+1}. {lb:25s} {'PASS' if cd else 'FAIL'}{pstr}")
    print(f"\n  => CONCLUSION: {med_type}")
    print("=" * 70)

    R = dict(
        n=n,
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
        cond=cond, all_ok=all_ok, med_type=med_type,
        boot=boot, model1=m1, model2=m2, model3=m3,
    )

    # Save CSV
    skip = {"boot", "model1", "model2", "model3", "cond"}
    row = {k: v for k, v in R.items() if k not in skip}
    row["cond1"], row["cond2"], row["cond3"], row["cond4"] = cond
    pd.DataFrame([row]).to_csv(REPORTS_DIR / "mediation_results.csv", index=False)
    print(f"\nSaved -> mediation_results.csv")
    return R


# =============================================================================
# 6.  PUBLICATION FIGURES
# =============================================================================

def fig1_descriptive(df):
    """Figure 1 - Sample characteristics and variable distributions."""
    print("\nGenerating Figure 1: Descriptive Statistics ...")
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 3, hspace=0.40, wspace=0.30)
    fig.suptitle("Figure 1. Sample Characteristics and Variable Distributions",
                 fontsize=15, fontweight="bold", y=0.99)

    # A - Caffeine
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(df["Caffeine_mg"], bins=40, color=COLORS["caffeine"],
            alpha=0.75, edgecolor="black", linewidth=0.6)
    ax.axvline(df["Caffeine_mg"].mean(), color="red", ls="--", lw=2,
               label=f'Mean = {df["Caffeine_mg"].mean():.1f} mg')
    ax.set_xlabel("Daily Caffeine Intake (mg)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("A. Caffeine Intake", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # B - Stress
    ax = fig.add_subplot(gs[0, 1])
    sc = df["Stress_Level"].value_counts().reindex(["Low", "Medium", "High"])
    cols = [COLORS["positive"], COLORS["neutral"], COLORS["negative"]]
    bars = ax.bar(sc.index, sc.values, color=cols, edgecolor="black", lw=1.3, width=0.55)
    for bar, cnt in zip(bars, sc.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{cnt}\n({cnt/len(df)*100:.1f}%)", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Stress Level", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("B. Stress Level", fontweight="bold")
    ax.set_ylim(0, sc.max() * 1.18)
    ax.grid(axis="y", alpha=0.3)

    # C - Sleep Duration
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(df["Sleep_Hours"], bins=40, color=COLORS["sleep"],
            alpha=0.75, edgecolor="black", linewidth=0.6)
    ax.axvline(df["Sleep_Hours"].mean(), color="red", ls="--", lw=2,
               label=f'Mean = {df["Sleep_Hours"].mean():.2f} h')
    ax.set_xlabel("Sleep Duration (hours)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("C. Sleep Duration (Outcome)", fontweight="bold", color=COLORS["sleep"])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # D - Age
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(df["Age"], bins=30, color="#9B59B6", alpha=0.7, edgecolor="black", lw=0.6)
    ax.axvline(df["Age"].mean(), color="red", ls="--", lw=2,
               label=f'Mean = {df["Age"].mean():.1f}')
    ax.set_xlabel("Age (years)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("D. Age", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # E - BMI
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(df["BMI"], bins=30, color="#3498DB", alpha=0.7, edgecolor="black", lw=0.6)
    ax.axvline(df["BMI"].mean(), color="red", ls="--", lw=2,
               label=f'Mean = {df["BMI"].mean():.1f}')
    ax.set_xlabel("BMI (kg/m2)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("E. BMI", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # F - Heart Rate
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(df["Heart_Rate"], bins=30, color="#E74C3C", alpha=0.7, edgecolor="black", lw=0.6)
    ax.axvline(df["Heart_Rate"].mean(), color="blue", ls="--", lw=2,
               label=f'Mean = {df["Heart_Rate"].mean():.0f} bpm')
    ax.set_xlabel("Resting Heart Rate (bpm)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("F. Heart Rate", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # G - Gender
    ax = fig.add_subplot(gs[2, 0])
    gc = df["Gender"].value_counts()
    bars = ax.bar(gc.index, gc.values, color=["#3498DB", "#E74C3C"],
                  edgecolor="black", lw=1.3, width=0.55)
    for bar, cnt in zip(bars, gc.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{cnt}\n({cnt/len(df)*100:.1f}%)", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Gender", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("G. Gender", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # H - Physical Activity
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(df["Physical_Activity_Hours"], bins=30, color="#27AE60", alpha=0.7,
            edgecolor="black", lw=0.6)
    ax.axvline(df["Physical_Activity_Hours"].mean(), color="red", ls="--", lw=2,
               label=f'Mean = {df["Physical_Activity_Hours"].mean():.1f} h/wk')
    ax.set_xlabel("Physical Activity (h/week)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("H. Physical Activity", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # I - Summary table
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    tbl = [
        ["Total Sample",         f'n = {len(df):,}'],
        ["Age Range",            f'{df["Age"].min():.0f} - {df["Age"].max():.0f} years'],
        ["Male %",               f'{df["Gender_Num"].mean()*100:.1f}%'],
        ["Caffeine (M +/- SD)",  f'{df["Caffeine_mg"].mean():.1f} +/- {df["Caffeine_mg"].std():.1f} mg'],
        ["Sleep Hours (M +/- SD)", f'{df["Sleep_Hours"].mean():.2f} +/- {df["Sleep_Hours"].std():.2f} h'],
        ["Stress (L / M / H)",   f'{(df["Stress_Level"]=="Low").sum()} / '
                                  f'{(df["Stress_Level"]=="Medium").sum()} / '
                                  f'{(df["Stress_Level"]=="High").sum()}'],
        ["BMI (M +/- SD)",       f'{df["BMI"].mean():.1f} +/- {df["BMI"].std():.1f}'],
        ["Heart Rate (M +/- SD)", f'{df["Heart_Rate"].mean():.0f} +/- {df["Heart_Rate"].std():.0f} bpm'],
    ]
    table = ax.table(cellText=tbl, colLabels=["Characteristic", "Value"],
                     loc="center", cellLoc="left", colWidths=[0.55, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    for j in range(2):
        table[(0, j)].set_facecolor(COLORS["dark"])
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(tbl) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor(COLORS["bg"])
    ax.set_title("I. Summary Statistics", fontweight="bold", pad=10)

    plt.savefig(FIGURES_DIR / "Fig1_Descriptive_Statistics.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved -> Fig1_Descriptive_Statistics.png")


def fig2_correlations(df, corr):
    """Figure 2 - Correlation heatmap and bivariate scatter plots."""
    print("Generating Figure 2: Correlation Analysis ...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle("Figure 2. Bivariate Correlation Analysis",
                 fontsize=15, fontweight="bold", y=1.00)

    # A - Heatmap
    ax = axes[0, 0]
    labels = ["Caffeine", "Stress", "Sleep\nHours", "Age", "BMI",
              "Phys Act", "HR"]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={"size": 8, "weight": "bold"},
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="gray",
                cbar_kws={"label": "Pearson r"})
    ax.set_title("A. Correlation Heatmap", fontweight="bold")

    def _scatter(ax, xvar, yvar, xlabel, ylabel, title, color):
        ax.scatter(df[xvar], df[yvar], alpha=0.35, s=12, color=color)
        z = np.polyfit(df[xvar], df[yvar], 1)
        xl = np.linspace(df[xvar].min(), df[xvar].max(), 100)
        ax.plot(xl, np.polyval(z, xl), "r-", lw=2.5)
        r, p = stats.pearsonr(df[xvar], df[yvar])
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.set_title(f"{title}\nr = {r:.3f}, {p_str}{_sig(p)}", fontweight="bold")
        ax.grid(alpha=0.3)

    _scatter(axes[0, 1], "Caffeine_mg", "Stress_Score",
             "Caffeine (mg)", "Stress Score",
             "B. Caffeine vs Stress (Path a)", COLORS["caffeine"])
    _scatter(axes[0, 2], "Caffeine_mg", "Sleep_Hours",
             "Caffeine (mg)", "Sleep Duration (h)",
             "C. Caffeine vs Sleep (Total Effect)", COLORS["caffeine"])
    _scatter(axes[1, 0], "Stress_Score", "Sleep_Hours",
             "Stress Score", "Sleep Duration (h)",
             "D. Stress vs Sleep (Path b)", COLORS["stress"])

    # E - Boxplot by stress level
    ax = axes[1, 1]
    order = ["Low", "Medium", "High"]
    sns.boxplot(x="Stress_Level", y="Sleep_Hours", data=df, ax=ax,
                order=order, palette=[COLORS["positive"], COLORS["neutral"], COLORS["negative"]],
                width=0.5)
    ax.set_xlabel("Stress Level", fontweight="bold")
    ax.set_ylabel("Sleep Duration (h)", fontweight="bold")
    ax.set_title("E. Sleep by Stress Group", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # F - Boxplot by caffeine quartile
    ax = axes[1, 2]
    df_tmp = df.copy()
    df_tmp["CafQ"] = pd.qcut(df_tmp["Caffeine_mg"], 4,
                              labels=["Q1\n(Low)", "Q2", "Q3", "Q4\n(High)"])
    sns.boxplot(x="CafQ", y="Sleep_Hours", data=df_tmp, ax=ax,
                palette="YlOrBr", width=0.5)
    ax.set_xlabel("Caffeine Quartile", fontweight="bold")
    ax.set_ylabel("Sleep Duration (h)", fontweight="bold")
    ax.set_title("F. Sleep by Caffeine Quartile", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Fig2_Correlation_Analysis.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved -> Fig2_Correlation_Analysis.png")


def fig3_mediation_diagram(R):
    """Figure 3 - Baron-Kenny mediation path diagram."""
    print("Generating Figure 3: Mediation Path Diagram ...")
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.suptitle("Figure 3. Baron-Kenny Mediation Model\n"
                 "Caffeine Intake -> Perceived Stress -> Sleep Duration",
                 fontsize=15, fontweight="bold", y=0.98)

    # Boxes
    for xy, label, sub, color in [
        ((0.3, 4.0), "CAFFEINE\nINTAKE (mg)", "(Independent Variable)", COLORS["caffeine"]),
        ((3.9, 6.8), "PERCEIVED\nSTRESS",      "(Mediator)",            COLORS["stress"]),
        ((7.5, 4.0), "SLEEP\nDURATION (h)",    "(Dependent Variable)",  COLORS["sleep"]),
    ]:
        box = FancyBboxPatch(xy, 2.2, 1.8, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="black", lw=2.5)
        ax.add_patch(box)
        cx, cy = xy[0] + 1.1, xy[1] + 1.05
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(cx, cy - 0.45, sub, ha="center", va="center",
                fontsize=8, color="white", style="italic")

    # path a
    arr_a = FancyArrowPatch((2.5, 5.3), (3.9, 7.3), arrowstyle="->",
                            mutation_scale=30, lw=3, color="#16A085", zorder=5)
    ax.add_patch(arr_a)
    a_pstr = "p < 0.001" if R['a_p'] < 0.001 else f"p = {R['a_p']:.4f}"
    ax.text(2.7, 6.6,
            f"a = {R['a']:.6f}\nSE = {R['a_se']:.6f}\n{a_pstr}{_sig(R['a_p'])}",
            ha="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#ECFDF5", ec="#16A085", lw=1.5))

    # path b
    arr_b = FancyArrowPatch((6.1, 7.3), (7.5, 5.3), arrowstyle="->",
                            mutation_scale=30, lw=3, color="#C0392B", zorder=5)
    ax.add_patch(arr_b)
    b_pstr = "p < 0.001" if R['b_p'] < 0.001 else f"p = {R['b_p']:.4f}"
    ax.text(7.3, 6.6,
            f"b = {R['b']:.6f}\nSE = {R['b_se']:.6f}\n{b_pstr}{_sig(R['b_p'])}",
            ha="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#FADBD8", ec="#C0392B", lw=1.5))

    # total c (dashed)
    arr_c = FancyArrowPatch((2.5, 5.5), (7.5, 5.5), arrowstyle="->",
                            mutation_scale=25, lw=2, color="gray", ls="--", alpha=0.7)
    ax.add_patch(arr_c)
    ax.text(5.0, 5.85, f"c = {R['c']:.6f} (Total Effect)",
            ha="center", fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray"))

    # direct c'
    arr_cp = FancyArrowPatch((2.5, 4.5), (7.5, 4.5), arrowstyle="->",
                             mutation_scale=30, lw=3, color="#8E44AD", zorder=4)
    ax.add_patch(arr_cp)
    ax.text(5.0, 3.85,
            f"c' = {R['cp']:.6f} (Direct Effect)",
            ha="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#F4ECF7", ec="#8E44AD", lw=2))

    # Results box
    rbox = FancyBboxPatch((0.3, 0.5), 9.4, 1.6, boxstyle="round,pad=0.15",
                          fc=COLORS["bg"], ec=COLORS["dark"], lw=2)
    ax.add_patch(rbox)
    rtxt = (f"Indirect Effect (a x b) = {R['ab']:.6f}   |   "
            f"95% Percentile Bootstrap CI = [{R['ci_lo']:.6f}, {R['ci_hi']:.6f}]   |   "
            f"Significant: {'YES' if R['ci_sig'] else 'NO'}\n"
            f"Proportion Mediated = {R['prop']:.1f}%   |   "
            f"R2 (with mediator) = {R['r2_med']:.4f}   |   "
            f"Conclusion: {R['med_type']}")
    ax.text(5.0, 1.3, rtxt, ha="center", va="center", fontsize=10,
            fontweight="bold", color=COLORS["dark"], family="monospace")

    # Conditions bar
    cond_labels = ["Step 1 c sig", "Step 2 a sig", "Step 3 b sig", "Step 4 |c'|<|c|"]
    cond_marks = ["+" if c else "-" for c in R["cond"]]
    cond_txt = "Baron-Kenny:  " + "   |   ".join(
        f"{lb} {mk}" for lb, mk in zip(cond_labels, cond_marks))
    ax.text(5.0, 0.15, cond_txt, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", fc="#D5F4E6", ec=COLORS["positive"], lw=1.5),
            family="monospace")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Fig3_Mediation_Model.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved -> Fig3_Mediation_Model.png")


def fig4_baron_kenny_steps(R):
    """Figure 4 - Four-step validation detail."""
    print("Generating Figure 4: Baron-Kenny Validation Steps ...")
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.suptitle("Figure 4. Baron-Kenny Four-Step Validation - Sleep Duration",
                 fontsize=15, fontweight="bold", y=0.98)

    steps = [
        ("Step 1: Total Effect (c path)",
         "Y = c * X + Covariates",
         f"Sleep Hours = {R['c']:.6f} * Caffeine + Covariates",
         f"c = {R['c']:.6f}  (SE = {R['c_se']:.6f})  p = {R['c_p']:.2e}",
         R["cond"][0], 8.6),
        ("Step 2: Path a (X -> M)",
         "M = a * X + Covariates",
         f"Stress = {R['a']:.6f} * Caffeine + Covariates",
         f"a = {R['a']:.6f}  (SE = {R['a_se']:.6f})  p = {R['a_p']:.2e}",
         R["cond"][1], 6.5),
        ("Step 3: Path b (M -> Y | X)",
         "Y = b * M + c' * X + Covariates",
         f"Sleep = {R['b']:.6f} * Stress + {R['cp']:.6f} * Caffeine + Cov",
         f"b = {R['b']:.6f}  (SE = {R['b_se']:.6f})  p = {R['b_p']:.2e}",
         R["cond"][2], 4.4),
        ("Step 4: Effect Reduction (|c'| < |c|)",
         "c = c' + a * b",
         f"|c'| = {abs(R['cp']):.6f}  <  |c| = {abs(R['c']):.6f}",
         f"Reduction: {abs(R['c']) - abs(R['cp']):.6f}  ({R['prop']:.1f}% mediated)",
         R["cond"][3], 2.3),
    ]

    for i, (title, model, eq, result, passed, y) in enumerate(steps):
        fc = "#D5F4E6" if passed else "#FADBD8"
        ec = COLORS["positive"] if passed else COLORS["negative"]
        box = FancyBboxPatch((0.3, y - 0.7), 9.4, 1.6, boxstyle="round,pad=0.1",
                             facecolor=fc, edgecolor=ec, lw=2.5)
        ax.add_patch(box)

        ax.text(0.85, y + 0.45, str(i + 1), fontsize=12, fontweight="bold",
                ha="center", va="center", color="white",
                bbox=dict(boxstyle="circle,pad=0.3", fc=ec, ec="black", lw=1.5))
        ax.text(2.0, y + 0.45, title, fontsize=11, fontweight="bold", ha="left")
        ax.text(8.5, y + 0.45, "PASS" if passed else "FAIL",
                fontsize=11, fontweight="bold", color=ec, ha="center")
        ax.text(0.55, y + 0.0, f"Model: {model}", fontsize=9, ha="left", style="italic")
        ax.text(0.55, y - 0.35, eq, fontsize=9, ha="left", family="monospace", fontweight="bold")
        ax.text(5.5, y - 0.35, result, fontsize=9, ha="left", family="monospace",
                color="#C0392B", fontweight="bold")

    stxt = (f"All four conditions met -> {R['med_type']}\n"
            f"Indirect effect a x b = {R['ab']:.6f},  "
            f"95% CI [{R['ci_lo']:.6f}, {R['ci_hi']:.6f}]")
    ax.text(5.0, 0.4, stxt, ha="center", fontsize=10, fontweight="bold", style="italic",
            bbox=dict(boxstyle="round,pad=0.7", fc="#FEF9E7", ec="#F39C12", lw=2))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Fig4_Baron_Kenny_Steps.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved -> Fig4_Baron_Kenny_Steps.png")


def fig5_bootstrap(R):
    """Figure 5 - Bootstrap distribution and effect decomposition."""
    print("Generating Figure 5: Bootstrap Analysis ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle("Figure 5. Indirect Effect - Percentile Bootstrap Analysis\n"
                 f"(B = {N_BOOTSTRAP:,} resamples)",
                 fontsize=15, fontweight="bold", y=1.00)

    # A - histogram
    ax1.hist(R["boot"], bins=60, color=COLORS["sleep"], alpha=0.7,
             edgecolor="black", lw=0.5, density=True)
    ax1.axvline(R["ab"], color="red", lw=2.5, label=f'Point est = {R["ab"]:.6f}')
    ax1.axvline(R["ci_lo"], color="orange", ls="--", lw=2, label=f'2.5th = {R["ci_lo"]:.6f}')
    ax1.axvline(R["ci_hi"], color="orange", ls="--", lw=2, label=f'97.5th = {R["ci_hi"]:.6f}')
    ax1.axvline(0, color="black", ls=":", lw=1.5, label="Null (zero)")
    ax1.set_xlabel("Indirect Effect (a x b)", fontweight="bold")
    ax1.set_ylabel("Density", fontweight="bold")
    ax1.set_title("A. Bootstrap Distribution of Indirect Effect", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # B - effect decomposition
    effects = [R["c"], R["cp"], R["ab"]]
    names = ["Total Effect\n(c)", "Direct Effect\n(c')", "Indirect Effect\n(a x b)"]
    colors = [COLORS["neutral"], COLORS["sleep"], COLORS["stress"]]
    bars = ax2.bar(range(3), effects, color=colors, edgecolor="black", lw=1.5,
                   width=0.55, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    for bar, eff in zip(bars, effects):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2,
                 h + (0.00002 if h >= 0 else -0.00008),
                 f"{eff:.6f}", ha="center",
                 va="bottom" if h >= 0 else "top",
                 fontsize=10, fontweight="bold", family="monospace")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(names, fontweight="bold")
    ax2.set_ylabel("Effect Size (unstandardised)", fontweight="bold")
    ax2.set_title("B. Effect Decomposition\nc = c' + (a x b)", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    eq = (f"c = c' + (a x b)\n"
          f"{R['c']:.6f} = {R['cp']:.6f} + ({R['ab']:.6f})")
    ax2.text(0.5, -0.20, eq, transform=ax2.transAxes, ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.6", fc=COLORS["bg"], ec=COLORS["dark"], lw=1.5),
             family="monospace", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Fig5_Bootstrap_Analysis.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved -> Fig5_Bootstrap_Analysis.png")


def fig6_summary(R):
    """Figure 6 - Comprehensive results summary."""
    print("Generating Figure 6: Summary ...")
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)
    fig.suptitle("Figure 6. Comprehensive Mediation Analysis Summary",
                 fontsize=15, fontweight="bold", y=0.99)

    # A - Coefficients table
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    header = ["Path", "B (unstd)", "SE", "t", "p", "Beta (std)", "Sig"]
    data = [
        ["Total Effect (c): X -> Y",
         f'{R["c"]:.6f}', f'{R["c_se"]:.6f}', f'{R["c_t"]:.3f}',
         f'{R["c_p"]:.2e}', f'{R["c_std"]:.4f}', "Yes" if R["cond"][0] else "No"],
        ["Path a: X -> M",
         f'{R["a"]:.6f}', f'{R["a_se"]:.6f}', f'{R["a_t"]:.3f}',
         f'{R["a_p"]:.2e}', f'{R["a_std"]:.4f}', "Yes" if R["cond"][1] else "No"],
        ["Path b: M -> Y | X",
         f'{R["b"]:.6f}', f'{R["b_se"]:.6f}', f'{R["b_t"]:.3f}',
         f'{R["b_p"]:.2e}', f'{R["b_std"]:.4f}', "Yes" if R["cond"][2] else "No"],
        ["Direct Effect (c'): X -> Y | M",
         f'{R["cp"]:.6f}', f'{R["cp_se"]:.6f}', f'{R["cp_t"]:.3f}',
         f'{R["cp_p"]:.2e}', f'{R["cp_std"]:.4f}',
         "Yes" if R["cp_p"] < ALPHA else "No"],
    ]
    table = ax.table(cellText=data, colLabels=header, loc="center", cellLoc="center",
                     colWidths=[0.22, 0.12, 0.12, 0.10, 0.14, 0.12, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    for j in range(len(header)):
        table[(0, j)].set_facecolor(COLORS["dark"])
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(data) + 1):
        for j in range(len(header)):
            table[(i, j)].set_text_props(family="monospace", fontsize=9)
            if i % 2 == 0:
                table[(i, j)].set_facecolor(COLORS["bg"])
    ax.text(0.5, 0.96, "A. Path Coefficients", transform=ax.transAxes,
            fontsize=12, fontweight="bold", ha="center")

    # B - Indirect effect
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    itxt = (f"INDIRECT EFFECT (a x b)\n"
            f"{'_' * 38}\n\n"
            f"Point Estimate:     {R['ab']:.6f}\n"
            f"95% Percentile CI:  [{R['ci_lo']:.6f}, {R['ci_hi']:.6f}]\n"
            f"CI Width:           {R['ci_hi'] - R['ci_lo']:.6f}\n\n"
            f"Significant:  {'YES (CI excludes zero)' if R['ci_sig'] else 'NO'}\n\n"
            f"Sobel Z = {R['sob_z']:.3f}   p = {R['sob_p']:.2e}\n\n"
            f"Bootstrap resamples: {N_BOOTSTRAP:,}")
    ax.text(0.5, 0.5, itxt, transform=ax.transAxes, fontsize=10,
            ha="center", va="center", family="monospace", fontweight="bold",
            bbox=dict(boxstyle="round,pad=1", fc="#ECF0F1", ec="#34495E", lw=2))
    ax.text(0.5, 0.96, "B. Indirect Effect", transform=ax.transAxes,
            fontsize=12, fontweight="bold", ha="center")

    # C - Mediation type
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    tc = COLORS["sleep"] if "PARTIAL" in R["med_type"] else (
         COLORS["positive"] if "FULL" in R["med_type"] else COLORS["negative"])
    mtxt = (f"MEDIATION TYPE\n{'_' * 38}\n\n"
            f"{R['med_type']}\n\n"
            f"Proportion Mediated: {R['prop']:.1f}%\n\n"
            f"Effect Reduction:\n"
            f"  c  = {R['c']:.6f}\n"
            f"  c' = {R['cp']:.6f}\n"
            f"  Delta  = {abs(R['c'] - R['cp']):.6f}")
    ax.text(0.5, 0.5, mtxt, transform=ax.transAxes, fontsize=10,
            ha="center", va="center", family="monospace", fontweight="bold", color=tc,
            bbox=dict(boxstyle="round,pad=1", fc=tc, alpha=0.12, ec=tc, lw=2.5))
    ax.text(0.5, 0.96, "C. Mediation Conclusion", transform=ax.transAxes,
            fontsize=12, fontweight="bold", ha="center")

    # D - Conditions checklist
    ax = fig.add_subplot(gs[2, 0])
    cnames = ["Total Effect\nSig (c)", "X -> M\nSig (a)",
              "M -> Y\nSig (b)", "Effect\nReduced"]
    cvals = R["cond"]
    ccols = [COLORS["positive"] if v else COLORS["negative"] for v in cvals]
    bars = ax.bar(range(4), [1]*4, color=ccols, edgecolor="black", lw=1.5, width=0.55)
    for bar, v in zip(bars, cvals):
        ax.text(bar.get_x() + bar.get_width()/2, 0.5,
                "PASS" if v else "FAIL", ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
    ax.set_xticks(range(4))
    ax.set_xticklabels(cnames, fontweight="bold", fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_title("D. Baron-Kenny Conditions", fontweight="bold")

    # E - Model quality
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    qtxt = (f"MODEL QUALITY\n{'_' * 38}\n\n"
            f"R2 (Total-Effect Model):  {R['r2_total']:.4f}\n"
            f"R2 (Mediated Model):      {R['r2_med']:.4f}\n\n"
            f"Delta R2 (mediator):      {R['r2_med'] - R['r2_total']:.4f}\n"
            f"Cohen's f2:               {R['f2']:.4f}\n\n"
            f"Sample Size:  n = {R['n']:,}\n"
            f"Bootstrap:    B = {N_BOOTSTRAP:,}")
    ax.text(0.5, 0.5, qtxt, transform=ax.transAxes, fontsize=10,
            ha="center", va="center", family="monospace", fontweight="bold",
            bbox=dict(boxstyle="round,pad=1", fc=COLORS["bg"], ec="#34495E", lw=2))
    ax.text(0.5, 0.96, "E. Model Quality", transform=ax.transAxes,
            fontsize=12, fontweight="bold", ha="center")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Fig6_Summary.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved -> Fig6_Summary.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("BARON-KENNY MEDIATION ANALYSIS")
    print("Caffeine -> Stress -> Sleep Duration")
    print("=" * 70 + "\n")

    df   = load_and_clean_data()
    desc = compute_descriptives(df)
    corr = compute_correlations(df)
    vif  = check_assumptions(df)
    R    = run_mediation(df)

    print("\n" + "=" * 70)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 70)
    fig1_descriptive(df)
    fig2_correlations(df, corr)
    fig3_mediation_diagram(R)
    fig4_baron_kenny_steps(R)
    fig5_bootstrap(R)
    fig6_summary(R)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Sample:  n = {len(df):,}")
    print(f"  Result:  {R['med_type']}  ({R['prop']:.1f}% mediated)")
    print(f"  Indirect effect = {R['ab']:.6f},  CI [{R['ci_lo']:.6f}, {R['ci_hi']:.6f}]")
    print(f"\n  Output directory: {REPORTS_DIR}")
    print(f"    descriptive_statistics.csv")
    print(f"    correlation_matrix.csv")
    print(f"    mediation_results.csv")
    print(f"    assumption_diagnostics.csv")
    print(f"    figures/Fig1-Fig6  (6 PNG files, 300 dpi)")
    print("=" * 70 + "\n")

    return df, R


if __name__ == "__main__":
    df, result = main()
