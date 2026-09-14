"""
================================================================================
NHANES 2017-2018 Mediation Analysis (v2)
Caffeine Intake -> Allostatic Load (physiological stress-burden proxy) -> Sleep Duration
================================================================================
Rationale
---------
The first NHANES mediation study (nhanes_mediation_analysis.py) used PHQ-9
(depressive symptoms) as the mediator because NHANES has no Perceived Stress
Scale. PHQ-9 measures a related-but-distinct construct (clinical depressive
symptoms, not perceived stress) and produced a null result.

This second study uses a different, more literature-precedented "chronic
stress" proxy: an ALLOSTATIC LOAD (AL) index (McEwen & Stellar, 1993; Seeman
et al., 1997) -- an objective, physiological composite reflecting cumulative
"wear and tear" from stress exposure, as opposed to PSS's subjective,
self-reported perceived stress.

IMPORTANT CONSTRUCT CAVEAT: AL is NOT perceived stress either. It is an
indirect physiological correlate, and there is no single standardized AL
formula in the literature (a 2016 review found 21 distinct scoring variants
used across NHANES studies). By user's choice, this analysis uses a REDUCED
AL index built only from data files already downloaded for this project
(BMX_J body measures, BPX_J blood pressure) -- 4 cardiovascular/metabolic
components: BMI, systolic BP, diastolic BP, resting heart rate. This is a
narrower AL than the classical 7-8 marker version (which also includes
cholesterol, HbA1c, CRP, albumin) and should be described as such.

AL scoring (standard "high-risk quartile" method, Seeman/Crimmins style)
------------------------------------------------------------------------
For each of the 4 components, 1 point is assigned if the respondent's value
falls in the sample's (survey-weighted) top quartile (>=75th percentile) --
all four components are "higher = worse" (unlike full AL indices, which
also include markers like HDL/albumin scored at the low quartile). AL score
range: 0-4.

Covariate note: because BMI and Heart_Rate are now INSIDE the mediator
construction, they are dropped from the covariate list (including them
would create circularity / mediator-covariate overlap). Covariates here are
Age, Gender, and Physical_Activity_Hours only.

Design handling is identical to nhanes_mediation_analysis.py: WLS with the
dietary day-1 weight WTDRD1, cluster-robust SEs on SDMVSTRA x SDMVPSU, and a
stratified-cluster bootstrap (PSUs resampled within strata) for the indirect
effect.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
NHANES_DIR = PROJECT_DIR / "data" / "nhanes"
OUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"

plt.style.use("default")
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.linewidth": 1.2, "grid.alpha": 0.3,
})
COLORS = {"caffeine": "#8B4513", "al": "#C0392B", "sleep": "#1F4788", "dark": "#2C3E50", "bg": "#F8F9FA"}
_sig = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def load_and_merge():
    print("=" * 70)
    print("STEP 1: LOAD NHANES 2017-2018 FILES")
    print("=" * 70)
    demo = pd.read_sas(NHANES_DIR / "DEMO_J.XPT", format="xport")[
        ["SEQN", "RIAGENDR", "RIDAGEYR", "SDMVPSU", "SDMVSTRA"]]
    diet = pd.read_sas(NHANES_DIR / "DR1TOT_J.XPT", format="xport")[["SEQN", "DR1TCAFF", "WTDRD1"]]
    bmx = pd.read_sas(NHANES_DIR / "BMX_J.XPT", format="xport")[["SEQN", "BMXBMI"]]
    bpx = pd.read_sas(NHANES_DIR / "BPX_J.XPT", format="xport")[
        ["SEQN", "BPXPLS", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4",
         "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]]
    slq = pd.read_sas(NHANES_DIR / "SLQ_J.XPT", format="xport")[["SEQN", "SLD012"]]
    paq = pd.read_sas(NHANES_DIR / "PAQ_J.XPT", format="xport")

    df = demo.merge(diet, on="SEQN", how="left") \
              .merge(bmx, on="SEQN", how="left") \
              .merge(bpx, on="SEQN", how="left") \
              .merge(slq, on="SEQN", how="left") \
              .merge(paq, on="SEQN", how="left")
    print(f"Merged: {len(df):,} respondents")
    return df


def build_variables(df):
    print("\n" + "=" * 70)
    print("STEP 2: BUILD VARIABLES (incl. reduced Allostatic Load index)")
    print("=" * 70)
    df = df.copy()
    df["Age"] = df["RIDAGEYR"]
    df["Gender_Num"] = (df["RIAGENDR"] == 1.0).astype(float)
    df["Caffeine_mg"] = df["DR1TCAFF"]
    df["Sleep_Hours"] = df["SLD012"]
    df["BMI"] = df["BMXBMI"]
    df["Heart_Rate"] = df["BPXPLS"]

    sy_cols = ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]
    di_cols = ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]
    df["SBP"] = df[sy_cols].mean(axis=1, skipna=True)
    df["DBP"] = df[di_cols].replace(0, np.nan).mean(axis=1, skipna=True)

    # Physical activity (same GPAQ construction as the other NHANES scripts)
    gates = ["PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"]
    days_cols = ["PAQ610", "PAQ625", "PAQ640", "PAQ655", "PAQ670"]
    mins_cols = ["PAD615", "PAD630", "PAD645", "PAD660", "PAD675"]
    for c in days_cols:
        df[c] = df[c].replace({77: np.nan, 99: np.nan})
    for c in mins_cols:
        df[c] = df[c].replace({7777: np.nan, 9999: np.nan})
    for g in gates:
        df[g] = df[g].replace({7: np.nan, 9: np.nan})
    total_hours = pd.Series(0.0, index=df.index)
    any_missing = pd.Series(False, index=df.index)
    for gate, dcol, mcol in zip(gates, days_cols, mins_cols):
        is_yes = df[gate] == 1
        is_unknown_gate = df[gate].isna()
        contribution = (df[dcol] * df[mcol]) / 60.0
        cat_missing = is_yes & (df[dcol].isna() | df[mcol].isna())
        total_hours = total_hours + contribution.where(is_yes, 0.0).fillna(0.0)
        any_missing = any_missing | is_unknown_gate | cat_missing
    df["Physical_Activity_Hours"] = total_hours.where(~any_missing, np.nan)

    df["Cluster"] = df["SDMVSTRA"].astype(str) + "_" + df["SDMVPSU"].astype(str)
    return df


def clean_and_score_AL(df):
    print("\n" + "=" * 70)
    print("STEP 3: FILTER, CLEAN, SCORE ALLOSTATIC LOAD")
    print("=" * 70)
    df = df[(df["Age"] >= 18) & (df["Age"] <= 65)].copy()
    print(f"After age filter (18-65): {len(df):,}")

    key_vars = ["Caffeine_mg", "Sleep_Hours", "BMI", "Heart_Rate", "SBP", "DBP",
                "Age", "Gender_Num", "Physical_Activity_Hours", "WTDRD1",
                "SDMVSTRA", "SDMVPSU"]
    n_before = len(df)
    df = df.dropna(subset=key_vars).copy()
    print(f"After complete-case filter: {len(df):,} (dropped {n_before - len(df):,})")

    # IQR trim on continuous variables (AL components excluded from this
    # trim for the same reason as PHQ-9 in the depression study: their
    # upper tail is exactly the physiologically meaningful high-risk range
    # the AL index is designed to flag).
    for col in ["Caffeine_mg", "Sleep_Hours", "Physical_Activity_Hours"]:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        before = len(df)
        df = df[(df[col] >= lo) & (df[col] <= hi)]
        rm = before - len(df)
        if rm > 0:
            print(f"  {col}: {rm} outliers removed (range: {lo:.2f} - {hi:.2f})")

    df["W"] = df["WTDRD1"] / df["WTDRD1"].mean()

    # Allostatic Load: 1 point per component in the (weighted) top quartile.
    def weighted_quantile(values, weights, q):
        order = np.argsort(values)
        v, w = values[order], weights[order]
        cw = np.cumsum(w) - 0.5 * w
        cw /= w.sum()
        return np.interp(q, cw, v)

    al_components = ["BMI", "SBP", "DBP", "Heart_Rate"]
    cutoffs = {}
    for comp in al_components:
        cutoffs[comp] = weighted_quantile(df[comp].values, df["W"].values, 0.75)
        df[f"{comp}_highrisk"] = (df[comp] >= cutoffs[comp]).astype(int)
    df["AL_Score"] = df[[f"{c}_highrisk" for c in al_components]].sum(axis=1)

    print("\nAL component high-risk cutoffs (survey-weighted 75th pct):")
    for c, v in cutoffs.items():
        print(f"  {c}: >= {v:.2f}")
    print(f"\nAL_Score distribution:\n{df['AL_Score'].value_counts().sort_index()}")

    print(f"\nFinal analytic sample: n = {len(df):,}")
    print(f"Strata: {df['SDMVSTRA'].nunique()}   Clusters: {df['Cluster'].nunique()}")
    return df.reset_index(drop=True), cutoffs


# =============================================================================
# MEDIATION (same design-based approach as nhanes_mediation_analysis.py)
# =============================================================================

def fit_wls_clustered(y, X, w, clusters):
    return sm.WLS(y, X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": clusters})


def run_mediation(df):
    print("\n" + "=" * 70)
    print("STEP 4: SURVEY-WEIGHTED BARON-KENNY MEDIATION (M = Allostatic Load)")
    print("=" * 70)

    covs = ["Age", "Gender_Num", "Physical_Activity_Hours"]  # BMI/HR excluded: inside M
    X = df["Caffeine_mg"].values.astype(float)
    M = df["AL_Score"].values.astype(float)
    Y = df["Sleep_Hours"].values.astype(float)
    C = df[covs].values.astype(float)
    W = df["W"].values.astype(float)
    clusters = df["Cluster"].values
    n = len(df)

    Xc, Mc, Cc = X - X.mean(), M - M.mean(), C - C.mean(axis=0)

    X1 = sm.add_constant(np.column_stack([Xc, Cc]))
    m1 = fit_wls_clustered(Y, X1, W, clusters)
    c, c_se, c_t, c_p = m1.params[1], m1.bse[1], m1.tvalues[1], m1.pvalues[1]
    r2_1 = m1.rsquared

    X2 = sm.add_constant(np.column_stack([Xc, Cc]))
    m2 = fit_wls_clustered(Mc, X2, W, clusters)
    a, a_se, a_t, a_p = m2.params[1], m2.bse[1], m2.tvalues[1], m2.pvalues[1]
    r2_2 = m2.rsquared

    X3 = sm.add_constant(np.column_stack([Xc, Mc, Cc]))
    m3 = fit_wls_clustered(Y, X3, W, clusters)
    cp, cp_se, cp_t, cp_p = m3.params[1], m3.bse[1], m3.tvalues[1], m3.pvalues[1]
    b, b_se, b_t, b_p = m3.params[2], m3.bse[2], m3.tvalues[2], m3.pvalues[2]
    r2_3 = m3.rsquared

    ab = a * b
    sob_se = np.sqrt(a**2 * b_se**2 + b**2 * a_se**2)
    sob_z = ab / sob_se if sob_se > 0 else 0
    sob_p = 2 * (1 - stats.norm.cdf(abs(sob_z)))

    print(f"\n  Stratified cluster bootstrap ({N_BOOTSTRAP:,} resamples) ...")
    strata, psu = df["SDMVSTRA"].values, df["SDMVPSU"].values
    strata_to_psus = {s: np.unique(psu[strata == s]) for s in np.unique(strata)}
    row_idx = np.arange(n)
    boot = np.empty(N_BOOTSTRAP)
    ok = 0
    for _ in range(N_BOOTSTRAP):
        sel_rows = []
        for s, psus in strata_to_psus.items():
            draw = np.random.choice(psus, size=len(psus), replace=True)
            for p in draw:
                sel_rows.append(row_idx[(strata == s) & (psu == p)])
        idx = np.concatenate(sel_rows)
        Xb, Mb, Yb, Cb, Wb = Xc[idx], Mc[idx], Y[idx], Cc[idx], W[idx]
        try:
            a_b = sm.WLS(Mb, sm.add_constant(np.column_stack([Xb, Cb])), weights=Wb).fit().params[1]
            b_b = sm.WLS(Yb, sm.add_constant(np.column_stack([Xb, Mb, Cb])), weights=Wb).fit().params[2]
            boot[ok] = a_b * b_b
            ok += 1
        except Exception:
            continue
    boot = boot[:ok]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    ci_sig = not (ci_lo <= 0 <= ci_hi)

    cond = [c_p < ALPHA, a_p < ALPHA, b_p < ALPHA, abs(cp) < abs(c)]
    all_ok = all(cond)
    med_type = ("FULL MEDIATION" if (all_ok and ci_sig and cp_p >= ALPHA) else
                "PARTIAL MEDIATION" if (all_ok and ci_sig) else "NO MEDIATION")
    prop = (ab / c * 100) if c != 0 else 0

    sx, sm_, sy = X.std(), M.std(), Y.std()
    a_std, b_std, c_std, cp_std = a*sx/sm_, b*sm_/sy, c*sx/sy, cp*sx/sy
    f2 = (r2_3 - r2_1) / (1 - r2_3) if r2_3 < 1 else np.nan

    predictors = sm.add_constant(np.column_stack([Xc, Mc, Cc]))
    vif_vals = [variance_inflation_factor(predictors, i) for i in range(1, predictors.shape[1])]
    dw = durbin_watson(m3.resid)

    print(f"  c  (total)  = {c:.5f}  SE={c_se:.5f}  p = {c_p:.3e} {_sig(c_p)}")
    print(f"  a  (X->M)   = {a:.5f}  SE={a_se:.5f}  p = {a_p:.3e} {_sig(a_p)}")
    print(f"  b  (M->Y|X) = {b:.5f}  SE={b_se:.5f}  p = {b_p:.3e} {_sig(b_p)}")
    print(f"  c' (direct) = {cp:.5f}  SE={cp_se:.5f}  p = {cp_p:.3e} {_sig(cp_p)}")
    print(f"  Indirect (a*b) = {ab:.6f}  95% CI [{ci_lo:.6f}, {ci_hi:.6f}]  sig={ci_sig}")
    print(f"  Proportion mediated = {prop:.1f}%")
    print(f"  R2 total={r2_1:.4f}  R2 mediated={r2_3:.4f}  f2={f2:.4f}")
    print(f"  Max VIF={max(vif_vals):.2f}  Durbin-Watson={dw:.3f}")
    print(f"  => {med_type}")

    return dict(
        n=n, a=a, a_se=a_se, a_t=a_t, a_p=a_p, a_std=a_std,
        b=b, b_se=b_se, b_t=b_t, b_p=b_p, b_std=b_std,
        c=c, c_se=c_se, c_t=c_t, c_p=c_p, c_std=c_std,
        cp=cp, cp_se=cp_se, cp_t=cp_t, cp_p=cp_p, cp_std=cp_std,
        ab=ab, ab_std=a_std*b_std, ci_lo=ci_lo, ci_hi=ci_hi, ci_sig=ci_sig,
        sob_z=sob_z, sob_p=sob_p, r2_total=r2_1, r2_a=r2_2, r2_med=r2_3,
        f2=f2, prop=prop, vif=dict(zip(["Caffeine", "AL", *covs], vif_vals)),
        dw=dw, cond=cond, med_type=med_type, boot=boot, model3=m3,
    )


# =============================================================================
# FIGURES
# =============================================================================

def fig_path_diagram(R):
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    fig.suptitle("NHANES 2017-2018 -- Survey-Weighted Mediation Model\n"
                 "Caffeine Intake -> Allostatic Load (physiological stress-burden proxy) -> Sleep Duration",
                 fontsize=13.5, fontweight="bold", y=0.98)
    for xy, label, sub, color in [
        ((0.3, 4.0), "CAFFEINE\nINTAKE (mg)", "(Dietary recall, day 1)", COLORS["caffeine"]),
        ((3.8, 6.8), "ALLOSTATIC LOAD\n(0-4)", "(Mediator, NOT perceived stress)", COLORS["al"]),
        ((7.5, 4.0), "SLEEP\nDURATION (h)", "(Outcome)", COLORS["sleep"]),
    ]:
        box = FancyBboxPatch(xy, 2.4, 1.8, boxstyle="round,pad=0.1", facecolor=color, edgecolor="black", lw=2.5)
        ax.add_patch(box)
        cx, cy = xy[0]+1.2, xy[1]+1.05
        ax.text(cx, cy, label, ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        ax.text(cx, cy-0.45, sub, ha="center", va="center", fontsize=7.5, color="white", style="italic")
    arr_a = FancyArrowPatch((2.6, 5.3), (3.9, 7.3), arrowstyle="->", mutation_scale=30, lw=3, color="#16A085", zorder=5)
    ax.add_patch(arr_a)
    ax.text(2.8, 6.6, f"a = {R['a']:.4f}\np = {R['a_p']:.3e}{_sig(R['a_p'])}", ha="center", fontsize=9,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", fc="#ECFDF5", ec="#16A085", lw=1.5))
    arr_b = FancyArrowPatch((6.3, 7.3), (7.5, 5.3), arrowstyle="->", mutation_scale=30, lw=3, color="#C0392B", zorder=5)
    ax.add_patch(arr_b)
    ax.text(7.5, 6.6, f"b = {R['b']:.4f}\np = {R['b_p']:.3e}{_sig(R['b_p'])}", ha="center", fontsize=9,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", fc="#FADBD8", ec="#C0392B", lw=1.5))
    arr_c = FancyArrowPatch((2.6, 5.5), (7.5, 5.5), arrowstyle="->", mutation_scale=25, lw=2, color="gray", ls="--", alpha=0.7)
    ax.add_patch(arr_c)
    ax.text(5.0, 5.85, f"c = {R['c']:.4f} (Total Effect)", ha="center", fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray"))
    arr_cp = FancyArrowPatch((2.6, 4.5), (7.5, 4.5), arrowstyle="->", mutation_scale=30, lw=3, color="#8E44AD", zorder=4)
    ax.add_patch(arr_cp)
    ax.text(5.0, 3.85, f"c' = {R['cp']:.4f} (Direct Effect)", ha="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#F4ECF7", ec="#8E44AD", lw=2))
    rbox = FancyBboxPatch((0.3, 0.5), 9.4, 1.6, boxstyle="round,pad=0.15", fc=COLORS["bg"], ec=COLORS["dark"], lw=2)
    ax.add_patch(rbox)
    rtxt = (f"Indirect Effect (a x b) = {R['ab']:.6f}   |   95% Stratified-Cluster Bootstrap CI = "
            f"[{R['ci_lo']:.6f}, {R['ci_hi']:.6f}]   |   Significant: {'YES' if R['ci_sig'] else 'NO'}\n"
            f"Proportion Mediated = {R['prop']:.1f}%   |   Conclusion: {R['med_type']}")
    ax.text(5.0, 1.3, rtxt, ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=COLORS["dark"], family="monospace")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "NHANES_AL_Fig1_Path_Diagram.png", facecolor="white")
    plt.close()


def fig_al_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor("white")
    counts = df["AL_Score"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=COLORS["al"], alpha=0.85, edgecolor="black")
    for x, v in counts.items():
        ax.text(x, v + 5, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Allostatic Load Score (0-4 high-risk components)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Distribution of Reduced Allostatic Load Index (BMI, SBP, DBP, Heart Rate)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "NHANES_AL_Fig2_Distribution.png", facecolor="white")
    plt.close()


def fig_bootstrap(R):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.hist(R["boot"], bins=60, color=COLORS["sleep"], alpha=0.75, edgecolor="black", lw=0.4, density=True)
    ax.axvline(R["ab"], color="red", lw=2.5, label=f'Point est = {R["ab"]:.6f}')
    ax.axvline(R["ci_lo"], color="orange", ls="--", lw=2, label=f'2.5th = {R["ci_lo"]:.6f}')
    ax.axvline(R["ci_hi"], color="orange", ls="--", lw=2, label=f'97.5th = {R["ci_hi"]:.6f}')
    ax.axvline(0, color="black", ls=":", lw=1.5, label="Null (zero)")
    ax.set_xlabel("Indirect Effect (a x b)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title(f"Stratified Cluster Bootstrap of Indirect Effect (B={N_BOOTSTRAP:,})", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "NHANES_AL_Fig3_Bootstrap.png", facecolor="white")
    plt.close()


def main():
    df = load_and_merge()
    df = build_variables(df)
    df, cutoffs = clean_and_score_AL(df)
    R = run_mediation(df)
    fig_al_distribution(df)
    fig_path_diagram(R)
    fig_bootstrap(R)
    df.to_csv(OUT_DIR / "nhanes_AL_analytic_sample.csv", index=False)
    print("\nSaved -> outputs/figures/NHANES_AL_Fig1-3, outputs/nhanes_AL_analytic_sample.csv")
    return df, R, cutoffs


if __name__ == "__main__":
    df, R, cutoffs = main()
