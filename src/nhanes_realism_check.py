"""
================================================================================
NHANES Distributional Realism Check for the Global Coffee Health Dataset
================================================================================
Purpose
-------
The Global Coffee Health dataset (Kaggle, uom190346a) is confirmed "purely
synthetic" by its creator (personal communication, March 2026). This script
builds a real-world comparison sample from NHANES 2017-2018 (CDC/NCHS,
nationally representative US survey) covering the closest available matches
to our analysis variables: caffeine intake (mg, dietary recall), sleep
duration, BMI, resting heart rate, physical activity, age and gender.

Data source: official CDC public files, downloaded directly (no
authentication required), NOT the full 5,078-variable harmonized 1988-2018
mega-file -- only the 6 domain files needed for our variables:
  DEMO_J (demographics), DR1TOT_J (dietary caffeine), BMX_J (body measures),
  BPX_J (pulse), PAQ_J (physical activity, GPAQ-based), SLQ_J (sleep).

Sampling note: rows are NOT taken as "the first N" -- NHANES SEQN order is
clustered by data-collection sequence and is not a random shuffle, so a
head(N) sample would risk non-random selection bias. Instead we draw a
simple random sample (fixed seed, reproducible) from the complete-case pool.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
N_SAMPLE = 300

PROJECT_DIR = Path(__file__).parent.parent
NHANES_DIR = PROJECT_DIR / "data" / "nhanes"
OUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"

COLORS = {
    "caffeine": "#8B4513", "stress": "#E67E22", "sleep": "#1F4788",
    "coffee_health": "#1F4788", "nhanes": "#27AE60",
}


def load_nhanes():
    print("=" * 70)
    print("STEP 1: LOAD NHANES 2017-2018 DOMAIN FILES AND MERGE ON SEQN")
    print("=" * 70)

    demo = pd.read_sas(NHANES_DIR / "DEMO_J.XPT", format="xport")[["SEQN", "RIAGENDR", "RIDAGEYR"]]
    diet = pd.read_sas(NHANES_DIR / "DR1TOT_J.XPT", format="xport")[["SEQN", "DR1TCAFF"]]
    bmx = pd.read_sas(NHANES_DIR / "BMX_J.XPT", format="xport")[["SEQN", "BMXBMI"]]
    bpx = pd.read_sas(NHANES_DIR / "BPX_J.XPT", format="xport")[["SEQN", "BPXPLS"]]
    slq = pd.read_sas(NHANES_DIR / "SLQ_J.XPT", format="xport")[["SEQN", "SLD012"]]
    paq = pd.read_sas(NHANES_DIR / "PAQ_J.XPT", format="xport")

    df = demo.merge(diet, on="SEQN", how="left") \
              .merge(bmx, on="SEQN", how="left") \
              .merge(bpx, on="SEQN", how="left") \
              .merge(slq, on="SEQN", how="left") \
              .merge(paq, on="SEQN", how="left")
    print(f"Merged sample (all ages): {len(df):,} respondents")
    return df


def build_variables(df):
    print("\n" + "=" * 70)
    print("STEP 2: BUILD ANALYSIS VARIABLES (matching Coffee Health columns)")
    print("=" * 70)

    df = df.copy()
    df["Age"] = df["RIDAGEYR"]
    df["Gender"] = df["RIAGENDR"].map({1.0: "Male", 2.0: "Female"})
    df["Caffeine_mg"] = df["DR1TCAFF"]
    df["BMI"] = df["BMXBMI"]
    df["Heart_Rate"] = df["BPXPLS"]
    # Weekday habitual sleep hours (NHANES does not collect a single
    # "average this week" item like LEMURS/Coffee Health; SLD012 is the
    # closest analogue -- flagged explicitly as a scope difference).
    df["Sleep_Hours"] = df["SLD012"]

    # Physical activity: GPAQ-style total weekly hours across five domains
    # (work-vigorous, work-moderate, transport walk/bike, recreation-vigorous,
    # recreation-moderate). Gate questions (PAQ605/620/635/650/665): 1=Yes,
    # 2=No, 9=Don't know. Day counts (PAQ610/625/640/655/670): 1-7,
    # 99=Don't know. Minutes (PAD615/630/645/660/675): 9999=Don't know.
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
        # "No" (2) on the gate -> 0 hours contribution from this domain.
        is_no = df[gate] == 2
        is_yes = df[gate] == 1
        is_unknown_gate = df[gate].isna()
        contribution = (df[dcol] * df[mcol]) / 60.0
        cat_missing = is_yes & (df[dcol].isna() | df[mcol].isna())
        total_hours = total_hours + contribution.where(is_yes, 0.0).fillna(0.0)
        any_missing = any_missing | is_unknown_gate | cat_missing
    df["Physical_Activity_Hours"] = total_hours.where(~any_missing, np.nan)

    return df


def clean_and_sample(df):
    print("\n" + "=" * 70)
    print("STEP 3: FILTER, COMPLETE-CASE SELECTION, RANDOM SUBSAMPLE")
    print("=" * 70)

    df = df[(df["Age"] >= 18) & (df["Age"] <= 65)].copy()
    print(f"After age filter (18-65): {len(df):,}")

    key_vars = ["Age", "Gender", "Caffeine_mg", "BMI", "Heart_Rate",
                "Sleep_Hours", "Physical_Activity_Hours"]
    df_complete = df.dropna(subset=key_vars).copy()
    print(f"After complete-case filter on {key_vars}: {len(df_complete):,}")

    n = min(N_SAMPLE, len(df_complete))
    sample = df_complete.sample(n=n, random_state=42).reset_index(drop=True)
    print(f"\nRandom subsample drawn (seed=42, NOT first-N): n = {len(sample):,}")
    return df_complete, sample


def compare_and_report(coffee_df, nhanes_sample, nhanes_full):
    print("\n" + "=" * 70)
    print("STEP 4: COMPARISON TABLE + FIGURE")
    print("=" * 70)

    vars_map = {
        "Caffeine_mg": "Caffeine Intake (mg/day)",
        "Sleep_Hours": "Sleep Duration (hours)",
        "BMI": "BMI (kg/m2)",
        "Heart_Rate": "Resting Heart Rate / Pulse (bpm)",
        "Physical_Activity_Hours": "Physical Activity (h/week)",
        "Age": "Age (years)",
    }

    rows = []
    for var, label in vars_map.items():
        c = coffee_df[var]
        n_s = nhanes_sample[var]
        n_f = nhanes_full[var]
        rows.append({
            "Variable": label,
            "CoffeeHealth_Mean": round(c.mean(), 2), "CoffeeHealth_SD": round(c.std(), 2),
            "NHANES_n300_Mean": round(n_s.mean(), 2), "NHANES_n300_SD": round(n_s.std(), 2),
            "NHANES_Full_Mean": round(n_f.mean(), 2), "NHANES_Full_SD": round(n_f.std(), 2),
            "NHANES_Full_n": int(n_f.count()),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(OUT_DIR / "nhanes_comparison_table.csv", index=False)
    nhanes_sample.to_csv(OUT_DIR / "nhanes_random_sample_n300.csv", index=False)

    # Figure: overlaid distributions, Coffee Health vs NHANES full complete-case pool
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle("Global Coffee Health (Synthetic) vs. NHANES 2017-2018 (Real, US) -- Distribution Comparison",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()
    for ax, (var, label) in zip(axes, vars_map.items()):
        c = coffee_df[var].dropna()
        n_f = nhanes_full[var].dropna()
        bins = 30
        ax.hist(c, bins=bins, density=True, alpha=0.5, color=COLORS["coffee_health"],
                label=f"Coffee Health (n={len(c):,})")
        ax.hist(n_f, bins=bins, density=True, alpha=0.5, color=COLORS["nhanes"],
                label=f"NHANES (n={len(n_f):,})")
        ax.set_title(label, fontweight="bold", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "NHANES_Comparison_Distributions.png", facecolor="white", dpi=200)
    plt.close()
    print("\nSaved -> outputs/figures/NHANES_Comparison_Distributions.png")
    print("Saved -> outputs/nhanes_comparison_table.csv")
    print("Saved -> outputs/nhanes_random_sample_n300.csv")
    return table


def main():
    df = load_nhanes()
    df = build_variables(df)
    nhanes_full, nhanes_sample = clean_and_sample(df)

    sys.path.insert(0, str(PROJECT_DIR / "src"))
    from main import load_and_clean_data
    coffee_df = load_and_clean_data()

    table = compare_and_report(coffee_df, nhanes_sample, nhanes_full)
    return coffee_df, nhanes_full, nhanes_sample, table


if __name__ == "__main__":
    coffee_df, nhanes_full, nhanes_sample, table = main()
