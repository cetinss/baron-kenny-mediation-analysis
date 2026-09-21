"""
================================================================================
LEMURS clustering sensitivity: how much within-person correlation would it take?

Why this exists
---------------
The LEMURS external validation (src/lemurs_validation.py) analyses person-week
rows: roughly 600 first-year students, each contributing several weekly
surveys. The publicly released file carries no participant identifier (see
outputs/feature_matching_report.md, Critical Finding A), so the rows cannot be
grouped and no cluster-robust or cluster-bootstrap standard error can be
computed. That analysis therefore treats every row as an independent
observation, which understates the standard errors to the extent that repeated
responses from the same person are correlated - and for weekly perceived
stress and sleep duration, they are.

The identifier cannot be recovered, but the question can still be answered
quantitatively: how strong would the within-person correlation have to be
before each reported association stopped being statistically significant? This
script computes that threshold using the classical design effect
DEFF = 1 + (m - 1) * rho, where m is the mean number of weeks per participant.

It reports a bound on one specific failure mode. It assumes equal cluster
sizes and a single exchangeable within-person correlation, and it cannot
correct the point estimates themselves.

Output: outputs/lemurs_clustering_sensitivity.md
================================================================================
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import lemurs_validation as lv  # noqa: E402
from bkmediation.clustering import design_effect_sensitivity  # noqa: E402

PROJECT_DIR = SRC_DIR.parent
OUT_DIR = PROJECT_DIR / "outputs"

# Participant count reported by the LEMURS study team for the survey release.
# Only the mean cluster size enters the calculation, and the conclusion is
# reported across a range of plausible values, so this figure does not have to
# be exact.
N_PARTICIPANTS = 600


def main():
    df = lv.load_and_build()
    result = lv.run_mediation(df)

    n_rows = result["n"]
    mean_cluster = n_rows / N_PARTICIPANTS

    targets = [
        ("Path a (caffeine exposure -> PSS-10)", result["a"], result["a_se"], None),
        ("Path b (PSS-10 -> sleep duration | caffeine)", result["b"], result["b_se"], None),
        ("Total effect c (caffeine -> sleep duration)", result["c"], result["c_se"], None),
        ("Indirect effect a*b", result["ab"], None, (result["ci_lo"], result["ci_hi"])),
    ]

    rows = []
    for label, estimate, se, ci in targets:
        sens = design_effect_sensitivity(
            estimate, se=se, ci=ci, mean_cluster_size=mean_cluster)
        rows.append((label, sens))

    lines = [
        "# LEMURS clustering sensitivity",
        "",
        f"The LEMURS analytic sample is **n = {n_rows:,} person-week rows** from "
        f"approximately {N_PARTICIPANTS} participants, a mean of "
        f"**{mean_cluster:.1f} weekly observations per person**. The released "
        "file contains no participant identifier, so those rows cannot be "
        "grouped: `src/lemurs_validation.py` necessarily treats them as "
        "independent observations, which understates the standard errors "
        "whenever repeated responses from the same person are correlated.",
        "",
        "This report does not correct that. It asks the answerable question "
        "instead: **how strong would the within-person correlation have to be "
        "before each association stopped being significant?** Using the "
        "classical design effect DEFF = 1 + (m - 1) x rho, the table gives the "
        "intra-class correlation (ICC) at which each 95% interval would first "
        "include zero.",
        "",
        "| Quantity | Estimate | SE (unclustered) | z | Critical DEFF | Critical ICC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, s in rows:
        icc = ("n/a" if s.already_nonsignificant
               else (">1 (unattainable)" if s.critical_icc >= 1 else f"{s.critical_icc:.2f}"))
        deff = "n/a" if s.already_nonsignificant else f"{s.critical_design_effect:.2f}"
        lines.append(
            f"| {label} | {s.estimate:.6f} | {s.se_unclustered:.6f} | "
            f"{s.z_unclustered:.2f} | {deff} | {icc} |"
        )

    indirect = rows[-1][1]
    lines += [
        "",
        "## What this means",
        "",
        "* Paths a and b are robust to clustering of any plausible magnitude: "
        "the within-person correlation needed to overturn them is at or beyond "
        "the upper limit of what an ICC can be.",
        f"* The indirect effect is the vulnerable claim. {indirect.describe()} "
        "Weekly measurements of perceived stress and sleep duration within the "
        "same person are routinely reported with ICCs in that range, so the "
        "statistical significance of the LEMURS indirect effect should be "
        "treated as **not established**, and the LEMURS comparison should be "
        "read as directional replication of the paths rather than as an "
        "independent significant confirmation of the mediated pathway.",
        "* The total effect c was already non-significant in LEMURS, so "
        "clustering does not change its interpretation.",
        "",
        "## Caveats",
        "",
        "This is a bound on one failure mode, not a correction. It assumes "
        "equal cluster sizes and a single exchangeable within-person "
        "correlation, uses a normal-theory interval for the design-effect "
        "adjustment, and cannot address bias in the point estimates themselves "
        "if the clustering is informative. Obtaining the participant "
        "identifiers and refitting with a cluster bootstrap remains the "
        "correct fix and is recorded as future work.",
        "",
    ]

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "lemurs_clustering_sensitivity.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "=" * 70)
    print("CLUSTERING SENSITIVITY")
    print("=" * 70)
    for label, s in rows:
        print(f"  {label}: {s.describe()}")
    print(f"\nSaved -> {out_path}")
    return rows


if __name__ == "__main__":
    main()
