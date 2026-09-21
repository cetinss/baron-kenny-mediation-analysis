"""Project-wide constants and paths.

Everything that the manuscript reports as an analytic choice is declared here
once, so that a reader can check the settings without reading the estimation
code, and so that the test suite can assert on them.
"""

from pathlib import Path

# ---- Paths ------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "Reports"
FIGURES_DIR = REPORTS_DIR / "figures"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

DEFAULT_DATASET = DATA_DIR / "synthetic_coffee_health_10000.csv"

# ---- Reproducibility --------------------------------------------------------
RANDOM_SEED = 42
N_BOOTSTRAP = 5000
ALPHA = 0.05

# ---- Model specification ----------------------------------------------------
# Exposure is reported per 100 mg of caffeine (Reviewer 3, S9): 100 mg is
# roughly one standard cup of filter coffee, so a one-unit change on this
# scale is an interpretable quantity, unlike a 1 mg change.
CAFFEINE_SCALE = 100.0
CAFFEINE_SCALE_LABEL = "per 100 mg caffeine/day"

# Mediator coding. The equal-interval numeric coding is the manuscript's
# primary specification; sensitivity_analysis.py varies it (unequal spacing,
# indicator coding, ordinal model) because the spacing is not empirically
# given. See outputs/stress_coding_sensitivity.md.
STRESS_CODING = {"Low": 2, "Medium": 5, "High": 8}
STRESS_LEVELS = ("Low", "Medium", "High")

# Gender coding. "three_level" is the corrected default: Female is the
# reference category and Male / Other each get their own indicator, so
# respondents who selected "Other" are no longer pooled with women
# (Reviewer 3, Methods comment). "binary_male" reproduces the original
# Male = 1 / everyone else = 0 coding for backward compatibility.
GENDER_SCHEMES = ("three_level", "binary_male")
DEFAULT_GENDER_SCHEME = "three_level"

# Covariates other than gender. Chosen a priori as variables that plausibly
# affect both perceived stress and sleep duration (common causes), not as
# post-treatment variables: age, adiposity, habitual physical activity and
# resting heart rate.
BASE_COVARIATES = ["Age", "BMI", "Physical_Activity_Hours", "Heart_Rate"]
GENDER_COVARIATES = {
    "three_level": ["Gender_Male", "Gender_Other"],
    "binary_male": ["Gender_Num"],
}
DEFAULT_COVARIATES = (
    BASE_COVARIATES[:1] + GENDER_COVARIATES[DEFAULT_GENDER_SCHEME] + BASE_COVARIATES[1:]
)

# ---- Cleaning rules ---------------------------------------------------------
AGE_RANGE = (18, 65)
OUTLIER_IQR_K = 1.5
CONTINUOUS_FOR_OUTLIERS = [
    "Caffeine_mg",
    "Sleep_Hours",
    "BMI",
    "Heart_Rate",
    "Physical_Activity_Hours",
]

# ---- Variable roles ---------------------------------------------------------
EXPOSURE = "Caffeine_mg"
MEDIATOR = "Stress_Score"
OUTCOME = "Sleep_Hours"

# ---- Robust inference -------------------------------------------------------
# Heteroskedasticity-consistent covariance for the reported standard errors
# (Reviewer 3, S9 / general comment). "nonrobust" restores classical OLS SEs.
DEFAULT_COV_TYPE = "HC3"
