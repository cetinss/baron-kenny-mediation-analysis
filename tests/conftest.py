"""Shared fixtures.

The suite is split into two kinds of test:

* tests on SIMULATED data, where the true indirect effect is known by
  construction, so the estimator can be checked against a right answer;
* tests on the project's own analytic sample, which pin the published
  numbers so that a refactor cannot silently move them. Those are skipped
  automatically when the dataset is not present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bkmediation.config import DEFAULT_DATASET  # noqa: E402
from bkmediation.data import build_analytic_sample  # noqa: E402

# Truth used by the simulated fixtures.
TRUE_A = 0.004
TRUE_B = -0.35
TRUE_CP = -0.0009
TRUE_AB = TRUE_A * TRUE_B

COV_NAMES = ["Cov1", "Cov2"]


def make_simulated(n=4000, a=TRUE_A, b=TRUE_B, cp=TRUE_CP, seed=7, noise_y=0.4):
    """Data from a known linear mediation process.

    Caffeine_mg -> Stress_Score (slope a) -> Sleep_Hours (slope b), with a
    direct path cp and two covariates that affect both M and Y.
    """
    rng = np.random.default_rng(seed)
    cov1 = rng.normal(0, 1, n)
    cov2 = rng.integers(0, 2, n).astype(float)
    x = rng.normal(250, 110, n)
    m = 4.0 + a * x + 0.30 * cov1 + 0.20 * cov2 + rng.normal(0, 1.0, n)
    y = 9.0 + cp * x + b * m + 0.10 * cov1 - 0.05 * cov2 + rng.normal(0, noise_y, n)
    return pd.DataFrame({
        "Caffeine_mg": x, "Stress_Score": m, "Sleep_Hours": y,
        "Cov1": cov1, "Cov2": cov2,
    })


@pytest.fixture(scope="session")
def simulated():
    return make_simulated()


@pytest.fixture(scope="session")
def sim_covariates():
    return list(COV_NAMES)


@pytest.fixture(scope="session")
def raw_dataset_available():
    return DEFAULT_DATASET.exists()


@pytest.fixture(scope="session")
def analytic_sample(raw_dataset_available):
    if not raw_dataset_available:
        pytest.skip(f"dataset not available: {DEFAULT_DATASET}")
    return build_analytic_sample()


@pytest.fixture(scope="session")
def analytic_result(analytic_sample):
    from bkmediation.mediation import run_mediation
    return run_mediation(analytic_sample, n_boot=1000)


@pytest.fixture
def small_frame():
    """Tiny hand-checkable frame for the cleaning functions."""
    return pd.DataFrame({
        "Age": [17, 18, 40, 65, 66],
        "Gender": ["Male", "Female", "Other", "Female", "Male"],
        "Stress_Level": ["Low", "Medium", "High", "Low", "Medium"],
        "Caffeine_mg": [100.0, 200.0, 300.0, 400.0, 500.0],
        "Sleep_Hours": [7.0, 6.5, 6.0, 8.0, 7.5],
        "BMI": [22.0, 25.0, 28.0, 24.0, 26.0],
        "Heart_Rate": [70, 72, 80, 65, 75],
        "Physical_Activity_Hours": [5.0, 3.0, 1.0, 7.0, 4.0],
    })
