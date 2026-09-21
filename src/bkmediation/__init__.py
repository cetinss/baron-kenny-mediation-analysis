"""
bkmediation - a small, tested package for Baron-Kenny mediation analysis.

The package implements the analysis reported in

    "Mediating effect of perceived stress on the relationship between
     caffeine intake and sleep duration"

as a set of documented, independently testable modules rather than a single
analysis script (Reviewer 3, general comment on the GitHub repository).

Public API
----------
    build_analytic_sample  - reproducible cleaning pipeline -> AnalyticSample
    run_mediation          - Baron-Kenny paths + bootstrap inference
    compare_sample_definitions - full raw n=10,000 vs the cleaned samples
    design_effect_sensitivity  - ICC that would overturn a clustered result
    MediationResult        - result object (paths table, effect sizes, CIs)
    evaluate_hypotheses    - formal H1/H2/H3 decisions
    collect_provenance     - Python/package versions, seed, git commit
    run_benchmarks         - runtime, scalability, cross-software agreement

Typical use
-----------
    from bkmediation import build_analytic_sample, run_mediation
    sample = build_analytic_sample()
    result = run_mediation(sample)
    print(result.paths_table())

Command line
------------
    python -m bkmediation all
"""

from .clustering import design_effect_sensitivity
from .comparisons import compare_gender_coding, compare_sample_definitions
from .config import (
    ALPHA,
    CAFFEINE_SCALE,
    DEFAULT_COVARIATES,
    N_BOOTSTRAP,
    RANDOM_SEED,
    STRESS_CODING,
)
from .data import AnalyticSample, build_analytic_sample, load_and_clean_data
from .hypotheses import HYPOTHESES, evaluate_hypotheses
from .mediation import MediationResult, run_mediation
from .provenance import collect_provenance

__version__ = "1.0.0"

__all__ = [
    "ALPHA",
    "compare_gender_coding",
    "compare_sample_definitions",
    "design_effect_sensitivity",
    "CAFFEINE_SCALE",
    "DEFAULT_COVARIATES",
    "N_BOOTSTRAP",
    "RANDOM_SEED",
    "STRESS_CODING",
    "AnalyticSample",
    "build_analytic_sample",
    "load_and_clean_data",
    "HYPOTHESES",
    "evaluate_hypotheses",
    "MediationResult",
    "run_mediation",
    "collect_provenance",
    "__version__",
]
