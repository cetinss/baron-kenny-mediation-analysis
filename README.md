# Caffeine, Stress and Sleep: a tested Baron-Kenny mediation pipeline

This repository accompanies a mediation study asking whether perceived stress
mediates the association between daily caffeine intake and sleep duration. The
analysis follows Baron & Kenny (1986) with percentile-bootstrap inference and
heteroskedasticity-robust standard errors.

> **The primary dataset is synthetic.** `data/synthetic_coffee_health_10000.csv`
> was generated for demonstration purposes, and its `Stress_Level` field was
> produced partly from sleep- and lifestyle-related quantities. Mediator and
> outcome are therefore not independently measured, and the indirect effect
> reported here is a **statistical association inside a simulated
> data-generating process** and a demonstration of the pipeline - not evidence
> about a biological mechanism. Findings are validated against two independent
> datasets (LEMURS, NHANES); see `outputs/`.

## Study design

| Role | Variable | Operationalisation |
|---|---|---|
| Exposure (X) | Daily caffeine intake | mg/day; coefficients reported **per 100 mg** |
| Mediator (M) | Perceived stress | ordinal Low/Medium/High coded 2/5/8 (sensitivity-tested) |
| Outcome (Y) | Sleep duration | hours/night |
| Covariates | Age, gender (Female reference; Male and Other indicators), BMI, physical activity, resting heart rate | |

Analytic sample: **n = 9,795** after the age filter (18-65), listwise deletion
and a 1.5 x IQR outlier rule. Every exclusion is listed in
`Reports/sample_exclusions.csv`.

## Hypotheses

| | Hypothesis | Decision rule |
|---|---|---|
| H1 | Higher daily caffeine intake is associated with shorter sleep duration | total effect c < 0, 95% CI excludes zero |
| H2 | Higher daily caffeine intake is associated with higher perceived stress | path a > 0, 95% CI excludes zero |
| H3 | Perceived stress mediates the caffeine-sleep association | percentile-bootstrap 95% CI for a*b excludes zero, effect negative |

The rules live in code (`src/bkmediation/hypotheses.py`), are evaluated by
`evaluate_hypotheses()` and are covered by tests, so the text and the analysis
cannot drift apart.

## Repository layout

```
src/bkmediation/          installable, documented, tested analysis package
    config.py             every analytic constant in one place
    data.py               loading, coding, cleaning -> AnalyticSample
    mediation.py          paths, robust SEs, bootstrap, effect sizes
    hypotheses.py         H1-H3 decision rules
    descriptives.py       Table 1 helpers
    diagnostics.py        VIF, Durbin-Watson, Breusch-Pagan, Jarque-Bera
    comparisons.py        sample-definition and gender-coding comparisons
    clustering.py         design-effect bound for clustered data
    benchmark.py          runtime, scalability, cross-software agreement
    provenance.py         versions, seed, git commit
    reporting.py          CSV and markdown writers
    cli.py                python -m bkmediation <command>
src/main.py               manuscript entry point: package + six report figures
src/make_manuscript_figures.py   manuscript Figures 1-8
src/sensitivity_analysis.py      stress-coding, outlier and confounding checks
src/lemurs_clustering_check.py   how much clustering would overturn LEMURS
src/lemurs_validation.py         external validation (LEMURS)
src/nhanes_mediation*.py         external validation (NHANES)
tests/                    pytest suite (107 tests)
Reports/                  tables and figures for the main analysis
outputs/                  sensitivity, external validation, benchmarks, provenance
```

## Installation

```bash
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate

pip install -e ".[test]"        # package + pytest
pip install -e ".[test,compare]"  # also pingouin, for the software comparison
```

## Running the analysis

```bash
python -m bkmediation analyze      # cleaning, estimation, all report tables
python -m bkmediation hypotheses   # H1/H2/H3 decision table
python -m bkmediation samples      # full raw n=10,000 vs each cleaned sample
python -m bkmediation gender       # corrected vs legacy gender coding
python -m bkmediation benchmark    # runtime, scalability, software comparison
python -m bkmediation provenance   # versions, seed, commit
python -m bkmediation all          # everything above

python src/main.py                 # same analysis plus the six report figures
python src/make_manuscript_figures.py
python src/sensitivity_analysis.py
```

`python src/main.py` and `python -m bkmediation analyze` report the same
numbers because both call the same package functions.

## Tests

```bash
pytest                    # full suite
pytest -m "not slow"      # skip the benchmark tests
```

The suite checks the estimator against simulated data with a known indirect
effect (so there is a right answer to recover), checks that the mediation
conclusion flips to "no mediation" when the mediating path is removed, pins
the published coefficients, and verifies that the indirect effect agrees with
independent implementations (`statsmodels` closed form, `statsmodels`
`Mediation`, `pingouin`).

## What is reported

* Every path (a, b, c, c') with an HC3 robust SE, a model-based 95% CI, a
  classical 95% CI and a percentile-bootstrap 95% CI.
* Exposure coefficients rescaled per 100 mg of caffeine per day.
* The indirect effect a*b with a 5,000-resample percentile-bootstrap interval
  (the primary inferential statement); the Sobel test only as a secondary
  normal-theory comparison.
* The proportion mediated **with a bootstrap interval**, since a*b/c is an
  unstable ratio.
* Cohen's f2 for the mediator's variance increment, explicitly labelled as
  such: it is not a mediation effect size and is not interpreted as one. The
  mediation effect sizes are the standardised indirect effect and the
  proportion mediated.
* The same model on the dataset exactly as distributed (n = 10,000, no age
  filter, no outlier rule) alongside each cleaned sample, so the cost of every
  exclusion rule is visible.

## Reproducibility

Random seed 42; 5,000 bootstrap resamples; the exact Python, package and
repository versions are written to `outputs/provenance.json` and
`outputs/provenance.md` on every run. `outputs/benchmark_report.md` records
runtime, scaling behaviour and agreement with other mediation software.

## Data sources and licensing

The MIT licence in `LICENSE` covers the code in this repository. The bundled
datasets are third-party material, redistributed here so the analysis can be
reproduced, and remain under their own terms:

| Dataset | Source | Status |
|---|---|---|
| `data/synthetic_coffee_health_10000.csv` | Global Coffee Health Dataset, L. Tharmalingam (Kaggle) | Synthetic; cite the original Kaggle entry |
| `data/lemurs/` | LEMURS differentially private synthetic release (AIM, eps = 5) | Cite Ghasemizade et al., *JAMIA Open*, as the dataset README requires |
| `data/nhanes/*.XPT` | NHANES 2017-2018, CDC / NCHS | US public domain |

## Citation

Please cite this repository using `CITATION.cff`.

## License

MIT. See `LICENSE`.
