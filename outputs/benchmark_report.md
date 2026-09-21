# Computational Benchmarks

Machine: Windows-11-10.0.26200-SP0; Python 3.13.5; numpy 2.5.2, statsmodels 0.14.6.
Seed 42; commit `3b8690ed`.

## 1. Runtime of the analysis pipeline

| Stage                   |   Median seconds |   Min seconds |   Max seconds |   Repeats |
|:------------------------|-----------------:|--------------:|--------------:|----------:|
| load_and_clean          |           0.0256 |        0.0254 |        0.027  |         3 |
| fit_paths               |           0.0155 |        0.0154 |        0.0159 |         3 |
| bootstrap_5000          |          10.559  |       10.3276 |       10.8251 |         3 |
| TOTAL (single analysis) |          10.6001 |      nan      |      nan      |         3 |

## 2. Scalability

Samples at or below the analytic sample size are drawn from the analytic sample itself; larger ones are simulated with the same covariate structure, so the curve extends beyond the study's n.

|      n | Source                       |   Fit seconds |   Bootstrap seconds (B=500) |   Bootstrap us per replicate |
|-------:|:-----------------------------|--------------:|----------------------------:|-----------------------------:|
|    500 | analytic sample (subsampled) |        0.0032 |                       0.092 |                        184.5 |
|   1000 | analytic sample (subsampled) |        0.0037 |                       0.129 |                        258.9 |
|   2500 | analytic sample (subsampled) |        0.0061 |                       0.354 |                        708.8 |
|   5000 | analytic sample (subsampled) |        0.0079 |                       0.567 |                       1134.5 |
|   9795 | analytic sample (subsampled) |        0.0156 |                       1.006 |                       2012.2 |
|  25000 | simulated                    |        0.0299 |                       2.733 |                       5465.7 |
| 100000 | simulated                    |        0.1153 |                      12.612 |                      25223.1 |

Bootstrap cost is linear in both the number of replicates and the sample size, because each replicate is a least-squares solve on an n x (k+2) design.

## 3. Agreement with established mediation software

All effects are per 100 mg of caffeine per day.

| Software                                               | Estimator                                |   Indirect effect (per 100 mg) |   95% CI low |   95% CI high |   Direct effect (per 100 mg) |   Total effect (per 100 mg) |   Monte Carlo SE |   Seconds |
|:-------------------------------------------------------|:-----------------------------------------|-------------------------------:|-------------:|--------------:|-----------------------------:|----------------------------:|-----------------:|----------:|
| bkmediation 1.0.0 (this package, B=1000)               | OLS paths + percentile bootstrap         |                      -0.102146 |    -0.115004 |    -0.0883645 |                   -0.0665095 |                   -0.168656 |     nan          |   nan     |
| statsmodels OLS (formula API), product of coefficients | closed form, no resampling               |                      -0.102146 |   nan        |   nan         |                   -0.0665095 |                   -0.168656 |     nan          |     0.035 |
| statsmodels.stats.mediation.Mediation (n_rep=1000)     | Imai-Keele-Tingley parametric simulation |                      -0.101871 |    -0.130248 |    -0.0733907 |                   -0.0664766 |                   -0.168348 |       0.00045867 |    67.2   |
| pingouin.mediation_analysis (n_boot=1000)              | Preacher-Hayes percentile bootstrap      |                      -0.102146 |    -0.114979 |    -0.088276  |                   -0.0665095 |                  nan        |     nan          |     3.79  |

Difference in the estimated indirect effect versus this package:

- statsmodels OLS (formula API), product of coefficients: absolute difference 2.51e-15 h (0.00% of the point estimate).
- statsmodels.stats.mediation.Mediation (n_rep=1000): absolute difference 2.75e-04 h (0.27% of the point estimate). That is 0.6 Monte Carlo standard errors of its own simulation noise.
- pingouin.mediation_analysis (n_boot=1000): absolute difference 1.68e-15 h (0.00% of the point estimate).

The closed-form statsmodels fit agrees to machine precision: the two implementations solve the same least-squares problem by different routes, which is the numerical check. The Imai-Keele-Tingley estimator draws the potential mediator at random in every replicate, so it agrees only up to its own Monte Carlo noise, and its interval is wider for that reason rather than because the effect is less certain.
