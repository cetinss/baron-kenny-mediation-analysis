# Sensitivity Analysis 2: Outlier Threshold & Robust Regression

**Purpose.** Reviewer 2 noted that excluding observations on both the exposure and outcome side using the 1.5xIQR rule could induce selection bias, and asked for the analysis to be repeated on raw (uncleaned) data, with a different threshold, and with robust regression.

## Outlier threshold comparison (OLS, classic vs. HC3-robust SE)

| Sample                        |    n |   c (total, OLS) |   c_p (OLS) |   c_p (HC3-robust SE) |        a |      a_p |         b |   b_p (OLS) |   b_p (HC3-robust SE) |   Indirect (a*b) |   95% CI lo |   95% CI hi | CI excl. 0   |   % mediated |
|:------------------------------|-----:|-----------------:|------------:|----------------------:|---------:|---------:|----------:|------------:|----------------------:|-----------------:|------------:|------------:|:-------------|-------------:|
| Raw (no outlier exclusion)    | 9946 |        -0.001682 |    0.000000 |              0.000000 | 0.002142 | 0.000000 | -0.484860 |    0.000000 |              0.000000 |        -0.001038 |   -0.001168 |   -0.000904 | True         |    61.752772 |
| 1.5xIQR (manuscript baseline) | 9795 |        -0.001687 |    0.000000 |              0.000000 | 0.002121 | 0.000000 | -0.481648 |    0.000000 |              0.000000 |        -0.001021 |   -0.001155 |   -0.000880 | True         |    60.564047 |
| 3xIQR (permissive)            | 9946 |        -0.001682 |    0.000000 |              0.000000 | 0.002142 | 0.000000 | -0.484860 |    0.000000 |              0.000000 |        -0.001038 |   -0.001174 |   -0.000902 | True         |    61.752772 |

## Huber M-estimation (RLM) at the manuscript baseline (1.5xIQR, n=9,795)
Huber M-estimation downweights high-residual observations during estimation itself, rather than only adjusting standard errors, and is therefore a stronger check on whether a small number of extreme points drive the reported effects.

- Path a (Caffeine -> Stress): a = 0.001850, p = <0.001 (***)
- Path b (Stress -> Sleep | Caffeine): b = -0.461857, p = <0.001 (***)
- Direct effect c': -0.000599, p = <0.001 (***)
- Indirect effect (a*b): -0.000854, 95% CI [-0.000960, -0.000154] (excludes 0)
- Proportion mediated: 48.9%

## Conclusion
The sign, statistical significance, and approximate magnitude of paths a, b, c and c' are stable across raw data, the 1.5xIQR manuscript sample, and a more permissive 3xIQR sample, and are unchanged under heteroskedasticity-robust (HC3) standard errors and Huber M-estimation. This indicates the reported mediation pattern is not an artefact of the specific outlier-exclusion rule or of a small number of high-leverage observations.
