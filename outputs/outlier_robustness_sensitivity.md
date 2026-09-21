# Sensitivity Analysis 2: Outlier Threshold & Robust Regression

**Purpose.** Excluding observations on both the exposure and the outcome side with the 1.5xIQR rule could induce selection bias, so the analysis is repeated on raw (uncleaned) data, with a different threshold, and with robust regression.

## Outlier threshold comparison (OLS, classic vs. HC3-robust SE)

| Sample                                     |     n |   c (total, OLS) |   c_p (OLS) |   c_p (HC3-robust SE) |        a |      a_p |         b |   b_p (OLS) |   b_p (HC3-robust SE) |   Indirect (a*b) |   95% CI lo |   95% CI hi | CI excl. 0   |   % mediated |
|:-------------------------------------------|------:|-----------------:|------------:|----------------------:|---------:|---------:|----------:|------------:|----------------------:|-----------------:|------------:|------------:|:-------------|-------------:|
| Full raw dataset (n=10,000, no age filter) | 10000 |        -0.001676 |    0.000000 |              0.000000 | 0.002128 | 0.000000 | -0.485028 |    0.000000 |              0.000000 |        -0.001032 |   -0.001163 |   -0.000899 | True         |    61.607184 |
| Raw (18-65, no outlier exclusion)          |  9946 |        -0.001682 |    0.000000 |              0.000000 | 0.002142 | 0.000000 | -0.484878 |    0.000000 |              0.000000 |        -0.001038 |   -0.001182 |   -0.000898 | True         |    61.754370 |
| 1.5xIQR (manuscript baseline)              |  9795 |        -0.001687 |    0.000000 |              0.000000 | 0.002121 | 0.000000 | -0.481648 |    0.000000 |              0.000000 |        -0.001021 |   -0.001149 |   -0.000884 | True         |    60.564898 |
| 3xIQR (permissive)                         |  9946 |        -0.001682 |    0.000000 |              0.000000 | 0.002142 | 0.000000 | -0.484878 |    0.000000 |              0.000000 |        -0.001038 |   -0.001182 |   -0.000898 | True         |    61.754370 |

## Huber M-estimation (RLM) at the manuscript baseline (1.5xIQR, n=9,795)
Huber M-estimation downweights high-residual observations during estimation itself, rather than only adjusting standard errors, and is therefore a stronger check on whether a small number of extreme points drive the reported effects.

**Preferred robust specification (Huber on the outcome models only).** M-estimation assumes a continuous response contaminated by outliers. That holds for sleep duration, but not for path a, whose dependent variable is the 3-level stress score: there the "extreme" values are the High-stress category itself, not contamination, so down-weighting them removes signal rather than noise. Path a is therefore kept at OLS here and Huber is applied to the total-effect and outcome models.

- Path a (OLS, Caffeine -> Stress): a = 0.002121, p = <0.001 (***)
- Path b (Huber, Stress -> Sleep | Caffeine): b = -0.461927, p = <0.001 (***)
- Direct effect c' (Huber): -0.000599, p = <0.001 (***)
- Indirect effect (a*b): -0.000980, 95% CI [-0.001103, -0.000848] (excludes 0)
- Proportion mediated: 56.0%

**Huber applied to all three models, including path a (reported for completeness).**

- Path a (Caffeine -> Stress): a = 0.001850, p = <0.001 (***)
- Path b (Stress -> Sleep | Caffeine): b = -0.461927, p = <0.001 (***)
- Direct effect c': -0.000599, p = <0.001 (***)
- Indirect effect (a*b): -0.000854, 95% CI [-0.000950, -0.000151] (excludes 0)
- Proportion mediated: 48.9%

The wide, strongly asymmetric interval in this second specification comes from path a, not from path b: across bootstrap replicates the Huber estimate of a is left-skewed and reaches values close to zero, while the Huber estimate of b stays within a narrow band. That instability is the artefact of M-estimating a 3-level dependent variable described above, which is why the first specification is the one to read as the robust-regression check.

## Conclusion
The sign, statistical significance, and approximate magnitude of paths a, b, c and c' are stable across raw data, the 1.5xIQR manuscript sample, and a more permissive 3xIQR sample, and are unchanged under heteroskedasticity-robust (HC3) standard errors and Huber M-estimation. This indicates the reported mediation pattern is not an artefact of the specific outlier-exclusion rule or of a small number of high-leverage observations.
