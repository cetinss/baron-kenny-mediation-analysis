# Mediation Analysis Report

Caffeine intake -> perceived stress -> sleep duration. Source: synthetic_coffee_health_10000.csv (SYNTHETIC, n = 9,795).

> **Synthetic data.** The primary dataset is synthetic. Its `Stress_Level` field was generated partly from sleep- and lifestyle-related quantities, so mediator and outcome are not independently measured and the indirect effect reported below is a statistical association inside a simulated data-generating process, not evidence of a biological mechanism. External validation against LEMURS and NHANES is reported separately under `outputs/`.

## 1. Sample

| Step                                                 |   n remaining |   n removed |
|:-----------------------------------------------------|--------------:|------------:|
| Raw records                                          |         10000 |           0 |
| After age filter (18-65 years)                       |          9946 |          54 |
| After listwise deletion on analysis variables        |          9946 |           0 |
| After 1.5x IQR outlier rule: Caffeine_mg             |          9907 |          39 |
| After 1.5x IQR outlier rule: Sleep_Hours             |          9881 |          26 |
| After 1.5x IQR outlier rule: BMI                     |          9843 |          38 |
| After 1.5x IQR outlier rule: Heart_Rate              |          9795 |          48 |
| After 1.5x IQR outlier rule: Physical_Activity_Hours |          9795 |           0 |
| Final analytic sample                                |          9795 |           0 |

Final analytic sample: **n = 9,795**. Covariates: Age, Gender_Male, Gender_Other, BMI, Physical_Activity_Hours, Heart_Rate. Gender coding: `three_level` (Female is the reference category; respondents reporting "Other" have their own indicator and are not pooled with women).

## 2. Descriptive statistics

| Variable                   |    n |   Mean |     SD |   Median |   Min |   Max |   Skewness |   Kurtosis |
|:---------------------------|-----:|-------:|-------:|---------:|------:|------:|-----------:|-----------:|
| Daily Caffeine Intake (mg) | 9795 | 236.45 | 135.12 |    234.8 |   0   | 620.6 |      0.184 |     -0.485 |
| Stress Score (2-8)         | 9795 |   3.17 |   1.95 |      2   |   2   |   8   |      1.415 |      0.707 |
| Sleep Duration (hours)     | 9795 |   6.65 |   1.21 |      6.6 |   3.3 |  10   |      0.027 |     -0.211 |
| Age (years)                | 9795 |  34.77 |  10.92 |     34   |  18   |  65   |      0.268 |     -0.616 |
| BMI (kg/m2)                | 9795 |  23.93 |   3.84 |     24   |  15   |  34.5 |     -0.031 |     -0.337 |
| Physical Activity (h/week) | 9795 |   7.49 |   4.32 |      7.5 |   0   |  15   |      0.002 |     -1.201 |
| Resting Heart Rate (bpm)   | 9795 |  70.48 |   9.64 |     70   |  50   |  96   |      0.019 |     -0.444 |

Gender:

| Category   |    n |    % |
|:-----------|-----:|-----:|
| Female     | 4897 | 50   |
| Male       | 4677 | 47.7 |
| Other      |  221 |  2.3 |

Stress level:

| Category   |    n |    % |
|:-----------|-----:|-----:|
| Low        | 6868 | 70.1 |
| Medium     | 2018 | 20.6 |
| High       |  909 |  9.3 |

## 3. Bivariate correlations

| Pair                        |       r |           p |    n |
|:----------------------------|--------:|------------:|-----:|
| Caffeine <-> Stress         |  0.1479 | 4.71767e-49 | 9795 |
| Caffeine <-> Sleep duration | -0.1902 | 2.03141e-80 | 9795 |
| Stress <-> Sleep duration   | -0.7893 | 0           | 9795 |

## 4. Hypothesis tests

Estimates and intervals are per 100 mg of caffeine per day. H1 and H2 use the HC3 model-based interval; H3 uses the percentile-bootstrap interval for the indirect effect.

| Hypothesis   | Statement                                                                             | Tested quantity   | Expected sign   |   Estimate (per 100 mg) |   95% CI low (per 100 mg) |   95% CI high (per 100 mg) | CI method            | p      | Decision   |
|:-------------|:--------------------------------------------------------------------------------------|:------------------|:----------------|------------------------:|--------------------------:|---------------------------:|:---------------------|:-------|:-----------|
| H1           | Higher daily caffeine intake is associated with shorter sleep duration.               | c                 | -               |                 -0.1687 |                   -0.186  |                    -0.1513 | model-based (HC3)    | <0.001 | SUPPORTED  |
| H2           | Higher daily caffeine intake is associated with higher perceived stress.              | a                 | +               |                  0.2121 |                    0.1833 |                     0.2408 | model-based (HC3)    | <0.001 | SUPPORTED  |
| H3           | Perceived stress mediates the association between caffeine intake and sleep duration. | a*b               | -               |                 -0.1021 |                   -0.1157 |                    -0.0883 | percentile bootstrap | <0.001 | SUPPORTED  |

## 5. Path estimates (manuscript Table 2)

Standard errors are heteroskedasticity-consistent (HC3). Rescaled coefficients for the exposure paths are per 100 mg caffeine/day; path b is per one point of the stress scale.

| Path   | Description                     |      Coef |   SE (HC3) |       t | p      |   95% CI low |   95% CI high |   Coef (rescaled) |   CI low (rescaled) |   CI high (rescaled) | Scale            |
|:-------|:--------------------------------|----------:|-----------:|--------:|:-------|-------------:|--------------:|------------------:|--------------------:|---------------------:|:-----------------|
| a      | a  (Caffeine -> Stress)         |  0.002121 |   0.000147 |   14.46 | <0.001 |     0.001833 |      0.002408 |            0.2121 |              0.1833 |               0.2408 | per 100 mg/day   |
| b      | b  (Stress -> Sleep | Caffeine) | -0.481648 |   0.00293  | -164.36 | <0.001 |    -0.487391 |     -0.475905 |           -0.4816 |             -0.4874 |              -0.4759 | per stress point |
| c      | c  (Caffeine -> Sleep, total)   | -0.001687 |   8.8e-05  |  -19.08 | <0.001 |    -0.00186  |     -0.001513 |           -0.1687 |             -0.186  |              -0.1513 | per 100 mg/day   |
| c'     | c' (Caffeine -> Sleep, direct)  | -0.000665 |   5.5e-05  |  -12.15 | <0.001 |    -0.000772 |     -0.000558 |           -0.0665 |             -0.0772 |              -0.0558 | per 100 mg/day   |

Classical (non-robust) intervals, for comparison:

| Path   |   SE (classical) |   95% CI low (classical) |   95% CI high (classical) |
|:-------|-----------------:|-------------------------:|--------------------------:|
| a      |      0.000144773 |              0.00183698  |               0.00240455  |
| b      |      0.00385656  |             -0.489208    |              -0.474088    |
| c      |      8.89537e-05 |             -0.00186092  |              -0.00151219  |
| c'     |      5.58369e-05 |             -0.000774547 |              -0.000555643 |

Percentile-bootstrap intervals for every path:

| Path   |   95% CI low (bootstrap) |   95% CI high (bootstrap) |
|:-------|-------------------------:|--------------------------:|
| a      |              0.00183316  |               0.00240267  |
| b      |             -0.487327    |              -0.476059    |
| c      |             -0.00185809  |              -0.00151072  |
| c'     |             -0.000770006 |              -0.000553516 |

## 6. Indirect, direct and total effects

| Effect                            |   Estimate | 95% CI low   | 95% CI high   | CI method            |
|:----------------------------------|-----------:|:-------------|:--------------|:---------------------|
| Indirect (a*b)                    |  -0.001021 | -0.001157    | -0.000883     | percentile bootstrap |
| Direct (c')                       |  -0.000665 | -0.000770    | -0.000554     | percentile bootstrap |
| Total (c)                         |  -0.001687 | -0.001858    | -0.001511     | percentile bootstrap |
| Indirect, per 100 mg              |  -0.102146 | -0.115687    | -0.088303     | percentile bootstrap |
| Indirect, completely standardised |  -0.114128 | n/a          | n/a           | point estimate       |
| Proportion mediated (%)           |  60.5649   | 55.511732    | 65.692373     | percentile bootstrap |

Primary inference: the percentile-bootstrap interval for a*b (5,000 resamples, seed 42). The Sobel test (z = -14.403, p = <0.001) is reported only as a normal-theory comparison.

Cohen's f2 = 1.5939 is the increment in explained variance when the mediator enters the outcome model (delta R2 = 0.5916). It is **not** a mediation effect size and is not interpreted as one; the mediation effect sizes are the standardised indirect effect (-0.1141) and the proportion mediated (60.6%, 95% CI [55.5%, 65.7%]).

## 7. Baron-Kenny conditions

| Condition             | Met   |
|:----------------------|:------|
| c_significant         | yes   |
| a_significant         | yes   |
| b_significant         | yes   |
| direct_effect_reduced | yes   |

Conclusion: **PARTIAL MEDIATION** (statistical indirect association under the assumed ordering, not an identified causal effect).

## 8. Assumption diagnostics

| Variable                |   VIF |
|:------------------------|------:|
| Caffeine_mg             |  1.03 |
| Stress_Score            |  1.02 |
| Age                     |  1    |
| Gender_Male             |  1.02 |
| Gender_Other            |  1.02 |
| BMI                     |  1    |
| Physical_Activity_Hours |  1    |
| Heart_Rate              |  1    |

| Model                    |   durbin_watson |   breusch_pagan_lm |   breusch_pagan_p | homoskedastic_at_05   |   jarque_bera |   jarque_bera_p |   residual_skew |   residual_kurtosis |        r2 |
|:-------------------------|----------------:|-------------------:|------------------:|:----------------------|--------------:|----------------:|----------------:|--------------------:|----------:|
| Total effect (Y ~ X + C) |         1.95993 |            6.78376 |      0.451737     | True                  |       11.3108 |      0.00349851 |       0.0129817 |             2.83556 | 0.037196  |
| Mediated (Y ~ X + M + C) |         2.01148 |          651.385   |      2.07897e-135 | False                 |     1474.51   |      0          |       0.838542  |             3.89458 | 0.628816  |
| Mediator (M ~ X + C)     |         1.97038 |          117.294   |      2.80191e-22  | False                 |     3320      |      0          |       1.37991   |             3.71983 | 0.0223184 |

Breusch-Pagan rejects homoskedasticity in the mediated model (p = <0.001), which is why HC3 standard errors are the reported ones.
