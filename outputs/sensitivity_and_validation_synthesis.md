# Sensitivity & External Validation - Synthesis

This note integrates the three new sensitivity analyses (stress-coding, outlier/robust regression, confounding/reverse-causation) with the pre-existing external-validation work (LEMURS, NHANES) so that the manuscript's new Methods/Results/Discussion subsections can draw on a single source of truth. Nothing here changes the primary analysis (Table 2 of the manuscript); every check below was run **in addition to**, not instead of, the pre-registered Baron-Kenny model.

## 1. Stress-coding sensitivity (this work)

Direction and significance of a, b, and the indirect effect are unchanged across 4 numeric codings, a fully coding-free indicator-variable specification, and (where available) a native-ordinal path-a model. See `stress_coding_sensitivity.md`.

## 2. Outlier-threshold / robust-regression sensitivity (this work)

Direction and significance of a, b, c, c' are unchanged across raw/1.5xIQR/3xIQR samples, HC3-robust SEs, and Huber M-estimation. See `outlier_robustness_sensitivity.md`.

## 3. Confounding & reverse-causation sensitivity (this work)

A reverse-ordering model (Caffeine -> Sleep -> Stress) is also statistically consistent with the data (indirect effect 0.002152, CI [0.001936, 0.002371]) -- directionality is assumed from theory, not identified by the design. An unmeasured confounder would need residual correlation |rho| >= 1.26 with both the mediator- and outcome-model residuals to fully explain away the indirect effect. See `confounding_sensitivity.md`.

## 4. External validation against independent datasets (pre-existing work)

- `comparison_summary.md` (found): LEMURS external validation - direction/significance comparison
- `limitations.md` (found): LEMURS external validation - full limitations register
- `nhanes_mediation_results.md` (found): NHANES real-data attempt 1 (PHQ-9 mediator)
- `nhanes_mediation_AL_results.md` (found): NHANES real-data attempt 2 (Allostatic Load mediator)
- `distributional_realism_check.md` (found): Synthetic-vs-NHANES distributional plausibility check

**Headline external-validation result:** across LEMURS (independent DP-synthetic dataset, real university-student sample, ~600 participants) and NHANES (real US national health survey), the direction of caffeine -> stress and stress -> sleep is replicated (5/5 paths directionally consistent in LEMURS); NHANES confirms a significant total effect of caffeine on sleep in real national data but neither depressive symptoms (PHQ-9) nor an allostatic-load index mediate it, indicating the mediation finding is probably specific to a perceived-stress construct rather than a general property of any stress-adjacent variable. This boundary condition is now stated explicitly in the Discussion.
