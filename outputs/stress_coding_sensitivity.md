# Sensitivity Analysis 1: Stress-Coding Scheme

**Purpose.** Reviewer 2 asked whether the arbitrary equal-interval coding of the mediator (Low=2, Medium=5, High=8) was justified, and requested that alternative codings, indicator variables, or an ordinal mediator model be used to test whether conclusions depend on this modelling choice.

**Mathematical note.** Because the indirect effect is a product of two linear-regression coefficients (a x b), it is invariant to any purely affine rescaling of the mediator (a linear transform of M rescales a and b reciprocally, leaving a*b unchanged). Equal-interval recodings such as 1/2/3 therefore cannot, by construction, change the indirect effect and are included only to make this invariance explicit. The informative tests are the **non-equal-interval** codings (which assume a different functional form for the ordinal-to-numeric mapping) and the **indicator-variable** coding (which assumes no numeric spacing at all).

## Equal-interval and non-equal-interval codings

| Scheme                                      |   a (X->M) |      a_p |   b (M->Y|X) |      b_p |   Indirect (a*b) |   95% CI lo |   95% CI hi | CI excl. 0   |   % mediated |
|:--------------------------------------------|-----------:|---------:|-------------:|---------:|-----------------:|------------:|------------:|:-------------|-------------:|
| Baseline equal-interval (2/5/8)             |   0.002121 | 0.000000 |    -0.481648 | 0.000000 |        -0.001021 |   -0.001149 |   -0.000884 | True         |    60.564898 |
| Equal-interval, unit scale (1/2/3)          |   0.000707 | 0.000000 |    -1.444944 | 0.000000 |        -0.001021 |   -0.001149 |   -0.000884 | True         |    60.564898 |
| Non-equal interval, compressed-low (1/3/9)  |   0.002268 | 0.000000 |    -0.369703 | 0.000000 |        -0.000838 |   -0.000957 |   -0.000718 | True         |    49.705395 |
| Non-equal interval, compressed-high (1/7/9) |   0.003388 | 0.000000 |    -0.305586 | 0.000000 |        -0.001035 |   -0.001165 |   -0.000902 | True         |    61.384164 |

## Indicator (dummy) coding - Low as reference
No numeric spacing assumed; Medium and High enter as separate indicator mediators in a linear-probability path-a model and a multiple-mediator path-b model.

- Path a (Caffeine -> P(Medium)): a = 0.000280, p = <0.001 (***)
- Path a (Caffeine -> P(High)):   a = 0.000213, p = <0.001 (***)
- Path b (Medium -> Sleep | X, High): b = -1.646164, p = <0.001 (***)
- Path b (High -> Sleep | X, Medium): b = -2.722945, p = <0.001 (***)
- Indirect effect via Medium: -0.000461, 95% CI [-0.000553, -0.000364] (excludes 0)
- Indirect effect via High:   -0.000581, 95% CI [-0.000695, -0.000464] (excludes 0)

## Native-ordinal path-a model (proportional-odds logit)
Treating the mediator in its native ordinal form (no numeric spacing assumed at all): caffeine intake predicts higher perceived-stress category with a proportional-odds coefficient of 0.002387 (p = <0.001, ***), consistent in sign and significance with the linear path-a estimate under every numeric coding tested above.

## Conclusion
Across four numeric codings and a fully coding-free indicator-variable specification, path a remains positive and significant, path b remains negative and significant, and the indirect effect remains negative with a bootstrap CI excluding zero. The conclusion that perceived stress partially mediates the caffeine-sleep association is therefore **not an artefact of the specific 2/5/8 numeric coding**.
