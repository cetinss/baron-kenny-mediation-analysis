# Sensitivity Analysis 3: Mediator-Outcome Confounding & Reverse Causation

**Purpose.** Reviewer 2 noted that the bootstrap CI for the indirect effect does not rule out unmeasured confounding or reverse causality, and asked for an alternative directional model and a confounding sensitivity analysis.

## (a) Reverse-ordering alternative model: Caffeine -> Sleep Duration -> Stress

- Path a' (Caffeine -> Sleep): a' = -0.001687, p = <0.001 (***)
- Path b' (Sleep -> Stress | Caffeine): b' = -1.275778, p = <0.001 (***)
- Indirect effect (a'*b'): 0.002152, 95% CI [0.001925, 0.002368] (excludes 0)

This reverse ordering is **also** statistically consistent with the data (as it must be in a single cross-sectional wave, since the correlational building blocks are shared by both orderings). This is reported not as evidence against the hypothesized Caffeine -> Stress -> Sleep pathway, but as an explicit, honest acknowledgment that **directionality is assumed from theory, not identified by the cross-sectional design**, and is now stated as an explicit limitation in the manuscript.

## (b) Unmeasured mediator-outcome confounder sensitivity (Imai, Keele & Yamamoto, 2010)

Residual SD of the path-a model (sigma_v): 1.9313

Residual SD of the outcome model (sigma_xi): 0.7368

**Sensitivity parameter rho\* = -1.2625**: an unmeasured confounder of the mediator (Stress_Score) and outcome (Sleep_Hours) relationship would have to induce a residual correlation of approximately 1.26 in magnitude, between the path-a and outcome-model residuals, to drive the indirect effect exactly to zero.

Because a correlation coefficient is mathematically bounded in [-1, 1], no unmeasured mediator-outcome confounder operating solely through this residual-correlation channel could fully explain away the indirect effect reported here, even at the theoretical maximum |rho| = 1: the requirement of 1.26 lies outside the admissible range. This is a bounded and therefore comparatively strong form of robustness evidence.

Two caveats apply in either case. First, this is a single-parameter model: it assumes a linear, additive confounding channel operating through the correlation of the two error terms, and says nothing about confounding that acts through a different functional form, through the exposure-mediator relationship, or through several weak confounders acting together. Second, it is a sensitivity statement about the estimated model, not a claim that no confounding exists.

### Bias-adjusted indirect effect across a grid of assumed confounder strengths (rho)

|       rho |    b(rho) |   Indirect effect a*b(rho) |
|----------:|----------:|---------------------------:|
| -0.500000 | -0.290894 |                  -0.000617 |
| -0.450000 | -0.309970 |                  -0.000657 |
| -0.400000 | -0.329045 |                  -0.000698 |
| -0.350000 | -0.348120 |                  -0.000738 |
| -0.300000 | -0.367196 |                  -0.000779 |
| -0.250000 | -0.386271 |                  -0.000819 |
| -0.200000 | -0.405347 |                  -0.000860 |
| -0.150000 | -0.424422 |                  -0.000900 |
| -0.100000 | -0.443497 |                  -0.000941 |
| -0.050000 | -0.462573 |                  -0.000981 |
| -0.000000 | -0.481648 |                  -0.001021 |
|  0.050000 | -0.500723 |                  -0.001062 |
|  0.100000 | -0.519799 |                  -0.001102 |
|  0.150000 | -0.538874 |                  -0.001143 |
|  0.200000 | -0.557950 |                  -0.001183 |
|  0.250000 | -0.577025 |                  -0.001224 |
|  0.300000 | -0.596100 |                  -0.001264 |
|  0.350000 | -0.615176 |                  -0.001305 |
|  0.400000 | -0.634251 |                  -0.001345 |
|  0.450000 | -0.653326 |                  -0.001386 |
|  0.500000 | -0.672402 |                  -0.001426 |

## Conclusion
Cross-sectional mediation statistics alone cannot adjudicate between the hypothesized causal ordering and its reverse, and cannot rule out an unmeasured mediator-outcome confounder in principle. The magnitude of confounding required to nullify the present indirect effect can, however, be quantified (rho* above) and reported so that readers can judge plausibility for themselves; both points are now made explicit in the Discussion / Limitations.
