# Gender-coding correction

The original pipeline coded gender as `Male = 1`, which placed the 221 respondents who reported "Other" in the same category as women. The corrected coding treats Female as the reference category and gives Male and Other their own indicators. The corrected coding is now the default in `bkmediation`; the table below reports both specifications so the effect of the correction is visible.

| Gender coding   | Description                                                     |    n |          a |         b |           c |           c' |         a*b |   a*b CI low |   a*b CI high |   Prop. mediated (%) | Conclusion        |
|:----------------|:----------------------------------------------------------------|-----:|-----------:|----------:|------------:|-------------:|------------:|-------------:|--------------:|---------------------:|:------------------|
| three_level     | Corrected: Female reference, separate Male and Other indicators | 9795 | 0.00212076 | -0.481648 | -0.00168656 | -0.000665095 | -0.00102146 |  -0.00115687 |  -0.000883033 |              60.5649 | PARTIAL MEDIATION |
| binary_male     | Original: Male = 1, Female and Other = 0                        | 9795 | 0.00212076 | -0.481648 | -0.00168658 | -0.000665119 | -0.00102146 |  -0.00115688 |  -0.000883137 |              60.564  | PARTIAL MEDIATION |

The two specifications differ in the indirect effect by 8.24e-11 (all four paths agree to six decimal places), and both yield partial mediation. The correction is therefore one of construct validity and of respect for how respondents described themselves, not one that changes any substantive conclusion.
