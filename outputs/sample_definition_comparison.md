# Sample definition sensitivity: the full raw dataset (n = 10,000)

The dataset is also analysed in its raw state, with the full n = 10,000 retained. The model below is re-estimated under four nested sample definitions, changing one exclusion rule at a time, so the cost of each rule is visible rather than argued for. Nothing else differs: same variables, same covariates, same estimator, same seed and same number of bootstrap resamples.

| Sample       | Definition                                                  |     n |   Excluded from raw |          a |         b |           c |           c' |         a*b |   a*b CI low |   a*b CI high |   Prop. mediated (%) |   Prop. CI low (%) |   Prop. CI high (%) | Conclusion        |
|:-------------|:------------------------------------------------------------|------:|--------------------:|-----------:|----------:|------------:|-------------:|------------:|-------------:|--------------:|---------------------:|-------------------:|--------------------:|:------------------|
| full_raw     | Full raw dataset: every row, no age filter, no outlier rule | 10000 |                   0 | 0.00212841 | -0.485028 | -0.00167568 | -0.000643341 | -0.00103234 |  -0.00116863 |  -0.000899079 |              61.6072 |            56.7139 |             66.4971 | PARTIAL MEDIATION |
| age_filtered | Adults 18-65 only, no outlier rule                          |  9946 |                  54 | 0.00214177 | -0.484878 | -0.00168166 | -0.000643162 | -0.0010385  |  -0.00118365 |  -0.000898302 |              61.7544 |            56.5602 |             66.7191 | PARTIAL MEDIATION |
| primary      | Primary analytic sample: adults 18-65, 1.5x IQR rule        |  9795 |                 205 | 0.00212076 | -0.481648 | -0.00168656 | -0.000665095 | -0.00102146 |  -0.00115687 |  -0.000883033 |              60.5649 |            55.5117 |             65.6924 | PARTIAL MEDIATION |
| permissive   | Adults 18-65, permissive 3x IQR rule                        |  9946 |                  54 | 0.00214177 | -0.484878 | -0.00168166 | -0.000643162 | -0.0010385  |  -0.00118365 |  -0.000898302 |              61.7544 |            56.5602 |             66.7191 | PARTIAL MEDIATION |

## Reading the table

* The raw file contains no missing values on any analysis variable, so the full raw specification retains all **n = 10,000** rows. The difference between it and the primary sample is therefore entirely the age filter and the outlier rule, not missing data.
* On the untouched data the indirect effect is **-0.001032** (95% CI [-0.001169, -0.000899]), with **61.6%** mediated; on the primary sample it is -0.001021 (95% CI [-0.001157, -0.000883]) with 60.6% mediated. Every path keeps its sign and significance.
* All four specifications yield the same conclusion (partial mediation).

The cleaning rules therefore change the estimates in the third decimal place and change no conclusion. Reporting the cleaned sample is a presentational choice, not a result-producing one.
