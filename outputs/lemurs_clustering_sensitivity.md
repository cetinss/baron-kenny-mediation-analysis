# LEMURS clustering sensitivity

The LEMURS analytic sample is **n = 2,953 person-week rows** from approximately 600 participants, a mean of **4.9 weekly observations per person**. The released file contains no participant identifier, so those rows cannot be grouped: `src/lemurs_validation.py` necessarily treats them as independent observations, which understates the standard errors whenever repeated responses from the same person are correlated.

This report does not correct that. It asks the answerable question instead: **how strong would the within-person correlation have to be before each association stopped being significant?** Using the classical design effect DEFF = 1 + (m - 1) x rho, the table gives the intra-class correlation (ICC) at which each 95% interval would first include zero.

| Quantity | Estimate | SE (unclustered) | z | Critical DEFF | Critical ICC |
|---|---:|---:|---:|---:|---:|
| Path a (caffeine exposure -> PSS-10) | 0.628975 | 0.118980 | 5.29 | 7.27 | >1 (unattainable) |
| Path b (PSS-10 -> sleep duration | caffeine) | -0.010452 | 0.002559 | -4.08 | 4.34 | 0.85 |
| Total effect c (caffeine -> sleep duration) | -0.028192 | 0.016580 | -1.70 | n/a | n/a |
| Indirect effect a*b | -0.006574 | 0.002080 | -3.16 | 2.60 | 0.41 |

## What this means

* Paths a and b are robust to clustering of any plausible magnitude: the within-person correlation needed to overturn them is at or beyond the upper limit of what an ICC can be.
* The indirect effect is the vulnerable claim. Treating the rows as independent gives |z| = 3.16. At a mean cluster size of 4.9, an intra-class correlation of rho = 0.41 (design effect 2.60) would be enough to widen the interval to include zero. Weekly measurements of perceived stress and sleep duration within the same person are routinely reported with ICCs in that range, so the statistical significance of the LEMURS indirect effect should be treated as **not established**, and the LEMURS comparison should be read as directional replication of the paths rather than as an independent significant confirmation of the mediated pathway.
* The total effect c was already non-significant in LEMURS, so clustering does not change its interpretation.

## Caveats

This is a bound on one failure mode, not a correction. It assumes equal cluster sizes and a single exchangeable within-person correlation, uses a normal-theory interval for the design-effect adjustment, and cannot address bias in the point estimates themselves if the clustering is informative. Obtaining the participant identifiers and refitting with a cluster bootstrap remains the correct fix and is recorded as future work.
