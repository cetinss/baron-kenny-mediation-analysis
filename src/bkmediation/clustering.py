"""How much clustering would it take to overturn a result?

The LEMURS external validation analyses person-week rows: roughly 600 students
each contributing several weekly surveys. The released file carries no
participant identifier, so the rows cannot be grouped and a cluster-robust or
cluster-bootstrap standard error cannot be computed (documented in
`outputs/limitations.md`). Treating the rows as independent, as that analysis
must, understates the standard errors whenever responses from the same person
are correlated - which for weekly perceived stress and sleep duration they
certainly are.

What can still be said precisely is how much clustering it would take to
overturn the conclusion. With mean cluster size m and intra-class correlation
rho, the variance inflation of a mean-like estimator is the classical design
effect

    DEFF = 1 + (m - 1) * rho

so the clustering-adjusted standard error is SE * sqrt(DEFF). Setting the
adjusted z statistic to the critical value gives the design effect, and hence
the ICC, at which the interval would first include zero. That number is
reportable, falsifiable and does not require the missing identifier.

This is a bound on one specific failure mode, not a correction: it assumes
equal cluster sizes and a single exchangeable correlation, and it cannot
recover the point estimate's own bias if the clustering is informative.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats

__all__ = ["DesignEffectSensitivity", "design_effect_sensitivity", "se_from_ci"]


def se_from_ci(ci_lo: float, ci_hi: float, alpha: float = 0.05) -> float:
    """Back out a standard error from a symmetric normal-theory interval."""
    if ci_hi < ci_lo:
        raise ValueError("ci_hi must not be below ci_lo")
    z = stats.norm.ppf(1 - alpha / 2)
    return (ci_hi - ci_lo) / (2 * z)


@dataclass
class DesignEffectSensitivity:
    estimate: float
    se_unclustered: float
    z_unclustered: float
    mean_cluster_size: float
    critical_design_effect: float
    critical_icc: float
    already_nonsignificant: bool

    def describe(self) -> str:
        if self.already_nonsignificant:
            return (
                f"The unclustered estimate is already non-significant "
                f"(|z| = {abs(self.z_unclustered):.2f}), so no amount of "
                f"clustering is needed to overturn it."
            )
        if self.critical_icc >= 1:
            return (
                f"Even a perfect intra-class correlation (rho = 1) at a mean "
                f"cluster size of {self.mean_cluster_size:.1f} would not make "
                f"this estimate non-significant "
                f"(|z| = {abs(self.z_unclustered):.2f}); the finding is robust "
                f"to clustering of this magnitude."
            )
        return (
            f"Treating the rows as independent gives |z| = "
            f"{abs(self.z_unclustered):.2f}. At a mean cluster size of "
            f"{self.mean_cluster_size:.1f}, an intra-class correlation of "
            f"rho = {self.critical_icc:.2f} (design effect "
            f"{self.critical_design_effect:.2f}) would be enough to widen the "
            f"interval to include zero."
        )


def design_effect_sensitivity(
    estimate: float,
    se: float | None = None,
    ci: tuple | None = None,
    mean_cluster_size: float = 1.0,
    alpha: float = 0.05,
) -> DesignEffectSensitivity:
    """ICC at which a clustering-adjusted interval would first include zero.

    Give either `se` or a normal-theory `ci` tuple. `mean_cluster_size` is the
    number of observations per cluster (for LEMURS: person-weeks per student).
    """
    if se is None:
        if ci is None:
            raise ValueError("provide either se or ci")
        se = se_from_ci(ci[0], ci[1], alpha)
    if se <= 0:
        raise ValueError("se must be positive")
    if mean_cluster_size < 1:
        raise ValueError("mean_cluster_size must be at least 1")

    z_crit = stats.norm.ppf(1 - alpha / 2)
    z = estimate / se
    already_ns = abs(z) <= z_crit

    if already_ns:
        deff_crit, icc_crit = 1.0, 0.0
    else:
        deff_crit = (z / z_crit) ** 2
        icc_crit = (
            float("inf") if mean_cluster_size == 1
            else (deff_crit - 1) / (mean_cluster_size - 1)
        )

    return DesignEffectSensitivity(
        estimate=estimate,
        se_unclustered=se,
        z_unclustered=z,
        mean_cluster_size=mean_cluster_size,
        critical_design_effect=deff_crit,
        critical_icc=icc_crit,
        already_nonsignificant=already_ns,
    )
