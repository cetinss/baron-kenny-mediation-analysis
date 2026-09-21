"""Formal evaluation of the three study hypotheses (H1-H3).

The hypotheses are stated here in the code as well as in the manuscript, so
that each one has a single, auditable decision rule and the text and the
analysis cannot drift apart.

    H1  Higher daily caffeine intake is associated with SHORTER sleep
        duration.
        Decision rule: total effect c < 0 and the 95% interval for c
        excludes zero.

    H2  Higher daily caffeine intake is associated with HIGHER perceived
        stress.
        Decision rule: path a > 0 and the 95% interval for a excludes zero.

    H3  Perceived stress mediates the association between caffeine intake
        and sleep duration.
        Decision rule: the percentile-bootstrap 95% interval for the
        indirect effect a*b excludes zero, in the hypothesised (negative)
        direction. This is the primary inferential test of the study.

Wording note: "associated with" is used deliberately. These are
associational hypotheses about a cross-sectional (and, for the primary
dataset, synthetic) data-generating process; supporting H3 does not establish
that caffeine changes sleep duration by way of stress.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ALPHA

__all__ = ["HYPOTHESES", "Hypothesis", "HypothesisOutcome", "evaluate_hypotheses"]


@dataclass(frozen=True)
class Hypothesis:
    key: str
    statement: str
    quantity: str          # 'c', 'a' or 'ab'
    expected_sign: int     # +1 or -1
    rule: str


HYPOTHESES = (
    Hypothesis(
        key="H1",
        statement="Higher daily caffeine intake is associated with shorter sleep duration.",
        quantity="c",
        expected_sign=-1,
        rule="Total effect c is negative and its 95% CI excludes zero.",
    ),
    Hypothesis(
        key="H2",
        statement="Higher daily caffeine intake is associated with higher perceived stress.",
        quantity="a",
        expected_sign=+1,
        rule="Path a is positive and its 95% CI excludes zero.",
    ),
    Hypothesis(
        key="H3",
        statement=("Perceived stress mediates the association between caffeine intake "
                   "and sleep duration."),
        quantity="ab",
        expected_sign=-1,
        rule=("Percentile-bootstrap 95% CI for the indirect effect a*b excludes zero "
              "and the effect is negative."),
    ),
)


@dataclass
class HypothesisOutcome:
    key: str
    statement: str
    quantity: str
    estimate: float
    ci_lo: float
    ci_hi: float
    ci_method: str
    p_value: float
    expected_sign: int
    supported: bool
    rule: str

    @property
    def decision(self) -> str:
        return "SUPPORTED" if self.supported else "NOT SUPPORTED"


def _boot_two_sided_p(draws) -> float:
    """Bootstrap p value: twice the smaller tail mass on either side of zero.

    Floored at 1/B, since a bootstrap cannot resolve p below its resolution.
    """
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return float("nan")
    tail = min((draws <= 0).mean(), (draws >= 0).mean())
    return float(max(2 * tail, 1.0 / draws.size))


def evaluate_hypotheses(result, alpha: float = ALPHA) -> pd.DataFrame:
    """Return one row per hypothesis with the estimate, interval and decision.

    `result` is a :class:`~bkmediation.mediation.MediationResult`.
    """
    outcomes = []
    for h in HYPOTHESES:
        if h.quantity == "ab":
            est = result.effect_sizes["ab"]
            lo, hi = result.effect_sizes["ab_ci"]
            method = "percentile bootstrap"
            p = _boot_two_sided_p(result.boot["ab"])
        else:
            path = result.paths[h.quantity]
            est = path["coef"]
            lo, hi = path["ci_lo"], path["ci_hi"]
            method = f"model-based ({result.cov_type})"
            p = path["p"]
        excludes_zero = not (lo <= 0 <= hi)
        right_sign = np.sign(est) == h.expected_sign
        outcomes.append(HypothesisOutcome(
            key=h.key, statement=h.statement, quantity=h.quantity,
            estimate=est, ci_lo=lo, ci_hi=hi, ci_method=method, p_value=p,
            expected_sign=h.expected_sign,
            supported=bool(excludes_zero and right_sign),
            rule=h.rule,
        ))

    scale = result.scale
    rows = []
    for o in outcomes:
        k = scale if o.quantity in ("c", "a", "ab") else 1.0
        rows.append({
            "Hypothesis": o.key,
            "Statement": o.statement,
            "Tested quantity": o.quantity if o.quantity != "ab" else "a*b",
            "Expected sign": "+" if o.expected_sign > 0 else "-",
            "Estimate": o.estimate,
            "Estimate (per 100 mg)": o.estimate * k,
            "95% CI low": o.ci_lo,
            "95% CI high": o.ci_hi,
            "95% CI low (per 100 mg)": o.ci_lo * k,
            "95% CI high (per 100 mg)": o.ci_hi * k,
            "CI method": o.ci_method,
            "p": o.p_value,
            "Decision": o.decision,
            "Decision rule": o.rule,
        })
    return pd.DataFrame(rows)
