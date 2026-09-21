"""Regression assumption diagnostics for the mediation models.

Multicollinearity (VIF), residual independence (Durbin-Watson),
homoskedasticity (Breusch-Pagan) and residual normality (skew/kurtosis,
Jarque-Bera). The Breusch-Pagan test is the reason HC3 standard errors are
the default in `mediation.py`: if it rejects, classical OLS intervals are the
wrong ones to report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from .config import EXPOSURE, MEDIATOR, OUTCOME

__all__ = ["vif_table", "residual_diagnostics", "run_diagnostics"]


def vif_table(df: pd.DataFrame, predictors) -> pd.DataFrame:
    """Variance inflation factors for the predictors of the mediated model."""
    X = sm.add_constant(df[list(predictors)].astype(float))
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        rows.append({"Variable": col,
                     "VIF": round(float(variance_inflation_factor(X.to_numpy(), i)), 2)})
    return pd.DataFrame(rows)


def residual_diagnostics(y, X) -> dict:
    """Durbin-Watson, Breusch-Pagan and Jarque-Bera on one fitted model."""
    y = np.asarray(y, float)
    X = sm.add_constant(np.asarray(X, float))
    fit = sm.OLS(y, X).fit()
    resid = fit.resid
    bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(resid, X)
    jb, jb_p, skew, kurt = jarque_bera(resid)
    return {
        "durbin_watson": float(durbin_watson(resid)),
        "breusch_pagan_lm": float(bp_lm),
        "breusch_pagan_p": float(bp_p),
        "homoskedastic_at_05": bool(bp_p >= 0.05),
        "jarque_bera": float(jb),
        "jarque_bera_p": float(jb_p),
        "residual_skew": float(skew),
        "residual_kurtosis": float(kurt),
        "r2": float(fit.rsquared),
    }


def run_diagnostics(sample, covariates=None) -> dict:
    """Diagnostics for both the total-effect and the mediated model."""
    df = getattr(sample, "df", sample)
    covariates = list(covariates or getattr(sample, "covariates", []))

    total_predictors = [EXPOSURE, *covariates]
    med_predictors = [EXPOSURE, MEDIATOR, *covariates]

    return {
        "vif": vif_table(df, med_predictors),
        "total_model": residual_diagnostics(df[OUTCOME], df[total_predictors]),
        "mediated_model": residual_diagnostics(df[OUTCOME], df[med_predictors]),
        "mediator_model": residual_diagnostics(df[MEDIATOR], df[total_predictors]),
    }
