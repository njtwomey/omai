"""Penalised IRLS at fixed smoothing parameters against statsmodels' GLMGam: coefficients, the leverage of each
observation (the diagonal of the influence matrix), the effective degrees of freedom (its trace) and the deviance, for
the Gaussian, Poisson and binomial families on a B-spline smooth with its second-derivative penalty. statsmodels weighs
the penalty as 2 alpha, so the engine's smoothing parameter is 2 alpha."""

from typing import Any, cast

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from statsmodels.gam.api import BSplines, GLMGam

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def record(
    name: str, family: sm.families.Family, family_name: str, y: FloatArray, x: FloatArray, df: int, alpha: float
) -> Case:
    frame = pd.DataFrame({"x": x})
    smoother = BSplines(frame, df=[df], degree=[3])
    model = GLMGam(y, exog=np.ones((len(y), 1)), smoother=smoother, alpha=cast(Any, alpha), family=family)
    fit = model.fit(maxiter=200, tol=1e-12)
    design = np.column_stack([np.ones(len(y)), np.asarray(smoother.basis, dtype=np.float64)])
    penalty = np.asarray(smoother.penalty_matrices[0], dtype=np.float64)
    output = {
        "coef": np.asarray(fit.params, dtype=np.float64),
        "hat": np.asarray(cast(Any, fit).hat_matrix_diag, dtype=np.float64),
        "edf": np.asarray([cast(Any, fit).hat_matrix_trace], dtype=np.float64),
        "deviance": np.asarray([cast(Any, fit).deviance], dtype=np.float64),
    }
    if family_name == "gamma-log":
        # statsmodels' hat matrix of a family with an estimated scale is computed after the fit with the scale in
        # it; the coefficients and deviance are what the penalised IRLS is checked on there.
        del output["hat"], output["edf"]
    return case(
        name,
        "glmgam",
        [design, penalty, y],
        output,
        rtol=1e-6,
        atol=1e-8,
        family=family_name,
        lam=2.0 * alpha,
    )


def cases() -> list[Case]:
    g = rng(515)
    n = 90
    x = np.sort(g.uniform(0, 1, n))
    out: list[Case] = []
    gauss = np.sin(6 * x) + 0.25 * g.normal(size=n)
    for alpha in (1e-5, 1e-3, 0.05):
        out.append(record(f"gaussian/{alpha}", sm.families.Gaussian(), "gaussian", gauss, x, 8, alpha))
    counts = g.poisson(np.exp(0.8 + np.sin(5 * x))).astype(np.float64)
    for alpha in (1e-4, 5e-3):
        out.append(record(f"poisson/{alpha}", sm.families.Poisson(), "poisson", counts, x, 8, alpha))
    prob = 1.0 / (1.0 + np.exp(-(2.5 * np.cos(4 * x))))
    flips = g.uniform(size=n) < prob
    for alpha in (1e-4, 1e-2):
        out.append(
            record(f"binomial/{alpha}", sm.families.Binomial(), "binomial", flips.astype(np.float64), x, 7, alpha)
        )
    gamma = g.gamma(shape=4.0, scale=np.exp(0.5 + np.sin(4 * x)) / 4.0)
    out.append(record("gamma/0.001", sm.families.Gamma(sm.families.links.Log()), "gamma-log", gamma, x, 7, 1e-3))
    return out
