"""Kernel density estimation against scipy.stats.gaussian_kde and statsmodels' bandwidth rules. Cross-validated
bandwidths and boundary corrections have no scipy function, so their references are the criteria written in numpy and
maximised with scipy.optimize.minimize_scalar at a tight tolerance."""

from typing import Literal

import numpy as np
import scipy
import scipy.stats as st
import statsmodels
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from scipy.optimize import minimize_scalar
from statsmodels.nonparametric.bandwidths import bw_scott, bw_silverman

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "statsmodels": statsmodels.__version__}


def cv_likelihood(x: FloatArray, h: float) -> float:
    n = x.size
    d2 = (x[:, None] - x[None, :]) ** 2
    k = np.exp(-d2 / (2 * h * h))
    np.fill_diagonal(k, 0.0)
    return float(np.sum(np.log(k.sum(1) / ((n - 1) * h * np.sqrt(2 * np.pi)))))


def cv_least_squares(x: FloatArray, h: float) -> float:
    n = x.size
    d2 = (x[:, None] - x[None, :]) ** 2
    isq = np.exp(-d2 / (4 * h * h)).sum() / (n * n * h * np.sqrt(4 * np.pi))
    k = np.exp(-d2 / (2 * h * h))
    np.fill_diagonal(k, 0.0)
    loo = 2 * k.sum() / (n * (n - 1) * h * np.sqrt(2 * np.pi))
    return float(loo - isq)


def best(x: FloatArray, f: str) -> float:
    crit = cv_likelihood if f == "likelihood" else cv_least_squares

    def objective(t: float) -> float:
        return -crit(x, float(np.exp(t)))

    h0 = np.log(bw_silverman(x))
    r = minimize_scalar(
        objective, bounds=(h0 - np.log(10), h0 + np.log(10)), method="bounded", options={"xatol": 1e-12}
    )
    return float(np.exp(r.x))


def cases() -> list[Case]:
    g = rng(4242)
    x = np.concatenate([g.normal(0, 1, 40), g.normal(3, 0.5, 20)])
    w = g.uniform(0.2, 2.0, x.size)
    X2 = g.normal(size=(50, 2)) @ np.array([[1.0, 0.5], [0.0, 0.8]])
    pts = np.linspace(-3, 5, 9)
    pts2 = g.normal(size=(6, 2))
    out: list[Case] = []
    rules: list[Literal["scott", "silverman"]] = ["scott", "silverman"]
    for bw in rules:
        kde = st.gaussian_kde(x, bw_method=bw)
        out.append(case(f"evaluate/{bw}", "evaluate", [x, pts], kde.evaluate(pts), rtol=1e-10, bandwidth=bw))
        out.append(case(f"factor/{bw}", "factor", [x, x], np.asarray(kde.factor), bandwidth=bw))
        kde = st.gaussian_kde(x, bw_method=bw, weights=w)
        out.append(
            case(
                f"evaluate/weighted/{bw}", "evaluateWeighted", [x, pts, w], kde.evaluate(pts), rtol=1e-10, bandwidth=bw
            )
        )
        kde = st.gaussian_kde(X2.T, bw_method=bw)
        out.append(case(f"evaluate/2d/{bw}", "evaluate", [X2, pts2], kde.evaluate(pts2.T), rtol=1e-10, bandwidth=bw))
        out.append(case(f"logpdf/2d/{bw}", "logpdf", [X2, pts2], kde.logpdf(pts2.T), rtol=1e-10, bandwidth=bw))
    kde = st.gaussian_kde(x, bw_method=0.3)
    out.append(case("evaluate/factor", "evaluate", [x, pts], kde.evaluate(pts), rtol=1e-10, bandwidth=0.3))
    out.append(case("covariance/2d", "covariance", [X2, X2], st.gaussian_kde(X2.T).covariance))
    out.append(case("bandwidthScott", "bandwidthScott", [x, x], np.asarray(bw_scott(x))))
    out.append(case("bandwidthSilverman", "bandwidthSilverman", [x, x], np.asarray(bw_silverman(x))))
    for method in ["likelihood", "least-squares"]:
        out.append(
            case(
                f"bandwidthCrossValidation/{method}",
                "bandwidthCrossValidation",
                [x, x],
                np.asarray(best(x, method)),
                rtol=1e-7,
                method=method,
            )
        )
        crit = cv_likelihood if method == "likelihood" else cv_least_squares
        out.append(
            case(
                f"bandwidthCriterion/{method}",
                "bandwidthCriterion",
                [x, x],
                np.asarray(crit(x, 0.4)),
                method=method,
                h=0.4,
            )
        )
    y = g.exponential(1.0, 50)
    kde = st.gaussian_kde(y)
    grid = np.linspace(0.0, 4.0, 9)
    reflected = kde.evaluate(grid) + kde.evaluate(-grid)
    h = np.sqrt(kde.covariance[0, 0])
    renormalised = kde.evaluate(grid) / st.norm.cdf(grid / h)
    out.append(case("boundary/reflection", "reflection", [y, grid], reflected, rtol=1e-10))
    out.append(case("boundary/renormalisation", "renormalisation", [y, grid], renormalised, rtol=1e-10))
    return out
