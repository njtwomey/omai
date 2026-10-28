"""Vector autoregression against statsmodels' VAR: coefficients, covariances, likelihood and criteria, lag-order
selection, forecasts with intervals, impulse responses, the variance decomposition, companion roots and Granger
causality tests."""

import warnings
from importlib.metadata import version
from typing import Any

import numpy as np
import numpy.typing as npt
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from statsmodels.tsa.api import VAR

REFERENCE: dict[str, str] = {"statsmodels": version("statsmodels"), "numpy": np.__version__}

type Vec = npt.NDArray[np.float64]

TOL = {"rtol": 1e-8, "atol": 1e-10}


def simulate(g: np.random.Generator, n: int) -> Vec:
    """A stable three-variable VAR(2) with correlated noise and a constant."""
    a1 = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, -0.1], [0.0, 0.3, 0.3]])
    a2 = np.array([[-0.2, 0.0, 0.1], [0.0, -0.1, 0.0], [0.1, 0.0, -0.2]])
    chol = np.linalg.cholesky(np.array([[1.0, 0.3, 0.1], [0.3, 0.8, 0.2], [0.1, 0.2, 0.6]]))
    y = np.zeros((n + 50, 3))
    for t in range(2, n + 50):
        y[t] = np.array([0.1, -0.2, 0.3]) + a1 @ y[t - 1] + a2 @ y[t - 2] + chol @ g.normal(size=3)
    return np.asarray(y[50:])


def cases() -> list[Case]:
    g = rng(5)
    y = simulate(g, 160)
    out: list[Case] = []
    for lags, trend in [(1, "c"), (2, "c"), (2, "n"), (3, "ct")]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r: Any = VAR(y).fit(lags, trend=trend)
        params = np.asarray(r.params)
        out.append(case(f"fit/{lags}{trend}/params", "varParams", [y], params, **TOL, lags=lags, trend=trend))
        out.append(
            case(f"fit/{lags}{trend}/sigma", "varSigma", [y], np.asarray(r.sigma_u), **TOL, lags=lags, trend=trend)
        )
        out.append(
            case(f"fit/{lags}{trend}/stderr", "varStdErrors", [y], np.asarray(r.stderr), **TOL, lags=lags, trend=trend)
        )
        out.append(
            case(
                f"fit/{lags}{trend}/criteria",
                "varCriteria",
                [y],
                np.array([r.llf, r.aic, r.bic, r.hqic, r.fpe]),
                **TOL,
                lags=lags,
                trend=trend,
            )
        )
        if trend != "ct":
            steps = 8
            fi = r.forecast_interval(y[-lags:], steps, alpha=0.1)
            out.append(
                case(
                    f"forecast/{lags}{trend}",
                    "varForecast",
                    [y],
                    np.stack([np.asarray(fi[0]), np.asarray(fi[1]), np.asarray(fi[2])]),
                    **TOL,
                    steps=steps,
                    alpha=0.1,
                    lags=lags,
                    trend=trend,
                )
            )
        out.append(
            case(
                f"roots/{lags}{trend}",
                "varStability",
                [y],
                np.sort(1.0 / np.abs(np.asarray(r.roots)))[::-1],
                **TOL,
                lags=lags,
                trend=trend,
            )
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel: Any = VAR(y).select_order(maxlags=5, trend="c")
    table = np.array([sel.ics[k] for k in ("aic", "bic", "hqic", "fpe")])[:, 1:]
    out.append(case("selectOrder", "varSelectOrder", [y], table, **TOL, maxLags=5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit2: Any = VAR(y).fit(2)
    irf: Any = fit2.irf(6)
    out.append(case("irf/plain", "varIrf", [y], np.asarray(irf.irfs), **TOL, lags=2, steps=6, orth=False))
    out.append(case("irf/orth", "varIrf", [y], np.asarray(irf.orth_irfs), **TOL, lags=2, steps=6, orth=True))
    out.append(case("irf/cumulative", "varIrfCum", [y], np.asarray(irf.cum_effects), **TOL, lags=2, steps=6))
    fevd: Any = fit2.fevd(8)
    out.append(case("fevd", "varFevd", [y], np.asarray(fevd.decomp), **TOL, lags=2, periods=8))
    for caused, causing in [([1], [0]), ([0], [1, 2]), ([0, 1], [2])]:
        for kind in ("f", "wald"):
            t: Any = fit2.test_causality(caused, causing, kind=kind)
            out.append(
                case(
                    f"granger/{caused}-{causing}/{kind}",
                    "granger",
                    [y],
                    np.array([t.test_statistic, t.pvalue]),
                    rtol=1e-7,
                    atol=1e-10,
                    lags=2,
                    caused=caused,
                    causing=causing,
                    kind=kind,
                )
            )
    return out
