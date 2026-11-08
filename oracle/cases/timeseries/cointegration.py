"""Cointegration against statsmodels: the Engle-Granger test (`coint`) with its MacKinnon p-values and critical values
for two and three series, Johansen's trace and maximum-eigenvalue tests (`coint_johansen`) over the deterministic orders
and lag counts, and the vector error-correction model (`VECM`) by maximum likelihood with its forecasts."""

import warnings
from importlib.metadata import version
from typing import Any

import numpy as np
import numpy.typing as npt
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

REFERENCE: dict[str, str] = {"statsmodels": version("statsmodels"), "numpy": np.__version__}

type Mat = npt.NDArray[np.float64]

TOL = {"rtol": 1e-7, "atol": 1e-9}


def system(g: np.random.Generator, n: int) -> Mat:
    """Three series: two share a stochastic trend (y1 = 0.8 y0 + stationary AR(1) spread), one is a free random walk."""
    e = g.normal(size=(n, 3))
    walk = np.cumsum(e[:, 0])
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.5 * spread[t - 1] + 0.6 * e[t, 1]
    free = np.cumsum(e[:, 2]) * 0.7
    return np.column_stack([walk + 0.3 * e[:, 1], 0.8 * walk + spread + 1.0, free])


def cases() -> list[Case]:
    g = rng(11)
    y = system(g, 180)
    out: list[Case] = []
    for trend in ("c", "ct", "ctt", "n"):
        for k, cols in (("2", [1]), ("3", [1, 2])):
            for autolag in ("aic", None):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r: Any = coint(y[:, 0], y[:, cols], trend=trend, maxlag=4, autolag=autolag)
                out.append(
                    case(
                        f"engle-granger/{trend}/{k}/{autolag}",
                        "engleGranger",
                        [y],
                        np.array([r[0], r[1], *r[2]]),
                        **TOL,
                        trend=trend,
                        cols=cols,
                        autolag=autolag,
                        maxlag=4,
                    )
                )
    for det in (-1, 0, 1):
        for kd in (1, 2, 3):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                j: Any = coint_johansen(y, det, kd)
            out.append(case(f"johansen/{det}/{kd}/eig", "johansenEig", [y], np.asarray(j.eig), **TOL, det=det, kd=kd))
            out.append(
                case(
                    f"johansen/{det}/{kd}/stats",
                    "johansenStats",
                    [y],
                    np.stack([j.lr1, j.lr2]),
                    **TOL,
                    det=det,
                    kd=kd,
                )
            )
            out.append(
                case(
                    f"johansen/{det}/{kd}/crit",
                    "johansenCrit",
                    [y],
                    np.concatenate([np.asarray(j.cvt), np.asarray(j.cvm)]),
                    rtol=1e-12,
                    atol=1e-12,
                    det=det,
                    kd=kd,
                )
            )
            out.append(
                case(
                    f"johansen/{det}/{kd}/evec", "johansenEvec", [y], np.abs(np.asarray(j.evec)), **TOL, det=det, kd=kd
                )
            )
    two = y[:, :2]
    for det in ("n", "co", "ci", "lo", "li", "colo", "cili"):
        for kd, rank in ((1, 1), (2, 1)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m: Any = VECM(two, k_ar_diff=kd, coint_rank=rank, deterministic=det).fit()
            tag = f"{det}/{kd}/{rank}"
            parts = {
                "alpha": m.alpha,
                "beta": m.beta,
                "gamma": m.gamma,
                "sigma": m.sigma_u,
                "llf": np.array([[m.llf]]),
            }
            for name, val in parts.items():
                out.append(
                    case(
                        f"vecm/{tag}/{name}",
                        "vecmPart",
                        [two],
                        np.atleast_2d(val),
                        **TOL,
                        det=det,
                        kd=kd,
                        rank=rank,
                        part=name,
                    )
                )
            if m.det_coef.size:
                out.append(
                    case(
                        f"vecm/{tag}/det",
                        "vecmPart",
                        [two],
                        np.atleast_2d(m.det_coef),
                        **TOL,
                        det=det,
                        kd=kd,
                        rank=rank,
                        part="det",
                    )
                )
            if m.det_coef_coint.size:
                out.append(
                    case(
                        f"vecm/{tag}/detcoint",
                        "vecmPart",
                        [two],
                        np.atleast_2d(m.det_coef_coint),
                        **TOL,
                        det=det,
                        kd=kd,
                        rank=rank,
                        part="detcoint",
                    )
                )
            fc = m.predict(steps=6, alpha=0.1)
            out.append(
                case(
                    f"vecm/{tag}/forecast",
                    "vecmForecast",
                    [two],
                    np.stack(fc),
                    **TOL,
                    det=det,
                    kd=kd,
                    rank=rank,
                    steps=6,
                    alpha=0.1,
                )
            )
    return out
