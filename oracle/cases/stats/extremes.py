"""Extreme-value fits against scipy.stats.genextreme and genpareto (fit with a tightened optimiser; xi = -c for the
GEV), return levels from scipy's quantile functions, and the mean excess function from its definition."""

from collections.abc import Callable

import numpy as np
import scipy
import scipy.stats as st
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from scipy.optimize import fmin

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def tight(func: Callable[..., float], x0: FloatArray, args: tuple[object, ...] = (), disp: int = 0) -> FloatArray:
    return np.asarray(fmin(func, x0, args=args, xtol=1e-12, ftol=1e-14, maxiter=40000, maxfun=80000, disp=0))


def cases() -> list[Case]:
    g = rng(1999)
    out: list[Case] = []
    for i, xi in enumerate([0.15, -0.2]):
        x = st.genextreme.rvs(-xi, loc=10.0, scale=2.0, size=80, random_state=g)
        c, loc, scale = st.genextreme.fit(x, optimizer=tight)
        out.append(case(f"gev/{i}", "gev", [x, x], np.asarray([loc, scale, -c]), rtol=1e-5))
        for T in [10.0, 100.0]:
            out.append(
                case(
                    f"gev/return/{i}/{T}",
                    "gevReturn",
                    [x, x],
                    np.asarray(st.genextreme.isf(1 / T, c, loc, scale)),
                    rtol=1e-5,
                    period=T,
                )
            )
    e = st.genpareto.rvs(0.2, scale=1.5, size=100, random_state=g)
    xi, _, scale = st.genpareto.fit(e, floc=0, optimizer=tight)
    out.append(case("gpd", "gpd", [e, e], np.asarray([scale, xi]), rtol=1e-5))
    out.append(
        case(
            "gpd/return",
            "gpdReturn",
            [e, e],
            np.asarray(st.genpareto.isf(1 / (200 * 0.1), xi, 0, scale)),
            rtol=1e-5,
            period=200,
            rate=0.1,
        )
    )
    thresholds = np.array([0.0, 0.5, 1.0, 2.0])
    me = np.array([np.mean(e[e > u] - u) for u in thresholds])
    out.append(case("meanExcess", "meanExcess", [e, thresholds], me))
    return out
