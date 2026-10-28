"""Sample correlation functions against statsmodels: acovf, acf (with Bartlett bands), pacf (Yule-Walker adjusted and
biased, and OLS) and ccf."""

import warnings
from importlib.metadata import version
from typing import Any, Literal, cast

import numpy as np
import statsmodels.tsa.stattools as ts
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"statsmodels": version("statsmodels"), "numpy": np.__version__}

TOL = {"rtol": 1e-10, "atol": 1e-12}


def cases() -> list[Case]:
    g = rng(77)
    n = 120
    e = g.normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + e[t] + 0.3 * e[t - 1]
    x += 2.0
    y = np.roll(x, 3) + 0.5 * g.normal(size=n)
    out: list[Case] = []
    for adjusted in (False, True):
        out.append(
            case(
                f"acovf/adjusted={adjusted}",
                "autocovariance",
                [x],
                ts.acovf(x, adjusted=adjusted, nlag=12),
                **TOL,
                nlags=12,
                adjusted=adjusted,
            )
        )
    out.append(
        case("acovf/nodemean", "autocovariance", [x], ts.acovf(x, demean=False, nlag=8), **TOL, nlags=8, demean=False)
    )
    acf_result = cast(tuple[Any, Any], ts.acf(x, nlags=15, alpha=0.05, result_object=False))
    r = np.asarray(acf_result[0])
    ci = np.asarray(acf_result[1])
    out.append(case("acf", "acf", [x], r, **TOL, nlags=15))
    out.append(case("acf/band", "acfBand", [x], (ci[:, 1] - r)[1:], **TOL, nlags=15, alpha=0.05))
    methods: list[tuple[Literal["ywadjusted", "ywmle", "ols"], str]] = [
        ("ywadjusted", "yule-walker"),
        ("ywmle", "yule-walker-mle"),
        ("ols", "ols"),
    ]
    for method, ours in methods:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = ts.pacf(x, nlags=12, method=method)
        tol = TOL if method != "ols" else {"rtol": 1e-8, "atol": 1e-10}
        out.append(case(f"pacf/{method}", "pacf", [x], p, **tol, nlags=12, method=ours))
    out.append(case("ccf", "ccf", [x, y], ts.ccf(x, y, adjusted=True)[:13], **TOL, nlags=12, adjusted=True))
    out.append(case("ccf/biased", "ccf", [x, y], ts.ccf(x, y, adjusted=False)[:13], **TOL, nlags=12, adjusted=False))
    return out
