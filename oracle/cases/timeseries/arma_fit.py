"""ARMA estimators against statsmodels: Hannan-Rissanen, the innovations estimator of moving averages, least-squares
autoregressions (AutoReg) and the information criteria over AR orders."""

import warnings
from importlib.metadata import version
from typing import Any

import numpy as np
import numpy.typing as npt
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen
from statsmodels.tsa.arima.estimators.innovations import innovations

REFERENCE: dict[str, str] = {"statsmodels": version("statsmodels"), "numpy": np.__version__}

TOL = {"rtol": 1e-8, "atol": 1e-10}

type Vec = npt.NDArray[np.float64]


def arma_series(g: np.random.Generator, n: int, ar: list[float], ma: list[float]) -> Vec:
    """A Gaussian ARMA series after a burn-in."""
    burn = 100
    e = g.normal(size=n + burn)
    x = np.zeros(n + burn)
    for t in range(max(len(ar), len(ma)), n + burn):
        x[t] = e[t] + sum(a * x[t - 1 - i] for i, a in enumerate(ar)) + sum(b * e[t - 1 - i] for i, b in enumerate(ma))
    return x[burn:]


def cases() -> list[Case]:
    g = rng(404)
    x = arma_series(g, 300, [0.6, -0.2], [0.5]) + 3.0
    out: list[Case] = []
    for p, q in [(1, 1), (2, 1), (1, 2), (2, 0)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for unbiased in (None, False):
                res = hannan_rissanen(x, ar_order=p, ma_order=q, unbiased=unbiased)
                par: Any = res.parameters
                label = "default" if unbiased is None else "biased"
                flat = np.concatenate([par.ar_params, par.ma_params, [par.sigma2]])
                name = f"hannanRissanen/{p}{q}/{label}"
                if unbiased is None:
                    out.append(case(name, "hannanRissanen", [x], flat, **TOL, order=[p, q]))
                else:
                    out.append(case(name, "hannanRissanen", [x], flat, **TOL, order=[p, q], unbiased=unbiased))
    for q in (1, 2, 3):
        estimates: Any = innovations(x, ma_order=q).parameters
        res_i: Any = estimates[q]
        out.append(
            case(f"innovations/{q}", "fitMa", [x], np.concatenate([res_i.ma_params, [res_i.sigma2]]), **TOL, order=q)
        )
    for p in (1, 2, 3):
        r = AutoReg(x, lags=p, trend="c").fit()
        out.append(
            case(f"autoreg/{p}", "fitArOls", [x], np.concatenate([np.asarray(r.params), [r.sigma2]]), **TOL, order=p)
        )
    for criterion in ("aic", "bic", "hqic"):
        scores = np.array([getattr(AutoReg(x, lags=p, trend="c", hold_back=6).fit(), criterion) for p in range(7)])
        out.append(
            case(
                f"selectArOrder/{criterion}",
                "selectArOrder",
                [x],
                scores,
                rtol=1e-9,
                atol=1e-8,
                maxOrder=6,
                criterion=criterion,
            )
        )
    return out
