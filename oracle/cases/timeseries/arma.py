"""ARMA theory against statsmodels: the exact autocovariance, autocorrelation and partial autocorrelation functions
(arma_acovf, arma_acf, arma_pacf), the moving-average and autoregressive weights (arma2ma, arma2ar) and the innovations
algorithm (innovations_algo)."""

from importlib.metadata import version
from typing import cast

import numpy as np
from omai_oracle.fixture import Case, case
from statsmodels.tsa.arima.estimators.innovations import innovations_algo
from statsmodels.tsa.arima_process import arma2ar, arma2ma, arma_acf, arma_acovf, arma_pacf

REFERENCE: dict[str, str] = {"statsmodels": version("statsmodels"), "numpy": np.__version__}

TOL = {"rtol": 1e-10, "atol": 1e-12}

MODELS: list[tuple[str, list[float], list[float], float]] = [
    ("ar1", [0.5], [], 1.0),
    ("ma1", [], [0.6], 2.0),
    ("ma3", [], [0.5, -0.3, 0.2], 1.0),
    ("arma11", [0.7], [0.4], 1.5),
    ("ar2ma2", [0.6, -0.3], [0.4, 0.2], 1.5),
    ("ar3ma1", [0.5, 0.1, -0.2], [-0.5], 0.5),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for name, ar, ma, sigma2 in MODELS:
        arp = np.r_[1.0, -np.array(ar)]
        map_ = np.r_[1.0, np.array(ma)]
        nlags = 12
        inputs = [np.array(ar, dtype=float), np.array(ma, dtype=float)]
        out.append(
            case(
                f"acovf/{name}",
                "armaAutocovariance",
                inputs,
                arma_acovf(arp, map_, nlags + 1, cast(int, sigma2)),
                **TOL,
                sigma2=sigma2,
                nlags=nlags,
            )
        )
        out.append(case(f"acf/{name}", "armaAcf", inputs, arma_acf(arp, map_, nlags + 1), **TOL, nlags=nlags))
        out.append(case(f"pacf/{name}", "armaPacf", inputs, arma_pacf(arp, map_, nlags + 1), **TOL, nlags=nlags))
        out.append(case(f"psi/{name}", "armaPsiWeights", inputs, arma2ma(arp, map_, 10), **TOL, n=10))
        # statsmodels' arma2ar returns the weights of the polynomial 1 - sum pi_j B^j with the opposite sign.
        out.append(case(f"pi/{name}", "armaPiWeights", inputs, -arma2ar(arp, map_, 11)[1:], **TOL, n=10))
        gamma = arma_acovf(arp, map_, 8, cast(int, sigma2))
        theta, v = innovations_algo(gamma, nobs=8)
        padded = np.zeros((8, 3))
        width = min(3, theta.shape[1])
        padded[:, :width] = theta[:, :width]
        flat = np.concatenate([v, padded[1:].ravel()])
        out.append(case(f"innovations/{name}", "innovations", [gamma], flat, **TOL))
    return out
