"""Autoregressive estimation against statsmodels (yule_walker, burg, levinson_durbin) and AR spectra by direct
evaluation of sigma^2 / |A(e^{jw})|^2 with numpy."""

import warnings
from importlib.metadata import version
from typing import Literal

import numpy as np
import statsmodels.regression.linear_model as lm
import statsmodels.tsa.stattools as ts
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"statsmodels": version("statsmodels"), "numpy": np.__version__}

TOL = {"rtol": 1e-9, "atol": 1e-12}


def ar_series(g: np.random.Generator, n: int) -> np.ndarray:
    """An AR(3) series."""
    e = g.normal(size=n + 50)
    x = np.zeros(n + 50)
    for t in range(3, n + 50):
        x[t] = 0.6 * x[t - 1] - 0.3 * x[t - 2] + 0.1 * x[t - 3] + e[t]
    return x[50:]


def cases() -> list[Case]:
    g = rng(408)
    out: list[Case] = []
    x = ar_series(g, 80) + 0.4
    for order in [1, 3, 5]:
        methods: list[Literal["adjusted", "mle"]] = ["adjusted", "mle"]
        for method in methods:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                fit = lm.yule_walker(x, order=order, method=method, result_object=True)
            rho, sigma = fit[0], fit[1]
            res = np.concatenate([rho, [sigma**2]])
            out.append(case(f"yuleWalker/{order}/{method}", "yuleWalker", [x], res, **TOL, order=order, method=method))
        ar, s2 = lm.burg(x, order=order)
        out.append(case(f"burg/{order}", "burg", [x], np.concatenate([ar, [s2]]), **TOL, order=order))
    acov = ts.acovf(x, fft=False)[:6]
    ld = ts.levinson_durbin(acov, nlags=5, isacov=True)
    res = np.concatenate([ld.arcoefs, [ld.sigma_v], ld.pacf[1:]])
    out.append(case("levinsonDurbin", "levinsonDurbin", [acov], res, **TOL))
    phi = np.array([0.6, -0.3, 0.1])
    f = np.linspace(0, 0.5, 9)
    z = np.exp(-2j * np.pi * np.outer(f, np.arange(1, 4)))
    two = 1.3 / np.abs(1 - z @ phi) ** 2
    one = two * np.where((f > 0) & (f < 0.5), 2, 1)
    out.append(case("arSpectrum/onesided", "arSpectrum", [phi], one, **TOL, sigma2=1.3, n=9, onesided=True))
    out.append(case("arSpectrum/twosided", "arSpectrum", [phi], two, **TOL, sigma2=1.3, n=9, onesided=False))
    return out
