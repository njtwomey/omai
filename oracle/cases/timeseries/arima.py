"""ARIMA and seasonal ARIMA against statsmodels' `ARIMA` (regression with ARIMA errors): fitted coefficients,
innovation variance, log-likelihood and information criteria by exact maximum likelihood, and forecasts with their
standard errors and intervals from the diffuse state-space model at the fitted parameters."""

import warnings
from importlib.metadata import version

import numpy as np
import numpy.typing as npt
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from statsmodels.tsa.arima.model import ARIMA

REFERENCE: dict[str, str] = {"statsmodels": version("statsmodels"), "numpy": np.__version__}

type Vec = npt.NDArray[np.float64]


def arma_series(g: np.random.Generator, n: int, ar: list[float], ma: list[float]) -> Vec:
    """A Gaussian ARMA series after a burn-in."""
    burn = 100
    e = g.normal(size=n + burn)
    x = np.zeros(n + burn)
    for t in range(max(len(ar), len(ma)), n + burn):
        x[t] = e[t] + sum(a * x[t - 1 - i] for i, a in enumerate(ar)) + sum(b * e[t - 1 - i] for i, b in enumerate(ma))
    return x[burn:]


def airline(g: np.random.Generator, n: int) -> Vec:
    """A (0, 1, 1)(0, 1, 1)_12 process: the MA series is integrated by (1 - B)(1 - B^12), with a level of 100."""
    burn = 40
    e = g.normal(size=n + burn)
    w = e.copy()
    w[1:] += -0.4 * e[:-1]
    w[12:] += -0.6 * e[:-12]
    w[13:] += 0.24 * e[:-13]
    y = np.zeros(n + burn)
    for t in range(n + burn):
        y[t] = w[t]
        y[t] += y[t - 1] if t >= 1 else 0.0
        y[t] += y[t - 12] if t >= 12 else 0.0
        y[t] -= y[t - 13] if t >= 13 else 0.0
    return y[burn:] + 100.0


def fit_cases(
    name: str,
    y: Vec,
    order: tuple[int, int, int],
    seasonal: tuple[int, int, int, int],
    trend: str,
    exog: Vec | None,
    rtol: float,
) -> list[Case]:
    """The coefficient and likelihood cases of one fit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(y, exog=exog, order=order, seasonal_order=seasonal, trend=trend).fit()
    inputs: list[object] = [y] if exog is None else [y, exog]
    spec = {"order": list(order), "seasonal": list(seasonal), "trend": trend}
    n_eff = len(y) - order[1] - seasonal[1] * seasonal[3]
    out = [case(f"params/{name}", "arimaFit", inputs, np.asarray(res.params), rtol=rtol, atol=1e-5, **spec)]
    # Information criteria are on the effective sample (the first differences are not observations of the likelihood).
    k = len(res.params)
    llf = float(res.llf)
    out.append(
        case(
            f"likelihood/{name}",
            "arimaLikelihood",
            inputs,
            np.array([llf, -2 * llf + 2 * k, -2 * llf + k * np.log(n_eff)]),
            rtol=1e-5,
            atol=1e-3,
            **spec,
        )
    )
    return out


def forecast_case(
    name: str,
    y: Vec,
    order: tuple[int, int, int],
    seasonal: tuple[int, int, int, int],
    trend: str,
    steps: int,
    exog: Vec | None = None,
    future: Vec | None = None,
) -> Case:
    """Forecasts at statsmodels' fitted parameters, which are passed to the engine so only the forecast is compared."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = ARIMA(y, exog=exog, order=order, seasonal_order=seasonal, trend=trend)
        params = np.asarray(mod.fit().params)
        res = mod.smooth(params)
        fc = res.get_forecast(steps, exog=future)
    ci = np.asarray(fc.conf_int(alpha=0.05))
    out = np.stack([np.asarray(fc.predicted_mean), np.asarray(fc.se_mean), ci[:, 0], ci[:, 1]])
    inputs: list[object] = [y, params] if exog is None or future is None else [y, params, np.vstack([exog, future])]
    spec = {"order": list(order), "seasonal": list(seasonal), "trend": trend, "steps": steps}
    return case(f"forecast/{name}", "arimaForecast", inputs, out, rtol=1e-5, atol=1e-6, **spec)


def cases() -> list[Case]:
    g = rng(2026)
    out: list[Case] = []
    y_arma = arma_series(g, 250, [0.6, -0.25], [0.4]) + 1.5
    centred = y_arma - 1.5
    walk = np.cumsum(arma_series(g, 220, [0.4], [0.3]) + 0.15)
    seasonal_ar = arma_series(g, 160, [0.4], [])
    quarterly = np.zeros(160)
    for t in range(160):
        quarterly[t] = seasonal_ar[t] + (0.5 * quarterly[t - 4] if t >= 4 else 0.0)
    x_reg = g.normal(size=(200, 2))
    err = arma_series(g, 200, [0.5], [])
    y_reg = x_reg @ np.array([2.0, -1.0]) + err
    air = airline(g, 144)

    out += fit_cases("ar1", centred, (1, 0, 0), (0, 0, 0, 0), "n", None, 2e-4)
    out += fit_cases("arma21-const", y_arma, (2, 0, 1), (0, 0, 0, 0), "c", None, 2e-4)
    out += fit_cases("ma2", centred, (0, 0, 2), (0, 0, 0, 0), "n", None, 2e-4)
    out += fit_cases("sar", quarterly, (1, 0, 0), (1, 0, 0, 4), "n", None, 2e-4)
    out += fit_cases("regression", y_reg, (1, 0, 0), (0, 0, 0, 0), "n", x_reg, 2e-4)
    out += fit_cases("arima111-drift", walk, (1, 1, 1), (0, 0, 0, 0), "t", None, 2e-3)
    out += fit_cases("airline", air, (0, 1, 1), (0, 1, 1, 12), "n", None, 2e-3)

    out.append(forecast_case("arma21-const", y_arma, (2, 0, 1), (0, 0, 0, 0), "c", 12))
    out.append(forecast_case("arima111-drift", walk, (1, 1, 1), (0, 0, 0, 0), "t", 12))
    out.append(forecast_case("airline", air, (0, 1, 1), (0, 1, 1, 12), "n", 24))
    out.append(forecast_case("sar", quarterly, (1, 0, 0), (1, 0, 0, 4), "n", 8))
    future = g.normal(size=(6, 2))
    out.append(forecast_case("regression", y_reg, (1, 0, 0), (0, 0, 0, 0), "n", 6, x_reg, future))
    return out
