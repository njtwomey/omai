"""Exponential smoothing against statsmodels' ETSModel: the smoothing recursions and the log-likelihood at given
parameters (`smooth`), forecasts with prediction intervals, and the maximised likelihood of a fit (the engine's
optimum must be at least as high)."""

import warnings
from importlib.metadata import version
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

REFERENCE: dict[str, str] = {"statsmodels": version("statsmodels"), "numpy": np.__version__}

type Vec = npt.NDArray[np.float64]

Kind = Literal["add", "mul"] | None

SPECS: list[tuple[str, Kind, Kind, bool, Kind, int]] = [
    ("ses", "add", None, False, None, 1),
    ("holt", "add", "add", False, None, 1),
    ("damped", "add", "add", True, None, 1),
    ("hw-add", "add", "add", False, "add", 4),
    ("hw-damped-add", "add", "add", True, "add", 4),
    ("seasonal-only", "add", None, False, "add", 4),
    ("mul-error-seasonal", "mul", "add", False, "mul", 4),
    ("mul-error-ses", "mul", None, False, None, 1),
    ("mul-trend", "add", "mul", False, None, 1),
    ("mul-damped", "mul", "add", True, None, 1),
]


def series(g: np.random.Generator, n: int) -> Vec:
    """A positive series with a trend, a seasonal pattern of period 4 and noise."""
    t = np.arange(n)
    return 60.0 + 0.4 * t + 6.0 * np.sin(2 * np.pi * t / 4) + 1.5 * g.normal(size=n)


def cases() -> list[Case]:
    g = rng(31)
    y = series(g, 60)
    out: list[Case] = []
    steps = 8
    for name, error, trend, damped, seasonal, period in SPECS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ETSModel(
                pd.Series(y),
                error=error or "add",
                trend=trend,
                damped_trend=damped,
                seasonal=seasonal,
                seasonal_periods=period if seasonal else None,
            )
            fit: Any = model.fit(disp=False, maxiter=2000)
            params = np.asarray(fit.params)
            res: Any = model.smooth(params)
            pred: Any = res.get_prediction(start=len(y), end=len(y) + steps - 1).summary_frame(alpha=0.05)
        additive = error == "add" and trend != "mul" and seasonal != "mul"
        parts = [[float(res.llf)], np.asarray(res.fittedvalues), np.asarray(res.forecast(steps))]
        if additive:
            # Intervals of the multiplicative models are simulated; the additive ones are analytic.
            parts += [np.asarray(pred["pi_lower"]), np.asarray(pred["pi_upper"])]
        flat = np.concatenate(parts)
        structure = {
            "error": error or "add",
            "trend": trend or "none",
            "damped": damped,
            "seasonal": seasonal or "none",
            "period": period,
        }
        out.append(
            case(
                f"smooth/{name}",
                "etsSmooth",
                [y, params],
                flat,
                rtol=1e-8,
                atol=1e-8,
                steps=steps,
                additive=additive,
                structure=structure,
            )
        )
        out.append(
            case(f"fit/{name}", "etsFitBound", [y], np.array([float(fit.llf)]), rtol=0.0, atol=0.0, structure=structure)
        )
    return out
