"""Seasonal decomposition against statsmodels: seasonal_decompose (additive and multiplicative, even and odd periods,
one- and two-sided filters, trend extrapolation), STL (defaults, robust, loess degrees and jumps) and MSTL."""

import warnings
from importlib.metadata import version
from typing import Any

import numpy as np
import numpy.typing as npt
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from statsmodels.tsa.seasonal import MSTL, STL, seasonal_decompose

REFERENCE: dict[str, str] = {"statsmodels": version("statsmodels"), "numpy": np.__version__}

type Vec = npt.NDArray[np.float64]


def seasonal_series(g: np.random.Generator, n: int, periods: list[int], multiplicative: bool = False) -> Vec:
    """A positive trend plus seasonal components of the given periods plus noise (and a few outliers)."""
    t = np.arange(n)
    trend = 20.0 + 0.1 * t + 0.002 * t**2 / 10
    season = sum(3.0 / (i + 1) * np.sin(2 * np.pi * t / p + i) for i, p in enumerate(periods))
    noise = 0.4 * g.normal(size=n)
    if multiplicative:
        return np.asarray(trend * (1.0 + 0.1 * season) * (1.0 + 0.02 * noise))
    y = trend + season + noise
    y[n // 3] += 9.0
    y[2 * n // 3] -= 7.0
    return np.asarray(y)


def stack3(trend: npt.ArrayLike, seasonal: npt.ArrayLike, resid: npt.ArrayLike) -> Vec:
    """The three components as a [3, n] array."""
    return np.vstack(
        [np.asarray(trend, dtype=float), np.asarray(seasonal, dtype=float), np.asarray(resid, dtype=float)]
    )


def cases() -> list[Case]:
    g = rng(12)
    out: list[Case] = []
    tol = {"rtol": 1e-9, "atol": 1e-10}
    y_add = seasonal_series(g, 96, [12])
    y_odd = seasonal_series(g, 70, [7])
    y_mul = seasonal_series(g, 96, [12], multiplicative=True)
    classical: list[tuple[str, Vec, int, str, bool, int]] = [
        ("additive-even", y_add, 12, "additive", False, 0),
        ("additive-odd", y_odd, 7, "additive", False, 0),
        ("multiplicative", y_mul, 12, "multiplicative", False, 0),
        ("one-sided", y_add, 12, "additive", True, 0),
        ("extrapolated", y_add, 12, "additive", False, 3),
    ]
    for name, y, period, model, one_sided, extrapolate in classical:
        r: Any = seasonal_decompose(
            y,
            model=model,
            period=period,
            two_sided=not one_sided,
            extrapolate_trend=extrapolate,
        )
        out.append(
            case(
                f"classical/{name}",
                "seasonalDecompose",
                [y],
                stack3(r.trend, r.seasonal, r.resid),
                **tol,
                period=period,
                model=model,
                oneSided=one_sided,
                extrapolateTrend=extrapolate,
            )
        )
    stl_cases: list[tuple[str, Vec, dict[str, Any], dict[str, Any]]] = [
        ("default", y_add, {"period": 12}, {}),
        ("seasonal-window", y_add, {"period": 12, "seasonal": 13}, {}),
        ("period-7", y_odd, {"period": 7, "seasonal": 9}, {}),
        ("degree-0", y_add, {"period": 12, "seasonal_deg": 0, "trend_deg": 0, "low_pass_deg": 0}, {}),
        ("jumps", y_add, {"period": 12, "seasonal_jump": 2, "trend_jump": 3, "low_pass_jump": 2}, {}),
        ("robust", y_add, {"period": 12, "robust": True}, {}),
        ("robust-iterations", y_add, {"period": 12, "robust": True}, {"inner_iter": 3, "outer_iter": 4}),
        ("trend-window", y_add, {"period": 12, "trend": 29, "low_pass": 15}, {}),
    ]
    for name, y, kwargs, fit_kwargs in stl_cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res: Any = STL(y, **kwargs).fit(**fit_kwargs)
        extra: dict[str, Any] = {}
        if fit_kwargs:
            extra = {"innerIterations": fit_kwargs["inner_iter"], "outerIterations": fit_kwargs["outer_iter"]}
        options = {
            "period": kwargs["period"],
            "seasonal": kwargs.get("seasonal"),
            "trend": kwargs.get("trend"),
            "lowPass": kwargs.get("low_pass"),
            "seasonalDegree": kwargs.get("seasonal_deg"),
            "trendDegree": kwargs.get("trend_deg"),
            "lowPassDegree": kwargs.get("low_pass_deg"),
            "robust": kwargs.get("robust"),
            "seasonalJump": kwargs.get("seasonal_jump"),
            "trendJump": kwargs.get("trend_jump"),
            "lowPassJump": kwargs.get("low_pass_jump"),
            **extra,
        }
        out.append(
            case(
                f"stl/{name}",
                "stl",
                [y],
                np.vstack([stack3(res.trend, res.seasonal, res.resid), np.asarray(res.weights, dtype=float)]),
                rtol=1e-8,
                atol=1e-9,
                options={k: v for k, v in options.items() if v is not None},
            )
        )
    y_multi = seasonal_series(g, 240, [12, 48])
    for name, periods, windows in [("two", [12, 48], None), ("windows", [12, 48], [11, 15]), ("one", [12], None)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m: Any = MSTL(y_multi, periods=periods, windows=windows).fit()
        seasonal = np.asarray(m.seasonal).reshape(len(y_multi), -1).T
        out.append(
            case(
                f"mstl/{name}",
                "mstl",
                [y_multi],
                np.vstack([np.asarray(m.trend), seasonal, np.asarray(m.resid)]),
                rtol=1e-8,
                atol=1e-9,
                periods=periods,
                windows=windows if windows is not None else [],
            )
        )
    return out
