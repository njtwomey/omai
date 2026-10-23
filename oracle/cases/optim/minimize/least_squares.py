"""Nonlinear and linear least squares against scipy.optimize: least_squares (lm, trf and dogbox, with bounds and
robust losses), curve_fit (parameters and covariance), nnls and lsq_linear (bvls). The TypeScript test defines the same
residuals."""

from collections.abc import Callable
from typing import Literal

import numpy as np
import scipy
import scipy.optimize as so
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type Residuals = Callable[[FloatArray], FloatArray]

T = np.linspace(0.0, 4.0, 12)
_g = rng(11)
Y_EXP = 2.5 * np.exp(-1.3 * T) + 0.4 + 0.02 * _g.normal(size=T.size)
Y_OUT = Y_EXP.copy()
Y_OUT[[3, 8]] += np.array([1.5, -1.0])


def rosenbrock(x: FloatArray) -> FloatArray:
    return np.array([10 * (x[1] - x[0] * x[0]), 1 - x[0]])


def helical(x: FloatArray) -> FloatArray:
    theta = np.arctan2(x[1], x[0]) / (2 * np.pi)
    return np.array([10 * (x[2] - 10 * theta), 10 * (np.hypot(x[0], x[1]) - 1), x[2]])


def beale(x: FloatArray) -> FloatArray:
    return np.array([1.5 - x[0] * (1 - x[1]), 2.25 - x[0] * (1 - x[1] ** 2), 2.625 - x[0] * (1 - x[1] ** 3)])


def exp_model(t: FloatArray, a: float, k: float, c: float) -> FloatArray:
    return a * np.exp(-k * t) + c


def exponential(x: FloatArray) -> FloatArray:
    return exp_model(T, float(x[0]), float(x[1]), float(x[2])) - Y_EXP


def outliers(x: FloatArray) -> FloatArray:
    return exp_model(T, float(x[0]), float(x[1]), float(x[2])) - Y_OUT


RESIDUALS: dict[str, tuple[Residuals, FloatArray]] = {
    "rosenbrock": (rosenbrock, np.array([-1.2, 1.0])),
    "helical": (helical, np.array([-1.0, 0.0, 0.0])),
    "beale": (beale, np.array([1.0, 1.0])),
    "exponential": (exponential, np.array([1.0, 1.0, 0.0])),
    "outliers": (outliers, np.array([1.0, 1.0, 0.0])),
}

type Loss = Literal["linear", "soft_l1", "huber", "cauchy", "arctan"]

METHODS: list[Literal["dogbox", "trf"]] = ["dogbox", "trf"]


def result(r: so.OptimizeResult) -> FloatArray:
    return np.concatenate([np.asarray(r.x, dtype=np.float64), [float(r.cost)]])


def cases() -> list[Case]:
    out: list[Case] = []
    for name in ["rosenbrock", "helical", "beale", "exponential"]:
        f, x0 = RESIDUALS[name]
        r = so.least_squares(f, x0, method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12)
        out.append(
            case(f"lm/{name}", "levenbergMarquardt", [x0, T, Y_EXP], result(r), rtol=1e-7, atol=1e-9, residuals=name)
        )
        out.append(case(f"gn/{name}", "gaussNewton", [x0, T, Y_EXP], result(r), rtol=1e-6, atol=1e-8, residuals=name))
    for name in ["rosenbrock", "helical", "beale", "exponential"]:
        f, x0 = RESIDUALS[name]
        r = so.least_squares(f, x0, method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12)
        out.append(case(f"trf/{name}", "trf", [x0, T, Y_EXP], result(r), rtol=1e-7, atol=1e-9, residuals=name))
    bounded: list[tuple[str, list[tuple[float, float]]]] = [
        ("rosenbrock", [(-2.0, 0.8), (-2.0, 2.0)]),
        ("beale", [(0.0, 2.5), (0.0, 0.4)]),
        ("exponential", [(0.0, 2.0), (0.0, 5.0), (0.5, 1.0)]),
    ]
    for name, bounds in bounded:
        f, x0 = RESIDUALS[name]
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        start = np.clip(x0, lb, ub)
        for method in METHODS:
            r = so.least_squares(f, start, method=method, bounds=(lb, ub), xtol=1e-12, ftol=1e-12, gtol=1e-12)
            out.append(
                case(
                    f"{method}-bounded/{name}",
                    method,
                    [start, T, Y_EXP, lb, ub],
                    np.concatenate([result(r), np.asarray(r.active_mask, dtype=np.float64)]),
                    rtol=1e-6,
                    atol=1e-8,
                    residuals=name,
                )
            )
    losses: list[Loss] = ["soft_l1", "huber", "cauchy", "arctan"]
    for loss in losses:
        f, x0 = RESIDUALS["outliers"]
        for method in METHODS:
            r = so.least_squares(f, x0, method=method, loss=loss, f_scale=0.1, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            out.append(
                case(
                    f"robust-{method}/{loss}",
                    method,
                    [x0, T, Y_OUT],
                    result(r),
                    rtol=1e-6,
                    atol=1e-8,
                    residuals="outliers",
                    loss=loss,
                )
            )
    # curve_fit: plain, weighted (relative and absolute sigma), and bounded.
    sigma = 0.02 + 0.01 * T
    fits: list[tuple[str, FloatArray | None, bool, tuple[list[float], list[float]] | None]] = [
        ("plain", None, False, None),
        ("sigma", sigma, False, None),
        ("absolute", sigma, True, None),
        ("bounded", None, False, ([0.0, 0.0, 0.5], [2.0, 5.0, 1.0])),
    ]
    for label, sg, absolute, bnds in fits:
        if bnds is None:
            popt, pcov = so.curve_fit(exp_model, T, Y_EXP, p0=[1.0, 1.0, 0.0], sigma=sg, absolute_sigma=absolute)
        else:
            popt, pcov = so.curve_fit(
                exp_model, T, Y_EXP, p0=[1.0, 1.0, 0.5], bounds=bnds, method="trf", ftol=1e-14, xtol=1e-14, gtol=1e-14
            )
        params: dict[str, object] = {"absoluteSigma": absolute}
        inputs: list[object] = [T, Y_EXP, np.zeros(1) if sg is None else sg]
        if bnds is not None:
            params["lower"] = bnds[0]
            params["upper"] = bnds[1]
        out.append(
            case(
                f"curveFit/{label}",
                "curveFit",
                inputs,
                np.concatenate([np.asarray(popt, dtype=np.float64), np.asarray(pcov, dtype=np.float64).ravel()]),
                rtol=1e-5,
                atol=1e-8,
                hasSigma=sg is not None,
                **params,
            )
        )
    g = rng(5)
    for i in range(3):
        a = g.normal(size=(8, 5))
        b = g.normal(size=8)
        x, rnorm = so.nnls(a, b)
        out.append(case(f"nnls/{i}", "nnls", [a, b], np.concatenate([x, [rnorm]]), rtol=1e-10, atol=1e-12))
        lb = np.full(5, -0.3)
        ub = np.full(5, 0.4)
        r = so.lsq_linear(a, b, bounds=(lb, ub), method="bvls", tol=1e-12)
        out.append(
            case(
                f"bvls/{i}",
                "lsqLinear",
                [a, b, lb, ub],
                np.concatenate([np.asarray(r.x, dtype=np.float64), [float(r.cost)]]),
                rtol=1e-10,
                atol=1e-12,
            )
        )
    return out
