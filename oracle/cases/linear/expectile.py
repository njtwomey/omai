"""Expectile regression against scipy: the asymmetric least-squares objective minimised by BFGS with its analytic
gradient at a tight tolerance (there is no expectile regression in statsmodels or scikit-learn), for several levels,
a ridge penalty and sample weights, and the Newey-Powell sandwich covariance from its definition in numpy."""

import numpy as np
import numpy.typing as npt
import scipy
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from scipy.optimize import minimize

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type Vec = npt.NDArray[np.float64]


def solve(x: Vec, y: Vec, w: Vec, tau: float, alpha: float) -> tuple[Vec, Vec]:
    n, d = x.shape
    a = np.column_stack([x, np.ones(n)])
    total = float(w.sum())

    def parts(theta: Vec) -> tuple[Vec, Vec]:
        r = y - a @ theta
        return r, np.where(r >= 0, tau, 1.0 - tau)

    def value(theta: Vec) -> float:
        r, s = parts(theta)
        return float(np.sum(w * s * r * r) / total + 0.5 * alpha * np.sum(theta[:d] ** 2))

    def gradient(theta: Vec) -> Vec:
        r, s = parts(theta)
        g = -2.0 * (a.T @ (w * s * r)) / total
        g[:d] += alpha * theta[:d]
        return g

    theta0 = np.zeros(d + 1)
    theta0[d] = float(np.average(y, weights=w))
    res = minimize(value, theta0, jac=gradient, method="BFGS", options={"gtol": 1e-13, "maxiter": 10000})
    theta = np.asarray(res.x, dtype=np.float64)
    r, s = parts(theta)
    bread = (a * (w * s)[:, None]).T @ a
    bread[np.arange(d), np.arange(d)] += 0.5 * alpha * total
    meat = (a * ((w * s * r) ** 2)[:, None]).T @ a
    inv = np.linalg.inv(bread)
    bse = np.sqrt(np.diag(inv @ meat @ inv))
    return theta, bse


def cases() -> list[Case]:
    g = rng(1987)
    out: list[Case] = []
    for name, n, d in [("one", 60, 1), ("three", 120, 3)]:
        x = g.normal(size=(n, d))
        y = x @ np.linspace(1.5, -0.5, d) + 1.0 + (0.5 + 0.4 * np.abs(x[:, 0])) * g.normal(size=n)
        w = np.ones(n)
        for tau in (0.1, 0.5, 0.8):
            for alpha in (0.0, 0.05):
                theta, bse = solve(x, y, w, tau, alpha)
                out.append(
                    case(
                        f"{name}/tau{tau}/alpha{alpha}",
                        "expectile",
                        [x, y],
                        {"coef": theta[:d], "intercept": np.array(float(theta[d])), "bse": bse},
                        rtol=1e-6,
                        atol=1e-8,
                        tau=tau,
                        alpha=alpha,
                    )
                )
    n = 80
    x = g.normal(size=(n, 2))
    y = x @ np.array([1.0, -2.0]) + 0.5 + g.standard_t(4, size=n)
    w = g.uniform(0.5, 3.0, size=n)
    theta, bse = solve(x, y, w, 0.9, 0.0)
    out.append(
        case(
            "weighted/tau0.9",
            "expectileWeighted",
            [x, y, w],
            {"coef": theta[:2], "intercept": np.array(float(theta[2])), "bse": bse},
            rtol=1e-6,
            atol=1e-8,
            tau=0.9,
            alpha=0.0,
        )
    )
    return out
