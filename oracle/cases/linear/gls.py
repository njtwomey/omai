"""Generalised least squares against statsmodels: GLS with a vector and an AR(1) matrix covariance (coefficients,
scale, log-likelihood, standard errors), GLSAR.iterative_fit (the AR coefficients and the final whitened fit), and
heteroscedastic regression: feasible GLS as the iterated log-residual regression written with statsmodels OLS and WLS,
and the joint Gaussian likelihood of the mean and a log-linear variance by BFGS (statsmodels has no model for it)."""

import numpy as np
import numpy.typing as npt
import scipy
import statsmodels
import statsmodels.api as sm
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from scipy.linalg import toeplitz
from scipy.optimize import minimize

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__, "scipy": scipy.__version__}

type Vec = npt.NDArray[np.float64]


def fgls(x: Vec, y: Vec, z: Vec, rounds: int) -> tuple[Vec, Vec]:
    """`rounds` rounds of WLS (weights exp(-z gamma)) then OLS of the log squared residuals on z, then a final WLS."""
    n = len(y)
    a = np.column_stack([x, np.ones(n)])
    zc = np.column_stack([z, np.ones(n)])
    gamma = np.zeros(zc.shape[1])
    for _ in range(rounds):
        w = np.exp(-(zc @ gamma))
        beta = np.asarray(sm.WLS(y, a, weights=w).fit().params, dtype=np.float64)
        resid = y - a @ beta
        gamma = np.asarray(sm.OLS(np.log(np.maximum(resid**2, 1e-300)), zc).fit().params, dtype=np.float64)
    w = np.exp(-(zc @ gamma))
    beta = np.asarray(sm.WLS(y, a, weights=w).fit().params, dtype=np.float64)
    return beta, gamma


def hetero_ml(x: Vec, y: Vec, z: Vec) -> tuple[Vec, Vec, float]:
    n = len(y)
    a = np.column_stack([x, np.ones(n)])
    zc = np.column_stack([z, np.ones(n)])
    p = a.shape[1]

    def nll(theta: Vec) -> float:
        r = y - a @ theta[:p]
        lv = zc @ theta[p:]
        return float(0.5 * np.sum(lv + r * r * np.exp(-lv)) / n)

    def grad(theta: Vec) -> Vec:
        r = y - a @ theta[:p]
        lv = zc @ theta[p:]
        inv = np.exp(-lv)
        return np.concatenate([-(a.T @ (r * inv)) / n, 0.5 * (zc.T @ (1.0 - r * r * inv)) / n])

    ols = np.linalg.lstsq(a, y, rcond=None)[0]
    start = np.concatenate([ols, np.zeros(zc.shape[1] - 1), [np.log(np.mean((y - a @ ols) ** 2))]])
    res = minimize(nll, start, jac=grad, method="BFGS", options={"gtol": 1e-12, "maxiter": 10000})
    theta = np.asarray(res.x, dtype=np.float64)
    llf = -float(n * (nll(theta) + 0.5 * np.log(2 * np.pi)))
    return theta[:p], theta[p:], llf


def cases() -> list[Case]:
    g = rng(1935)
    out: list[Case] = []
    # GLS with known covariance.
    for name in ("vector", "ar1"):
        n = 30
        x = g.normal(size=(n, 2))
        exog = np.column_stack([x, np.ones(n)])
        if name == "vector":
            sigma: Vec = np.exp(g.normal(size=n))
            noise = np.sqrt(sigma) * g.normal(size=n)
        else:
            sigma = np.asarray(toeplitz(0.6 ** np.arange(n)), dtype=np.float64)
            noise = np.linalg.cholesky(sigma) @ g.normal(size=n)
        y = exog @ np.array([1.0, -2.0, 0.5]) + noise
        res = sm.GLS(y, exog, sigma=sigma).fit()
        out.append(
            case(
                f"gls/{name}",
                "gls",
                [x, y, sigma],
                {
                    "coef": np.asarray(res.params)[:2],
                    "intercept": np.array(float(np.asarray(res.params)[2])),
                    "scale": np.array(float(res.scale)),
                    "logLikelihood": np.array(float(res.llf)),
                    "bse": np.asarray(res.bse),
                },
                rtol=1e-8,
                atol=1e-9,
            )
        )
    # GLSAR with AR errors.
    for order, maxiter in [(1, 1), (1, 5), (2, 8)]:
        n = 120
        t = np.arange(n) / 10.0
        e = np.zeros(n)
        eps = g.normal(size=n)
        for i in range(n):
            e[i] = 0.6 * (e[i - 1] if i > 0 else 0.0) - 0.2 * (e[i - 2] if i > 1 else 0.0) + 0.5 * eps[i]
        x = np.column_stack([t, np.sin(t)])
        exog = np.column_stack([x, np.ones(n)])
        y = exog @ np.array([2.0, 1.0, 0.5]) + e
        model = sm.GLSAR(y, exog, rho=order)
        res = model.iterative_fit(maxiter=maxiter)
        out.append(
            case(
                f"glsar/order{order}/iter{maxiter}",
                "glsar",
                [x, y],
                {
                    "coef": np.asarray(res.params)[:2],
                    "intercept": np.array(float(np.asarray(res.params)[2])),
                    "rho": np.asarray(model.rho, dtype=np.float64),
                    "scale": np.array(float(res.scale)),
                    "bse": np.asarray(res.bse),
                },
                rtol=1e-6,
                atol=1e-8,
                order=order,
                maxIter=maxiter,
            )
        )
    # Heteroscedastic regression.
    for name, n, d in [("one", 200, 1), ("two", 300, 2)]:
        x = g.normal(size=(n, d))
        sd = 0.3 * np.exp(0.4 * x[:, 0])
        y = x @ np.linspace(1.5, -1.0, d) + 1.0 + sd * g.normal(size=n)
        for rounds in (1, 3, 8):
            beta, gamma = fgls(x, y, x, rounds)
            out.append(
                case(
                    f"heteroscedastic/fgls/{name}/rounds{rounds}",
                    "heteroscedastic",
                    [x, y],
                    {"coef": beta[:d], "intercept": np.array(float(beta[d])), "gamma": gamma},
                    rtol=1e-6,
                    atol=1e-8,
                    method="fgls",
                    rounds=rounds,
                )
            )
        beta, gamma, llf = hetero_ml(x, y, x)
        out.append(
            case(
                f"heteroscedastic/ml/{name}",
                "heteroscedastic",
                [x, y],
                {
                    "coef": beta[:d],
                    "intercept": np.array(float(beta[d])),
                    "gamma": gamma,
                    "logLikelihood": np.array(llf),
                },
                rtol=1e-5,
                atol=1e-6,
                method="ml",
            )
        )
    return out
