"""Outer-iteration smoothing-parameter selection against an independent numpy/scipy implementation of the same model.

mgcv and pyGAM are not in the oracle environment and statsmodels' GLMGam has no smoothing-parameter selection, so the
reference is written here from the definitions: penalised IRLS to convergence at each trial log lambda, the
Laplace-approximate REML score (observed information from the second derivative of the log-likelihood in the linear
predictor, the scale profiled out for the gamma family), a brute-force profile of the score over a grid of log lambda,
and scipy's bounded scalar minimiser to refine the grid's best point. The design and penalty are statsmodels' B-spline
smooth."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special
import scipy.stats
import statsmodels
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from statsmodels.gam.api import BSplines

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__, "scipy": scipy.__version__}

GRID = np.arange(-14.0, 16.0 + 1e-9, 0.5)


@dataclass(frozen=True)
class Family:
    mean: Callable[[FloatArray], FloatArray]
    fisher: Callable[[FloatArray], FloatArray]
    observed: Callable[[FloatArray, FloatArray], FloatArray]
    half_dev: Callable[[FloatArray, FloatArray], float]
    loglik: Callable[[FloatArray, FloatArray, float], float]


def _family(name: str) -> Family:
    """Mean, Fisher weight, observed weight (minus the second derivative of the half deviance in eta), half deviance and
    log-likelihood of a family with its canonical or log link, written directly in the linear predictor."""
    if name == "poisson":

        def mean(eta: FloatArray) -> FloatArray:
            return np.exp(eta)

        def fisher(mu: FloatArray) -> FloatArray:
            return mu

        def observed(y: FloatArray, mu: FloatArray) -> FloatArray:
            return mu

        def half_dev(y: FloatArray, mu: FloatArray) -> float:
            return float(np.sum(scipy.special.xlogy(y, y / mu) - (y - mu)))

        def loglik(y: FloatArray, mu: FloatArray, phi: float) -> float:
            return float(np.sum(scipy.stats.poisson.logpmf(y, mu)))

        return Family(mean, fisher, observed, half_dev, loglik)
    if name == "binomial":

        def mean(eta: FloatArray) -> FloatArray:
            return scipy.special.expit(eta)

        def fisher(mu: FloatArray) -> FloatArray:
            return mu * (1 - mu)

        def observed(y: FloatArray, mu: FloatArray) -> FloatArray:
            return mu * (1 - mu)

        def half_dev(y: FloatArray, mu: FloatArray) -> float:
            return float(np.sum(scipy.special.xlogy(y, y / mu) + scipy.special.xlogy(1 - y, (1 - y) / (1 - mu))))

        def loglik(y: FloatArray, mu: FloatArray, phi: float) -> float:
            return float(np.sum(scipy.stats.bernoulli.logpmf(y, mu)))

        return Family(mean, fisher, observed, half_dev, loglik)
    if name == "gamma-log":

        def mean(eta: FloatArray) -> FloatArray:
            return np.exp(eta)

        def fisher(mu: FloatArray) -> FloatArray:
            return np.ones_like(mu)

        def observed(y: FloatArray, mu: FloatArray) -> FloatArray:
            # d/d eta of the half deviance (y - mu)/mu - log(y/mu) is 1 - y/mu; minus its second derivative is y/mu.
            return y / mu

        def half_dev(y: FloatArray, mu: FloatArray) -> float:
            return float(np.sum((y - mu) / mu - np.log(y / mu)))

        def loglik(y: FloatArray, mu: FloatArray, phi: float) -> float:
            return float(np.sum(scipy.stats.gamma.logpdf(y, a=1.0 / phi, scale=mu * phi)))

        return Family(mean, fisher, observed, half_dev, loglik)
    if name == "gaussian":

        def mean(eta: FloatArray) -> FloatArray:
            return eta

        def fisher(mu: FloatArray) -> FloatArray:
            return np.ones_like(mu)

        def observed(y: FloatArray, mu: FloatArray) -> FloatArray:
            return np.ones_like(mu)

        def half_dev(y: FloatArray, mu: FloatArray) -> float:
            return float(0.5 * np.sum((y - mu) ** 2))

        def loglik(y: FloatArray, mu: FloatArray, phi: float) -> float:
            return float(np.sum(scipy.stats.norm.logpdf(y, mu, np.sqrt(phi))))

        return Family(mean, fisher, observed, half_dev, loglik)
    raise ValueError(name)


def _link_derivative_start(name: str, y: FloatArray) -> FloatArray:
    if name == "binomial":
        return scipy.special.logit((y + 0.5) / 2.0)
    if name == "gaussian":
        return y.copy()
    return np.log(np.maximum(y, 0.1)) if name != "poisson" else np.log(y + 0.5)


def reference(name: str, X: FloatArray, S0: FloatArray, y: FloatArray, known_scale: bool) -> dict[str, FloatArray]:
    fam = _family(name)
    mean, fisher, observed, half_dev, loglik = fam.mean, fam.fisher, fam.observed, fam.half_dev, fam.loglik
    n, p = X.shape
    pen_eigs = np.linalg.eigvalsh(S0)
    rank = int(np.sum(pen_eigs > 1e-9 * pen_eigs.max()))
    log_pdet_unit = float(np.sum(np.log(pen_eigs[pen_eigs > 1e-9 * pen_eigs.max()])))
    null_dim = p - rank
    start = _link_derivative_start(name, y)

    def fit(log_lam: float, eta0: FloatArray) -> tuple[FloatArray, FloatArray]:
        lam = np.exp(log_lam)
        eta = eta0.copy()
        pdev_old = np.inf
        beta = np.zeros(p)
        for _ in range(500):
            mu = mean(eta)
            w = fisher(mu)
            # Working response: z = eta + (y - mu) / (d mu / d eta), with d mu / d eta = w for these canonical/log links
            # (Poisson, binomial, gamma-log: w = 1 and d mu / d eta = mu) written out per family below.
            if name == "gamma-log":
                z = eta + (y - mu) / mu
            elif name == "gaussian":
                z = y
            else:
                z = eta + (y - mu) / w
            A = X.T @ (w[:, None] * X) + lam * S0
            beta = np.linalg.solve(A, X.T @ (w * z))
            eta = X @ beta
            pdev = 2 * half_dev(y, mean(eta)) + lam * float(beta @ S0 @ beta)
            if abs(pdev - pdev_old) < 1e-13 * (abs(pdev) + 1):
                break
            pdev_old = pdev
        return beta, eta

    def score(log_lam: float, eta0: FloatArray) -> tuple[float, FloatArray, FloatArray]:
        lam = np.exp(log_lam)
        beta, eta = fit(log_lam, eta0)
        mu = mean(eta)
        H = X.T @ (observed(y, mu)[:, None] * X) + lam * S0
        _, logdet_h = np.linalg.slogdet(H)
        pen = lam * float(beta @ S0 @ beta)
        log_pdet = rank * log_lam + log_pdet_unit
        if known_scale:
            phi = 1.0
        else:
            guess = max((2 * half_dev(y, mu) + pen) / max(n - null_dim, 1), 1e-12)

            def f(lphi: float) -> float:
                phi_ = float(np.exp(lphi))
                return -loglik(y, mu, phi_) + pen / (2 * phi_) - 0.5 * null_dim * lphi

            lphi = scipy.optimize.minimize_scalar(
                f, bounds=(np.log(guess) - 8, np.log(guess) + 8), method="bounded", options={"xatol": 1e-10}
            ).x
            phi = float(np.exp(lphi))
        value = (
            -loglik(y, mu, phi)
            + pen / (2 * phi)
            + 0.5 * logdet_h
            - 0.5 * log_pdet
            - 0.5 * null_dim * (np.log(phi) if not known_scale else 0.0)
            - 0.5 * null_dim * np.log(2 * np.pi)
        )
        return float(value), beta, eta

    eta = start
    values = []
    for ll in GRID:
        v, _, _ = score(float(ll), eta)
        values.append(v)
    best = int(np.argmin(values))
    lo, hi = GRID[max(best - 1, 0)], GRID[min(best + 1, len(GRID) - 1)]
    refined = scipy.optimize.minimize_scalar(
        lambda t: score(float(t), eta)[0], bounds=(float(lo), float(hi)), method="bounded", options={"xatol": 1e-9}
    )
    log_lam = float(refined.x)
    value, beta, eta_hat = score(log_lam, eta)
    mu = mean(eta_hat)
    lam = np.exp(log_lam)
    Hf = X.T @ (fisher(mu)[:, None] * X) + lam * S0
    edf = float(np.trace(np.linalg.solve(Hf, X.T @ (fisher(mu)[:, None] * X))))
    return {
        "logLambda": np.asarray([log_lam]),
        "criterion": np.asarray([value]),
        "edf": np.asarray([edf]),
        "coef": beta,
        "rank": np.asarray([float(rank)]),
    }


def record(name: str, family: str, y: FloatArray, x: FloatArray, df: int, known_scale: bool) -> Case:
    frame = pd.DataFrame({"x": x})
    smoother = BSplines(frame, df=[df], degree=[3])
    design = np.column_stack([np.ones(len(y)), np.asarray(smoother.basis, dtype=np.float64)])
    s0 = np.zeros((design.shape[1], design.shape[1]))
    s0[1:, 1:] = np.asarray(smoother.penalty_matrices[0], dtype=np.float64)
    out = reference(family, design, s0, y, known_scale)
    rank = float(out.pop("rank")[0])
    return case(
        name,
        "outerReml",
        [design, s0, y],
        out,
        rtol=2e-3,
        atol=5e-3,
        family=family,
        rank=rank,
        knownScale=1.0 if known_scale else 0.0,
    )


def cases() -> list[Case]:
    g = rng(717)
    n = 100
    x = np.sort(g.uniform(0, 1, n))
    out: list[Case] = []
    out.append(record("gaussian", "gaussian", np.sin(6 * x) + 0.25 * g.normal(size=n), x, 9, False))
    out.append(record("poisson", "poisson", g.poisson(np.exp(1.0 + np.sin(5 * x))).astype(np.float64), x, 9, True))
    # A binary response needs more data to identify a bend: 400 points of a strong sine.
    xb = np.sort(g.uniform(0, 1, 400))
    prob = 1.0 / (1.0 + np.exp(-(4.0 * np.sin(2 * np.pi * xb))))
    out.append(record("binomial", "binomial", (g.uniform(size=400) < prob).astype(np.float64), xb, 9, True))
    out.append(
        record("gamma-log", "gamma-log", g.gamma(shape=5.0, scale=np.exp(0.5 + np.sin(4 * x)) / 5.0), x, 8, False)
    )
    return out
