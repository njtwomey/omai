"""Extended-family additive models against an independent numpy/scipy implementation.

gamlss, mgcv and pyGAM are not in the oracle environment, so the reference is written here from the definitions, with
scipy's densities for the likelihoods and no shared code with the engine: the derivatives of the log-likelihood in the
linear predictors come from five-point finite differences of the density (not from automatic differentiation or hand
derivation), the penalised maximum-likelihood coefficients from Newton's method with a line search on those, the
Laplace-approximate REML score from the same penalised information, and the smoothing parameter (and the family's own
parameters, the beta precision, the negative binomial dispersion, the ordinal cut points) by a brute-force profile over
a
grid of log lambda followed by scipy's Nelder-Mead on the joint score. Each model has one smooth predictor on a
statsmodels B-spline basis with its second-derivative penalty; the other predictors are unpenalised. Two cases per
model:
the penalised fit and the score at fixed hyperparameters (coefficients, log-likelihood, effective degrees of freedom,
score), and the selected hyperparameters with the score and effective degrees of freedom there. The Cox model is the
penalised partial likelihood for untied times, written out directly."""

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

GRID = np.arange(-6.0, 14.0 + 1e-9, 1.0)
STEP = 1e-3

type LogLik = Callable[[FloatArray, list[FloatArray], FloatArray], FloatArray]
"""Per-observation log-likelihood from the response, the linear predictors and the aux parameters."""


@dataclass(frozen=True)
class Problem:
    """A penalised model in the stacked coefficients: log-likelihood, its gradient and information, the unit penalty."""

    p: int
    loglik: Callable[[FloatArray, FloatArray], float]
    gradient: Callable[[FloatArray, FloatArray], FloatArray]
    information: Callable[[FloatArray, FloatArray], FloatArray]
    penalty: FloatArray
    start: FloatArray
    aux0: FloatArray


def _stencil_gradient(f: Callable[[FloatArray], FloatArray], x: FloatArray) -> FloatArray:
    """Five-point derivative of a vectorised scalar function, observation by observation."""
    h = STEP
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)


def eta_problem(
    ll: LogLik,
    y: FloatArray,
    designs: list[FloatArray],
    penalty: FloatArray,
    start: FloatArray,
    aux0: FloatArray,
) -> Problem:
    """Chain rule from per-observation derivatives (five-point differences) to the stacked coefficients."""
    widths = [d.shape[1] for d in designs]
    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(int)
    k_count = len(designs)

    def split(beta: FloatArray) -> list[FloatArray]:
        return [designs[k] @ beta[offsets[k] : offsets[k + 1]] for k in range(k_count)]

    def total(beta: FloatArray, aux: FloatArray) -> float:
        return float(np.sum(ll(y, split(beta), aux)))

    def partial(etas: list[FloatArray], aux: FloatArray, k: int) -> FloatArray:
        def along(t: FloatArray) -> FloatArray:
            moved = [t if j == k else etas[j] for j in range(k_count)]
            return ll(y, moved, aux)

        return _stencil_gradient(along, etas[k])

    def gradient(beta: FloatArray, aux: FloatArray) -> FloatArray:
        etas = split(beta)
        out = np.zeros(offsets[-1])
        for k in range(k_count):
            out[offsets[k] : offsets[k + 1]] = designs[k].T @ partial(etas, aux, k)
        return out

    def information(beta: FloatArray, aux: FloatArray) -> FloatArray:
        etas = split(beta)
        info = np.zeros((offsets[-1], offsets[-1]))
        for k in range(k_count):
            for m in range(k, k_count):

                def g_k(t: FloatArray, k: int = k, m: int = m) -> FloatArray:
                    moved = [t if j == m else etas[j] for j in range(k_count)]
                    return partial(moved, aux, k)

                d2 = _stencil_gradient(g_k, etas[m])
                block = -(designs[k].T * d2) @ designs[m]
                info[offsets[k] : offsets[k + 1], offsets[m] : offsets[m + 1]] = block
                if m != k:
                    info[offsets[m] : offsets[m + 1], offsets[k] : offsets[k + 1]] = block.T
        return info

    return Problem(int(offsets[-1]), total, gradient, information, penalty, start, aux0)


def cox_problem(time: FloatArray, event: FloatArray, design: FloatArray, penalty: FloatArray) -> Problem:
    """The partial likelihood for untied times (Breslow and Efron coincide), with its score and information."""
    order = np.argsort(time)
    d, x = event[order], design[order]
    p = x.shape[1]

    def loglik(beta: FloatArray, aux: FloatArray) -> float:
        eta = x @ beta
        shift = float(eta.max())
        w = np.exp(eta - shift)
        risk = np.cumsum(w[::-1])[::-1]
        return float(np.sum(d * (eta - shift - np.log(risk))))

    def moments(beta: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        w = np.exp(x @ beta - float((x @ beta).max()))
        s0 = np.cumsum(w[::-1])[::-1]
        s1 = np.cumsum((w[:, None] * x)[::-1], axis=0)[::-1]
        s2 = np.cumsum((w[:, None, None] * (x[:, :, None] * x[:, None, :]))[::-1], axis=0)[::-1]
        return s0, s1, s2

    def gradient(beta: FloatArray, aux: FloatArray) -> FloatArray:
        s0, s1, _ = moments(beta)
        return np.asarray(np.sum(d[:, None] * (x - s1 / s0[:, None]), axis=0), dtype=np.float64)

    def information(beta: FloatArray, aux: FloatArray) -> FloatArray:
        s0, s1, s2 = moments(beta)
        mean = s1 / s0[:, None]
        var = s2 / s0[:, None, None] - mean[:, :, None] * mean[:, None, :]
        return np.asarray(np.sum(d[:, None, None] * var, axis=0), dtype=np.float64)

    return Problem(p, loglik, gradient, information, penalty, np.zeros(p), np.zeros(0))


def penalised_fit(prob: Problem, lam: float, aux: FloatArray, beta0: FloatArray) -> FloatArray:
    """Newton's method with a line search on the penalised log-likelihood; negative eigenvalues are flipped."""
    beta = beta0.copy()

    def objective(b: FloatArray) -> float:
        value = prob.loglik(b, aux)
        return float(-value + 0.5 * lam * b @ prob.penalty @ b) if np.isfinite(value) else np.inf

    f = objective(beta)
    for _ in range(200):
        g = prob.gradient(beta, aux) - lam * prob.penalty @ beta
        a_mat = prob.information(beta, aux) + lam * prob.penalty
        w, v = np.linalg.eigh(a_mat)
        w = np.maximum(np.abs(w), 1e-7 * float(np.abs(w).max()))
        delta = v @ ((v.T @ g) / w)
        decrement = float(g @ delta)
        step = 1.0
        trial, f_trial = beta, f
        while step > 1e-12:
            trial = beta + step * delta
            f_trial = objective(trial)
            if np.isfinite(f_trial) and f_trial <= f - 1e-4 * step * decrement + 1e-14 * abs(f):
                break
            step /= 2
        if step <= 1e-12:
            break
        beta, f_old, f = trial, f, f_trial
        if decrement < 1e-14 * (1 + abs(f)) and abs(f_old - f) < 1e-14 * (1 + abs(f)):
            break
    return beta


def score(
    prob: Problem, rank: int, log_pdet_unit: float, log_lam: float, aux: FloatArray, beta0: FloatArray
) -> tuple[float, FloatArray, float, float]:
    """The Laplace-approximate REML score, the coefficients, the log-likelihood and the effective degrees of freedom."""
    lam = float(np.exp(log_lam))
    beta = penalised_fit(prob, lam, aux, beta0)
    info = prob.information(beta, aux)
    h = info + lam * prob.penalty
    w, v = np.linalg.eigh(h)
    w = np.maximum(np.abs(w), 1e-7 * float(np.abs(w).max()))
    log_det = float(np.sum(np.log(w)))
    edf = float(np.trace((v / w) @ v.T @ info))
    ll = prob.loglik(beta, aux)
    null_dim = prob.p - rank
    value = (
        -ll
        + 0.5 * lam * float(beta @ prob.penalty @ beta)
        + 0.5 * log_det
        - 0.5 * (rank * log_lam + log_pdet_unit)
        - 0.5 * null_dim * np.log(2 * np.pi)
    )
    return float(value), beta, ll, edf


def reference(prob: Problem, log_lam: float, aux: FloatArray) -> dict[str, FloatArray]:
    """The penalised fit and score at fixed hyperparameters."""
    eig = np.linalg.eigvalsh(prob.penalty)
    keep = eig > 1e-9 * eig.max()
    rank, log_pdet = int(keep.sum()), float(np.sum(np.log(eig[keep])))
    value, beta, ll, edf = score(prob, rank, log_pdet, log_lam, aux, prob.start)
    return {
        "coef": beta,
        "logLik": np.asarray([ll]),
        "edf": np.asarray([edf]),
        "criterion": np.asarray([value]),
    }


def selected(prob: Problem) -> dict[str, FloatArray]:
    """The hyperparameters minimising the score: a grid of log lambda at the start aux values, then Nelder-Mead."""
    eig = np.linalg.eigvalsh(prob.penalty)
    keep = eig > 1e-9 * eig.max()
    rank, log_pdet = int(keep.sum()), float(np.sum(np.log(eig[keep])))
    values = [score(prob, rank, log_pdet, float(g), prob.aux0, prob.start)[0] for g in GRID]
    x0 = np.concatenate([[GRID[int(np.argmin(values))]], prob.aux0])

    def f(theta: FloatArray) -> float:
        return score(prob, rank, log_pdet, float(theta[0]), theta[1:], prob.start)[0]

    best = scipy.optimize.minimize(
        f, x0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-11, "maxiter": 2000, "maxfev": 4000}
    )
    value, _, ll, edf = score(prob, rank, log_pdet, float(best.x[0]), best.x[1:], prob.start)
    return {
        "logLambda": np.asarray([best.x[0]]),
        "aux": np.asarray(best.x[1:], dtype=np.float64),
        "criterion": np.asarray([value]),
        "edf": np.asarray([edf]),
        "logLik": np.asarray([ll]),
    }


def _ll_beta(y: FloatArray, eta: list[FloatArray], aux: FloatArray) -> FloatArray:
    mu, phi = scipy.special.expit(eta[0]), np.exp(aux[0])
    return np.asarray(scipy.stats.beta.logpdf(y, mu * phi, (1 - mu) * phi), dtype=np.float64)


def _ll_nb(y: FloatArray, eta: list[FloatArray], aux: FloatArray) -> FloatArray:
    mu, theta = np.exp(eta[0]), np.exp(aux[0])
    return np.asarray(scipy.stats.nbinom.logpmf(y, theta, theta / (theta + mu)), dtype=np.float64)


def _ll_gaulss(y: FloatArray, eta: list[FloatArray], aux: FloatArray) -> FloatArray:
    return np.asarray(scipy.stats.norm.logpdf(y, eta[0], np.exp(eta[1])), dtype=np.float64)


def _ll_zip(y: FloatArray, eta: list[FloatArray], aux: FloatArray) -> FloatArray:
    lam, pi = np.exp(eta[0]), scipy.special.expit(eta[1])
    zero = np.log(pi + (1 - pi) * np.exp(-lam))
    pos = np.log1p(-pi) + scipy.stats.poisson.logpmf(y, lam)
    return np.asarray(np.where(y == 0, zero, pos), dtype=np.float64)


def _ll_ocat(y: FloatArray, eta: list[FloatArray], aux: FloatArray) -> FloatArray:
    cuts = np.concatenate([[-1.0], -1.0 + np.cumsum(np.exp(aux))])
    k = 4
    upper = np.where(y >= k - 1, np.inf, cuts[np.minimum(y.astype(int), len(cuts) - 1)])
    lower = np.where(y <= 0, -np.inf, cuts[np.maximum(y.astype(int) - 1, 0)])
    return np.log(scipy.special.expit(upper - eta[0]) - scipy.special.expit(lower - eta[0]))


def _ll_multinomial(y: FloatArray, eta: list[FloatArray], aux: FloatArray) -> FloatArray:
    logits = np.column_stack([np.zeros_like(eta[0]), *eta])
    return np.asarray(
        logits[np.arange(len(y)), y.astype(int)] - scipy.special.logsumexp(logits, axis=1), dtype=np.float64
    )


def _ll_gamma(y: FloatArray, eta: list[FloatArray], aux: FloatArray) -> FloatArray:
    mu, s = np.exp(eta[0]), np.exp(eta[1])
    return np.asarray(scipy.stats.gamma.logpdf(y, a=1 / s**2, scale=mu * s**2), dtype=np.float64)


def _ll_student(y: FloatArray, eta: list[FloatArray], aux: FloatArray) -> FloatArray:
    return np.asarray(scipy.stats.t.logpdf(y, np.exp(eta[2]), eta[0], np.exp(eta[1])), dtype=np.float64)


def _basis(x: FloatArray, df: int) -> tuple[FloatArray, FloatArray]:
    """The intercept and B-spline design and the unit penalty over the spline columns (zero on the intercept)."""
    smoother = BSplines(pd.DataFrame({"x": x}), df=[df], degree=[3])
    design = np.column_stack([np.ones(len(x)), np.asarray(smoother.basis, dtype=np.float64)])
    s0 = np.zeros((design.shape[1], design.shape[1]))
    s0[1:, 1:] = np.asarray(smoother.penalty_matrices[0], dtype=np.float64)
    return design, s0


def _embed(s0: FloatArray, total: int) -> FloatArray:
    out = np.zeros((total, total))
    out[: s0.shape[0], : s0.shape[0]] = s0
    return out


def _linear(x: FloatArray) -> FloatArray:
    return np.column_stack([np.ones(len(x)), x - x.mean()])


def _start(constants: list[float], designs: list[FloatArray]) -> FloatArray:
    parts = []
    for c, d in zip(constants, designs, strict=True):
        b = np.zeros(d.shape[1])
        b[0] = c
        parts.append(b)
    return np.concatenate(parts)


def record(
    name: str,
    family: str,
    ll: LogLik,
    y: FloatArray,
    designs: list[FloatArray],
    s0: FloatArray,
    constants: list[float],
    aux0: FloatArray,
    log_lam: float,
    aux: FloatArray,
) -> list[Case]:
    total = sum(d.shape[1] for d in designs)
    prob = eta_problem(ll, y, designs, _embed(s0, total), _start(constants, designs), aux0)
    inputs = [y, *designs, prob.penalty]
    fixed = reference(prob, log_lam, aux)
    chosen = selected(prob)
    return [
        case(
            f"{name}-fixed",
            "extendedFixed",
            inputs,
            fixed,
            rtol=1e-5,
            atol=1e-6,
            family=family,
            logLambda=log_lam,
            aux=aux,
            aux0=aux0,
            constants=np.asarray(constants),
        ),
        case(
            f"{name}-selected",
            "extendedSelected",
            inputs,
            chosen,
            rtol=1e-2,
            atol=2e-2,
            family=family,
            aux0=aux0,
            constants=np.asarray(constants),
        ),
    ]


def cases() -> list[Case]:
    g = rng(2024)
    out: list[Case] = []
    n = 90
    x = np.sort(g.uniform(0, 1, n))
    spline, s0 = _basis(x, 7)
    lin = _linear(x)

    mu = scipy.special.expit(-0.2 + 1.3 * np.sin(2 * np.pi * x))
    y = np.clip(g.beta(mu * 18, (1 - mu) * 18), 1e-4, 1 - 1e-4)
    out += record(
        "beta",
        "beta",
        _ll_beta,
        y,
        [spline],
        s0,
        [float(scipy.special.logit(y.mean()))],
        np.asarray([np.log(10.0)]),
        2.0,
        np.asarray([np.log(15.0)]),
    )

    lam_nb = np.exp(1.0 + 0.8 * np.sin(2 * np.pi * x))
    y = g.negative_binomial(3.0, 3.0 / (3.0 + lam_nb)).astype(np.float64)
    out += record(
        "negative-binomial",
        "negativeBinomial",
        _ll_nb,
        y,
        [spline],
        s0,
        [float(np.log(y.mean()))],
        np.asarray([np.log(2.0)]),
        2.0,
        np.asarray([np.log(3.0)]),
    )

    y = np.sin(2 * np.pi * x) + np.exp(-1.0 + 1.0 * x) * g.normal(size=n)
    out += record(
        "gaussian-location-scale",
        "gaussianLocationScale",
        _ll_gaulss,
        y,
        [spline, lin],
        s0,
        [float(y.mean()), float(np.log(y.std()))],
        np.zeros(0),
        1.0,
        np.zeros(0),
    )

    count = g.poisson(np.exp(0.6 + 0.8 * np.sin(2 * np.pi * x)))
    infl = g.uniform(size=n) < scipy.special.expit(-0.8 + 2.0 * x)
    y = np.where(infl, 0, count).astype(np.float64)
    out += record(
        "zero-inflated-poisson",
        "zeroInflatedPoisson",
        _ll_zip,
        y,
        [spline, lin],
        s0,
        [float(np.log(max(y[y > 0].mean(), 0.5))), -1.0],
        np.zeros(0),
        2.0,
        np.zeros(0),
    )

    cuts = np.asarray([-1.0, 0.4, 1.9])
    latent = 1.4 * np.sin(2 * np.pi * x) + g.logistic(size=n)
    y = (latent[:, None] > cuts[None, :]).sum(axis=1).astype(np.float64)
    out += record(
        "ordered-categorical",
        "orderedCategorical",
        _ll_ocat,
        y,
        [spline],
        s0,
        [0.0],
        np.asarray([np.log(1.4), np.log(1.5)]),
        2.0,
        np.asarray([np.log(1.4), np.log(1.5)]),
    )

    p1 = np.exp(1.5 * np.sin(2 * np.pi * x))
    p2 = np.exp(1.0 - 2.0 * x)
    pr = np.column_stack([np.ones(n), p1, p2])
    pr = pr / pr.sum(axis=1, keepdims=True)
    u = g.uniform(size=n)
    y = (u[:, None] > np.cumsum(pr, axis=1)).sum(axis=1).astype(np.float64)
    out += record(
        "multinomial", "multinomial", _ll_multinomial, y, [spline, lin], s0, [0.0, 0.0], np.zeros(0), 2.0, np.zeros(0)
    )

    mu_g = np.exp(1.0 + 0.7 * np.sin(2 * np.pi * x))
    sg = np.exp(-0.9 + 0.8 * x)
    y = g.gamma(1 / sg**2, mu_g * sg**2)
    out += record(
        "gamlss-gamma",
        "gamlssGamma",
        _ll_gamma,
        y,
        [spline, lin],
        s0,
        [float(np.log(y.mean())), -0.5],
        np.zeros(0),
        2.0,
        np.zeros(0),
    )

    # The degrees of freedom are only identified with enough heavy-tailed data and a spline flexible enough not to
    # mistake its own misfit for Gaussian noise: 200 points of a t with 2 degrees of freedom, a gentle penalty.
    nt = 200
    xt = np.sort(g.uniform(0, 1, nt))
    spline_t, s0_t = _basis(xt, 7)
    y = np.sin(2 * np.pi * xt) + 0.3 * g.standard_t(2.0, size=nt)
    out += record(
        "gamlss-student-t",
        "gamlssStudentT",
        _ll_student,
        y,
        [spline_t, np.ones((nt, 1)), np.ones((nt, 1))],
        s0_t,
        [float(y.mean()), -1.0, float(np.log(4.0))],
        np.zeros(0),
        -1.0,
        np.zeros(0),
    )

    # Cox: untied times, log hazard sin, uniform censoring; no intercept (the baseline absorbs it).
    nc = 120
    xc = np.sort(g.uniform(0, 1, nc))
    spline_c, s0_c = _basis(xc, 6)
    design_c, pen_c = spline_c[:, 1:], s0_c[1:, 1:]
    raw = g.exponential(1.0, nc) / (0.4 * np.exp(1.1 * np.sin(2 * np.pi * xc)))
    cens = 0.5 + 6 * g.uniform(size=nc)
    time, event = np.minimum(raw, cens), (raw <= cens).astype(np.float64)
    prob = cox_problem(time, event, design_c, pen_c)
    inputs = [time, event, design_c, pen_c]
    out.append(
        case(
            "cox-fixed",
            "extendedCoxFixed",
            inputs,
            reference(prob, 1.5, np.zeros(0)),
            rtol=1e-5,
            atol=1e-6,
            logLambda=1.5,
        )
    )
    chosen = selected(prob)
    del chosen["aux"]
    out.append(case("cox-selected", "extendedCoxSelected", inputs, chosen, rtol=1e-2, atol=2e-2))
    return out
