"""Maximum-likelihood fits against scipy.stats' fit (locations fixed at zero where the family has one), with scipy's
numerical optimiser tightened so the reference is the maximiser itself, and standard errors against central
differences of the summed scipy log-density (the observed information)."""

from collections.abc import Callable, Sequence

import numpy as np
import scipy
import scipy.stats as st
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from scipy.optimize import fmin, minimize

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def tight(func: Callable[..., float], x0: FloatArray, args: tuple[object, ...] = (), disp: int = 0) -> FloatArray:
    return np.asarray(fmin(func, x0, args=args, xtol=1e-12, ftol=1e-14, maxiter=20000, maxfun=40000, disp=0))


def standard_errors(nll: Callable[[FloatArray], float], theta: Sequence[float]) -> FloatArray:
    t = np.asarray(theta, dtype=float)
    p = t.size
    H = np.zeros((p, p))
    h = 1e-4 * np.maximum(1.0, np.abs(t))
    for i in range(p):
        for j in range(p):
            ei = np.eye(p)[i] * h[i]
            ej = np.eye(p)[j] * h[j]
            H[i, j] = (nll(t + ei + ej) - nll(t + ei - ej) - nll(t - ei + ej) + nll(t - ei - ej)) / (4 * h[i] * h[j])
    return np.sqrt(np.diag(np.linalg.inv(H)))


def cases() -> list[Case]:
    g = rng(8080)
    out: list[Case] = []
    x = g.normal(1.5, 2.0, 40)
    mu, sd = st.norm.fit(x)
    se = standard_errors(lambda t: -float(np.sum(st.norm.logpdf(x, t[0], t[1]))), [mu, sd])
    out.append(
        case(
            "normal",
            "fit",
            [x, x],
            np.asarray([mu, sd, *se, float(np.sum(st.norm.logpdf(x, mu, sd)))]),
            rtol=1e-5,
            family="normal",
        )
    )
    y = g.gamma(2.5, 1.3, 50)
    a, _, scale = st.gamma.fit(y, floc=0, optimizer=tight)
    se = standard_errors(lambda t: -float(np.sum(st.gamma.logpdf(y, t[0], scale=1 / t[1]))), [a, 1 / scale])
    out.append(
        case(
            "gamma",
            "fit",
            [y, y],
            np.asarray([a, 1 / scale, *se, float(np.sum(st.gamma.logpdf(y, a, scale=scale)))]),
            rtol=1e-6,
            family="gamma",
        )
    )
    c, _, scale = st.weibull_min.fit(y, floc=0, optimizer=tight)
    se = standard_errors(lambda t: -float(np.sum(st.weibull_min.logpdf(y, t[0], scale=t[1]))), [c, scale])
    out.append(
        case(
            "weibull",
            "fit",
            [y, y],
            np.asarray([c, scale, *se, float(np.sum(st.weibull_min.logpdf(y, c, scale=scale)))]),
            rtol=1e-6,
            family="weibull",
        )
    )
    u = g.beta(2.0, 5.0, 40)
    ab = minimize(
        lambda t: -float(np.sum(st.beta.logpdf(u, t[0], t[1]))),
        [2.0, 5.0],
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000},
    ).x
    se = standard_errors(lambda t: -float(np.sum(st.beta.logpdf(u, t[0], t[1]))), list(ab))
    out.append(
        case(
            "beta",
            "fit",
            [u, u],
            np.asarray([ab[0], ab[1], *se, float(np.sum(st.beta.logpdf(u, ab[0], ab[1])))]),
            rtol=1e-6,
            family="beta",
        )
    )
    s, _, scale = st.lognorm.fit(y, floc=0)
    out.append(case("logNormal", "fitParams", [y, y], np.asarray([np.log(scale), s]), rtol=1e-10, family="logNormal"))
    out.append(
        case("exponential", "fitParams", [y, y], np.asarray([1 / st.expon.fit(y, floc=0)[1]]), family="exponential")
    )
    loc, scale = st.laplace.fit(x)
    out.append(case("laplace", "fitParams", [x, x], np.asarray([loc, scale]), family="laplace"))
    for name, dist in [("logistic", st.logistic), ("gumbel", st.gumbel_r), ("cauchy", st.cauchy)]:
        loc, scale = dist.fit(x, optimizer=tight)
        se = standard_errors(lambda t, d=dist: -float(np.sum(d.logpdf(x, t[0], t[1]))), [loc, scale])
        out.append(
            case(
                name,
                "fit",
                [x, x],
                np.asarray([loc, scale, *se, float(np.sum(dist.logpdf(x, loc, scale)))]),
                rtol=1e-5,
                family=name,
            )
        )
    tt = g.standard_t(4, 60) * 1.5 + 0.3
    df, loc, scale = st.t.fit(tt, optimizer=tight)
    out.append(case("studentT", "fitParams", [tt, tt], np.asarray([df, loc, scale]), rtol=1e-5, family="studentT"))
    out.append(case("uniform", "fitParams", [x, x], np.asarray([x.min(), x.max()]), family="uniform"))
    pw = (g.pareto(3.0, 30) + 1) * 2.0
    b, _, scale = st.pareto.fit(pw, floc=0)
    out.append(case("pareto", "fitParams", [pw, pw], np.asarray([b, scale]), rtol=1e-6, family="pareto"))
    k = g.poisson(3.2, 40).astype(float)
    out.append(case("poisson", "fitParams", [k, k], np.asarray([k.mean()]), family="poisson"))
    bern = (g.uniform(size=40) < 0.3).astype(float)
    out.append(case("bernoulli", "fitParams", [bern, bern], np.asarray([bern.mean()]), family="bernoulli"))
    geo = g.geometric(0.25, 40).astype(float)
    out.append(case("geometric", "fitParams", [geo, geo], np.asarray([1 / geo.mean()]), family="geometric"))
    nb = g.negative_binomial(3.0, 0.4, 80).astype(float)
    res = minimize(
        lambda t: -float(np.sum(st.nbinom.logpmf(nb, np.exp(t[0]), 1 / (1 + np.exp(-t[1]))))),
        [1.0, 0.0],
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000},
    )
    out.append(
        case(
            "negativeBinomial",
            "fitParams",
            [nb, nb],
            np.asarray([np.exp(res.x[0]), 1 / (1 + np.exp(-res.x[1]))]),
            rtol=1e-6,
            family="negativeBinomial",
        )
    )
    return out
