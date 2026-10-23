"""Normal-distribution functions against mpmath."""

import mpmath as mp
import numpy as np
import scipy
import scipy.special as sp
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.precise import Mpf, invert, mpvec, value_and_grads

REFERENCE: dict[str, str] = {"mpmath": mp.__version__, "scipy": scipy.__version__}


def upper(x: Mpf) -> Mpf:
    return mp.ncdf(-x)


def ndtri(p: Mpf) -> Mpf:
    return invert(mp.ncdf, upper, p, float(sp.ndtri(float(p))))


def log_ndtr(x: Mpf) -> Mpf:
    return mp.log(mp.ncdf(x))


def d_owens_h(h: Mpf, a: Mpf) -> Mpf:
    return -mp.npdf(h) * mp.erf(a * h / mp.sqrt(2)) / 2


def d_owens_a(h: Mpf, a: Mpf) -> Mpf:
    return mp.exp(-h * h * (1 + a * a) / 2) / (2 * mp.pi * (1 + a * a))


def log_pdf(x: Mpf) -> Mpf:
    return mp.log(mp.npdf(x))


def owens_t(h: Mpf, a: Mpf) -> Mpf:
    def integrand(t: Mpf) -> Mpf:
        return mp.exp(-h * h * (1 + t * t) / 2) / (1 + t * t)

    return mp.quad(integrand, [0, a]) / (2 * mp.pi)


def log_interval(a: Mpf, b: Mpf) -> Mpf:
    return mp.log(mp.ncdf(b) - mp.ncdf(a))


def cases() -> list[Case]:
    g = rng(2)
    x = np.concatenate([g.uniform(-8, 8, 30), g.uniform(-38, -8, 10), g.uniform(8, 38, 5)])
    out: list[Case] = []
    # Values near 1 need many digits (log(1 - 1e-265) is 0 at 50); 400 resolve every float64 tail.
    out += value_and_grads("ndtr", "ndtr", mp.ncdf, [x], wrt=[0], atol=1e-300, dps=400)
    out += value_and_grads("logNdtr", "logNdtr", log_ndtr, [x], wrt=[0], dps=400)
    p = np.concatenate([g.uniform(0, 1, 30), np.exp(g.uniform(np.log(1e-300), np.log(1e-4), 10))])
    out.append(case("ndtri", "ndtri", [p], mpvec(ndtri, p), rtol=1e-13, atol=1e-16))
    out += value_and_grads("normalPdf", "normalPdf", mp.npdf, [x], wrt=[0], atol=1e-300)
    out += value_and_grads("normalLogPdf", "normalLogPdf", log_pdf, [x], wrt=[0])
    h = np.concatenate([g.uniform(-4, 4, 30), g.uniform(-12, 12, 10)])
    a = np.concatenate([g.uniform(-3, 3, 30), g.uniform(-40, 40, 10)])
    out += value_and_grads(
        "owensT",
        "owensT",
        owens_t,
        [h, a],
        wrt=[0, 1],
        atol=1e-300,
        grad_rtol=1e-12,
        grad_atol=1e-300,
        exact={0: d_owens_h, 1: d_owens_a},
    )
    lo = np.concatenate([g.uniform(-6, 0, 15), g.uniform(0, 30, 15), g.uniform(-30, -10, 10)])
    hi = lo + np.exp(g.uniform(np.log(1e-6), np.log(5), 40))
    # Narrow intervals lose about log10(1 / width) digits to the difference of log-CDFs (documented on the function).
    out += value_and_grads(
        "logIntervalProbability",
        "logIntervalProbability",
        log_interval,
        [lo, hi],
        wrt=[0, 1],
        rtol=1e-10,
        grad_rtol=1e-8,
        dps=400,
    )
    return out
