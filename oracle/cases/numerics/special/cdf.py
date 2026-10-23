"""Distribution functions of test statistics against mpmath (and scipy for the exact Smirnov distribution)."""

from functools import partial

import mpmath as mp
import numpy as np
import scipy
import scipy.special as sp
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.precise import Mpf, invert, mpvec, value_and_grads

REFERENCE: dict[str, str] = {"mpmath": mp.__version__, "scipy": scipy.__version__}


def stdtr(df: Mpf, t: Mpf) -> Mpf:
    x = df / (df + t * t)
    tail = mp.betainc(df / 2, mp.mpf(1) / 2, 0, x, regularized=True) / 2
    return 1 - tail if t > 0 else tail


def stdtr_upper(df: Mpf, t: Mpf) -> Mpf:
    return stdtr(df, -t)


def stdtrit(df: Mpf, p: Mpf) -> Mpf:
    return invert(partial(stdtr, df), partial(stdtr_upper, df), p, float(sp.stdtrit(float(df), float(p))))


def chdtr(k: Mpf, x: Mpf) -> Mpf:
    return mp.gammainc(k / 2, 0, x / 2, regularized=True)


def chdtrc(k: Mpf, x: Mpf) -> Mpf:
    return mp.gammainc(k / 2, x / 2, mp.inf, regularized=True)


def fdtr(a: Mpf, b: Mpf, v: Mpf) -> Mpf:
    return mp.betainc(a / 2, b / 2, 0, a * v / (a * v + b), regularized=True)


def fdtrc(a: Mpf, b: Mpf, v: Mpf) -> Mpf:
    return mp.betainc(b / 2, a / 2, 0, b / (b + a * v), regularized=True)


def kolmogorov(x: Mpf) -> Mpf:
    def term(k: Mpf) -> Mpf:
        return (-1) ** (k - 1) * mp.exp(-2 * k * k * x * x)

    return 2 * mp.nsum(term, [1, mp.inf])


def cases() -> list[Case]:
    g = rng(6)
    df = np.exp(g.uniform(np.log(0.2), np.log(1e4), 40))
    t = np.concatenate([g.uniform(-5, 5, 30), g.uniform(-1e3, 1e3, 5), g.uniform(-1e-6, 1e-6, 5)])
    out: list[Case] = []
    out += value_and_grads(
        "stdtr",
        "stdtr",
        stdtr,
        [df, t],
        wrt=[0, 1],
        rtol=1e-12,
        atol=1e-300,
        grad_rtol=1e-9,
        grad_atol=1e-300,
        dps=300,
    )
    p = np.concatenate([g.uniform(0, 1, 30), np.exp(g.uniform(np.log(1e-30), np.log(1e-4), 10))])
    out.append(case("stdtrit", "stdtrit", [df, p], mpvec(stdtrit, df, p), rtol=1e-11))
    k = np.exp(g.uniform(np.log(0.5), np.log(200), 30))
    x = k * np.exp(g.uniform(np.log(0.05), np.log(5), 30))
    out += value_and_grads("chdtr", "chdtr", chdtr, [k, x], wrt=[1], rtol=1e-12, atol=1e-300, dps=300)
    out += value_and_grads("chdtrc", "chdtrc", chdtrc, [k, x], wrt=[1], rtol=1e-12, atol=1e-300, dps=300)
    d1, d2 = np.exp(g.uniform(np.log(0.5), np.log(300), 30)), np.exp(g.uniform(np.log(0.5), np.log(300), 30))
    fx = np.exp(g.uniform(np.log(0.01), np.log(50), 30))
    out += value_and_grads("fdtr", "fdtr", fdtr, [d1, d2, fx], wrt=[2], rtol=1e-12, atol=1e-300, dps=300)
    out += value_and_grads("fdtrc", "fdtrc", fdtrc, [d1, d2, fx], wrt=[2], rtol=1e-12, atol=1e-300, dps=300)
    kx = np.exp(g.uniform(np.log(0.1), np.log(6), 30))
    out += value_and_grads(
        "kolmogorov", "kolmogorov", kolmogorov, [kx], wrt=[0], rtol=1e-13, atol=1e-300, grad_rtol=1e-12
    )
    for n in [1, 5, 20, 100, 1000]:
        d = g.uniform(0, 1, 15)
        out.append(case(f"smirnov/{n}", "smirnov", [d], sp.smirnov(n, d), rtol=1e-11, atol=1e-300, n=n))
    return out
