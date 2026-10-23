"""Incomplete gamma and beta functions and their inverses against mpmath, including shape-parameter derivatives."""

from functools import partial

import mpmath as mp
import numpy as np
import scipy
import scipy.special as sp
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.precise import F64, Mpf, invert, mpvec, value_and_grads

REFERENCE: dict[str, str] = {"mpmath": mp.__version__, "scipy": scipy.__version__}


def p_(a: Mpf, x: Mpf) -> Mpf:
    return mp.gammainc(a, 0, x, regularized=True)


def q_(a: Mpf, x: Mpf) -> Mpf:
    return mp.gammainc(a, x, mp.inf, regularized=True)


def i_(a: Mpf, b: Mpf, x: Mpf) -> Mpf:
    return mp.betainc(a, b, 0, x, regularized=True)


def gamma_density(a: Mpf, x: Mpf) -> Mpf:
    return x ** (a - 1) * mp.exp(-x) / mp.gamma(a)


def neg_gamma_density(a: Mpf, x: Mpf) -> Mpf:
    return -gamma_density(a, x)


def beta_density(a: Mpf, b: Mpf, x: Mpf) -> Mpf:
    return x ** (a - 1) * (1 - x) ** (b - 1) / mp.beta(a, b)


def log_p(a: Mpf, x: Mpf) -> Mpf:
    return mp.log(p_(a, x))


def log_q(a: Mpf, x: Mpf) -> Mpf:
    return mp.log(q_(a, x))


def gamma_start(a: Mpf, p: Mpf, x0: float) -> Mpf:
    # P(a, x) ~ x^a / Gamma(a + 1) for small x, when scipy's estimate underflows.
    return mp.mpf(x0) if x0 > 0 else mp.exp((mp.log(p) + mp.loggamma(a + 1)) / a)


def gammaincinv(a: Mpf, p: Mpf) -> Mpf:
    x0 = gamma_start(a, p, float(sp.gammaincinv(float(a), float(p))))
    return invert(partial(p_, a), partial(q_, a), p, x0, "positive")


def gammainccinv(a: Mpf, q: Mpf) -> Mpf:
    x0 = gamma_start(a, 1 - q, float(sp.gammainccinv(float(a), float(q))))
    return invert(partial(q_, a), partial(p_, a), q, x0, "positive")


def lower_beta_inverse(a: Mpf, b: Mpf, p: Mpf) -> Mpf:
    """The x with I_x(a, b) = p, for answers below 1/2."""
    x0 = float(sp.betaincinv(float(a), float(b), float(p)))
    # I_x(a, b) ~ x^a / (a B(a, b)) for small x, when scipy's estimate underflows.
    # (Otherwise the answer is below 1/2 by construction.)
    start = mp.exp((mp.log(p) + mp.log(a) + mp.log(mp.beta(a, b))) / a) if x0 <= 0 else mp.mpf(min(x0, 0.5))

    def upper(x: Mpf) -> Mpf:
        return i_(b, a, 1 - x)

    return invert(partial(i_, a, b), upper, p, start, "unit")


def betaincinv(a: Mpf, b: Mpf, p: Mpf) -> Mpf:
    # Answers above 1/2 are solved for 1 - x = I^{-1}(b, a, 1 - p), which can be far below float64 resolution near 1.
    if p > i_(a, b, mp.mpf(1) / 2):
        return 1 - lower_beta_inverse(b, a, 1 - p)
    return lower_beta_inverse(a, b, p)


def gamma_shape_grad(a: Mpf, x: Mpf) -> Mpf:
    """dP/da from whichever tail mpmath resolves (the other is 1 minus a number near 1)."""

    def lower(t: Mpf) -> Mpf:
        return p_(t, x)

    def upper(t: Mpf) -> Mpf:
        return q_(t, x)

    return mp.diff(lower, a) if x <= a else -mp.diff(upper, a)


def beta_shape_grad(which: int, a: Mpf, b: Mpf, x: Mpf) -> Mpf:
    """dI/da (which = 0) or dI/db (which = 1), from the tail on the side of x where the fraction converges."""

    def in_a(t: Mpf) -> Mpf:
        return i_(t, b, x)

    def in_b(t: Mpf) -> Mpf:
        return i_(a, t, x)

    # The upper tail 1 - I_x(a, b) = I_{1-x}(b, a), differentiated where I is near 1.
    def upper_in_a(t: Mpf) -> Mpf:
        return i_(b, t, 1 - x)

    def upper_in_b(t: Mpf) -> Mpf:
        return i_(t, a, 1 - x)

    if x < (a + 1) / (a + b + 2):
        return mp.diff(in_a, a) if which == 0 else mp.diff(in_b, b)
    return -mp.diff(upper_in_a, a) if which == 0 else -mp.diff(upper_in_b, b)


def cases() -> list[Case]:
    g = rng(5)
    a = np.exp(g.uniform(np.log(1e-2), np.log(500), 40))
    x = a * np.exp(g.uniform(np.log(1e-2), np.log(10), 40))
    out: list[Case] = []
    out += value_and_grads(
        "gammainc",
        "gammainc",
        p_,
        [a, x],
        wrt=[1],
        rtol=1e-12,
        atol=1e-300,
        grad_rtol=1e-12,
        grad_atol=1e-300,
        exact={1: gamma_density},
    )
    out += value_and_grads(
        "gammaincc",
        "gammaincc",
        q_,
        [a, x],
        wrt=[1],
        rtol=1e-12,
        atol=1e-300,
        grad_rtol=1e-12,
        grad_atol=1e-300,
        exact={1: neg_gamma_density},
    )
    da: F64 = mpvec(gamma_shape_grad, a, x)
    out.append(case("gammainc/d0", "gammainc", [a, x], da, rtol=1e-10, atol=1e-300, wrt=0))
    out.append(case("gammaincc/d0", "gammaincc", [a, x], -da, rtol=1e-10, atol=1e-300, wrt=0))
    tail_a = np.exp(g.uniform(np.log(0.5), np.log(50), 20))
    tail_x = np.concatenate([tail_a[:10] * 30, tail_a[10:] / 1000])
    out += value_and_grads(
        "logGammainc", "logGammainc", log_p, [tail_a, tail_x], wrt=[0, 1], rtol=1e-12, grad_rtol=1e-9, dps=400
    )
    out += value_and_grads(
        "logGammaincc", "logGammaincc", log_q, [tail_a, tail_x], wrt=[0, 1], rtol=1e-12, grad_rtol=1e-9, dps=400
    )
    p = np.concatenate([g.uniform(0, 1, 25), np.exp(g.uniform(np.log(1e-200), np.log(1e-4), 15))])
    ia = np.exp(g.uniform(np.log(1e-2), np.log(500), 40))
    out.append(case("gammaincinv", "gammaincinv", [ia, p], mpvec(gammaincinv, ia, p), rtol=1e-12))
    out.append(case("gammainccinv", "gammainccinv", [ia, p], mpvec(gammainccinv, ia, p), rtol=1e-12))
    ba = np.exp(g.uniform(np.log(1e-2), np.log(1e3), 40))
    bb = np.exp(g.uniform(np.log(1e-2), np.log(1e3), 40))
    bx = np.concatenate(
        [
            g.uniform(0, 1, 30),
            np.exp(g.uniform(np.log(1e-8), np.log(1e-2), 5)),
            1 - np.exp(g.uniform(np.log(1e-8), np.log(1e-2), 5)),
        ]
    )
    out += value_and_grads(
        "betainc",
        "betainc",
        i_,
        [ba, bb, bx],
        wrt=[2],
        rtol=1e-12,
        atol=1e-300,
        grad_rtol=1e-12,
        grad_atol=1e-300,
        exact={2: beta_density},
    )
    for which in (0, 1):
        grad = mpvec(partial(beta_shape_grad, which), ba, bb, bx)
        out.append(case(f"betainc/d{which}", "betainc", [ba, bb, bx], grad, rtol=1e-10, atol=1e-300, wrt=which))
    # Integer shapes make the continued fraction terminate while its shape derivatives do not (a past bug).
    ia2, ib2 = np.meshgrid(np.array([0.5, 1.0, 2.0, 3.0, 5.0]), np.array([1.0, 2.0, 3.0, 0.7]))
    ix2 = np.resize(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), ia2.size)
    ints = [ia2.ravel(), ib2.ravel(), ix2]
    for which in (0, 1):
        grad = mpvec(partial(beta_shape_grad, which), *ints)
        out.append(case(f"betainc/integer-shapes/d{which}", "betainc", ints, grad, rtol=1e-10, atol=1e-300, wrt=which))
    bp = np.concatenate([g.uniform(0, 1, 30), np.exp(g.uniform(np.log(1e-100), np.log(1e-4), 10))])
    out.append(case("betaincinv", "betaincinv", [ba, bb, bp], mpvec(betaincinv, ba, bb, bp), rtol=1e-12, atol=1e-300))
    return out
