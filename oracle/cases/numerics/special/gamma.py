"""Gamma-family functions against mpmath."""

from functools import partial

import mpmath as mp
import numpy as np
from whitebox_oracle.fixture import Case
from whitebox_oracle.helpers import rng
from whitebox_oracle.precise import Mpf, value_and_grads

REFERENCE: dict[str, str] = {"mpmath": mp.__version__}


def lgamma(v: Mpf) -> Mpf:
    return mp.log(abs(mp.gamma(v)))


def polygamma(n: int, v: Mpf) -> Mpf:
    return mp.polygamma(n, v)


def lbeta(u: Mpf, v: Mpf) -> Mpf:
    return mp.log(mp.beta(u, v))


def factorial(v: Mpf) -> Mpf:
    return mp.gamma(v + 1)


def lfactorial(v: Mpf) -> Mpf:
    return mp.loggamma(v + 1)


def lbinom(u: Mpf, v: Mpf) -> Mpf:
    return mp.log(mp.binomial(u, v))


def cases() -> list[Case]:
    g = rng(4)
    pos = np.concatenate([np.exp(g.uniform(np.log(1e-6), np.log(170), 30)), [0.5, 1.0, 2.0, 3.5, 10.0]])
    near = g.uniform(0.75, 2.25, 20)
    neg = g.uniform(-20, 0, 20)
    neg = neg[np.abs(neg - np.round(neg)) > 1e-3]
    big = np.exp(g.uniform(np.log(171), np.log(1e8), 15))
    out: list[Case] = []
    out += value_and_grads("gamma/positive", "gamma", mp.gamma, [pos], wrt=[0], rtol=1e-12, grad_rtol=1e-10)
    out += value_and_grads("gamma/negative", "gamma", mp.gamma, [neg], wrt=[0], rtol=1e-12, grad_rtol=1e-10)
    out += value_and_grads("lgamma/positive", "lgamma", lgamma, [np.concatenate([pos, big])], wrt=[0], rtol=1e-13)
    out += value_and_grads("lgamma/near-zeros", "lgamma", lgamma, [near], wrt=[0], rtol=1e-13, atol=1e-16)
    out += value_and_grads("lgamma/negative", "lgamma", lgamma, [neg], wrt=[0], rtol=1e-12, atol=1e-14)
    out += value_and_grads("rgamma", "rgamma", mp.rgamma, [np.concatenate([pos, neg])], wrt=[0], rtol=1e-12)
    out += value_and_grads(
        "digamma/positive", "digamma", mp.digamma, [np.concatenate([pos, big])], wrt=[0], rtol=1e-13, atol=1e-15
    )
    out += value_and_grads("digamma/negative", "digamma", mp.digamma, [neg], wrt=[0], rtol=1e-12, atol=1e-13)
    for n in [1, 2, 3, 5]:
        xs = np.concatenate([np.exp(g.uniform(np.log(1e-3), np.log(1e3), 20)), g.uniform(-6, 0, 10)])
        xs = xs[np.abs(xs - np.round(xs)) > 1e-2]
        out += value_and_grads(
            f"polygamma/{n}", "polygamma", partial(polygamma, n), [xs], wrt=[0], rtol=1e-12, grad_rtol=1e-10, n=n
        )
    a, b = np.exp(g.uniform(np.log(1e-3), np.log(1e5), 30)), np.exp(g.uniform(np.log(1e-3), np.log(1e5), 30))
    out += value_and_grads("lbeta", "lbeta", lbeta, [a, b], wrt=[0, 1], rtol=1e-12, atol=1e-14)
    sa, sb = np.exp(g.uniform(np.log(1e-2), np.log(60), 30)), np.exp(g.uniform(np.log(1e-2), np.log(60), 30))
    out += value_and_grads("beta", "beta", mp.beta, [sa, sb], wrt=[0, 1], rtol=1e-12)
    n = np.array([0.0, 1, 5, 10, 20, 50, 100, 170, 2.5, 7.25])
    out += value_and_grads("factorial", "factorial", factorial, [n], wrt=[0], rtol=1e-12)
    out += value_and_grads(
        "lfactorial", "lfactorial", lfactorial, [np.concatenate([n, big])], wrt=[0], rtol=1e-13, atol=1e-15
    )
    nn = np.array([5.0, 10, 30, 100, 1000, 1e6, 1e9, 7.5, 20.25, 52])
    kk = np.array([2.0, 3, 15, 50, 2, 7, 12, 2.5, 0.5, 26])
    out += value_and_grads("binom", "binom", mp.binomial, [nn, kk], wrt=[0, 1], rtol=1e-12)
    out += value_and_grads("lbinom", "lbinom", lbinom, [nn, kk], wrt=[0, 1], rtol=1e-12)
    return out
