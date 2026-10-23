"""Bessel functions of real order against mpmath."""

import mpmath as mp
import numpy as np
from whitebox_oracle.fixture import Case
from whitebox_oracle.helpers import rng
from whitebox_oracle.precise import Mpf, value_and_grads

REFERENCE: dict[str, str] = {"mpmath": mp.__version__}


def ive(v: Mpf, x: Mpf) -> Mpf:
    return mp.besseli(v, x) * mp.exp(-abs(x))


def kve(v: Mpf, x: Mpf) -> Mpf:
    return mp.besselk(v, x) * mp.exp(x)


def log_iv(v: Mpf, x: Mpf) -> Mpf:
    return mp.log(mp.besseli(v, x))


def i0e(x: Mpf) -> Mpf:
    return ive(mp.mpf(0), x)


def i1e(x: Mpf) -> Mpf:
    return ive(mp.mpf(1), x)


def cases() -> list[Case]:
    g = rng(8)
    v = np.concatenate([g.uniform(0, 3, 15), g.uniform(3, 30, 10), [0.0, 1.0, 0.5, 2.0, 5.0]])
    x = np.concatenate(
        [
            np.exp(g.uniform(np.log(1e-3), np.log(2), 10)),
            g.uniform(2, 60, 15),
            np.exp(g.uniform(np.log(60), np.log(1e4), 5)),
        ]
    )
    out: list[Case] = []
    out += value_and_grads("ive", "ive", ive, [v, x], wrt=[1], rtol=1e-12, atol=1e-300)
    out += value_and_grads("kve", "kve", kve, [v, x], wrt=[1], rtol=1e-12)
    out += value_and_grads("logIv", "logIv", log_iv, [v, x], wrt=[0, 1], rtol=1e-12, grad_rtol=1e-6)
    small = x < 50
    out += value_and_grads("iv", "iv", mp.besseli, [v[small], x[small]], wrt=[1], rtol=1e-12, atol=1e-300)
    out += value_and_grads("kv", "kv", mp.besselk, [v[small], x[small]], wrt=[1], rtol=1e-12)
    # J and Y oscillate: compare with an absolute tolerance scaled to their envelope.
    out += value_and_grads(
        "jv", "jv", mp.besselj, [v, x], wrt=[0, 1], rtol=1e-11, atol=1e-13, grad_rtol=1e-7, grad_atol=1e-9
    )
    out += value_and_grads("yv", "yv", mp.bessely, [v, x], wrt=[1], rtol=1e-11, atol=1e-13, grad_atol=1e-12)
    nv = g.uniform(-4.9, -0.1, 15)
    nx = np.exp(g.uniform(np.log(0.1), np.log(30), 15))
    out += value_and_grads("jv/negative-order", "jv", mp.besselj, [nv, nx], wrt=[1], rtol=1e-11, atol=1e-13)
    out += value_and_grads("ive/negative-order", "ive", ive, [nv, nx], wrt=[1], rtol=1e-11, atol=1e-14)
    ix = g.uniform(-30, 30, 20)
    out += value_and_grads("i0e", "i0e", i0e, [ix], wrt=[0], rtol=1e-13)
    out += value_and_grads("i1e", "i1e", i1e, [ix], wrt=[0], rtol=1e-13, atol=1e-300)
    return out
