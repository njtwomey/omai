"""Bessel functions of real order against scipy.special."""

import numpy as np
import scipy
import scipy.special as sp
from mdml_oracle.fixture import Case
from mdml_oracle.helpers import FloatArray, loguniform, reference, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def log_iv(v: FloatArray, x: FloatArray) -> FloatArray:
    return np.log(sp.ive(v, x)) + x


def cases() -> list[Case]:
    g = rng(8)
    v = np.concatenate([g.uniform(0, 10, 25), [0.0, 0.5, 1.0, 2.0]])
    x = np.concatenate([loguniform(g, 25, 0.01, 100), [1.0, 5.0, 500.0, 2000.0]])
    nv, nx = g.uniform(-4.9, -0.1, 10), loguniform(g, 10, 0.1, 30)
    ix = g.uniform(-30, 30, 20)
    small = x < 50
    # J and Y oscillate: an absolute tolerance on the scale of their envelope.
    return [
        reference("ive", "ive", sp.ive, [v, x]),
        reference("kve", "kve", sp.kve, [v, x]),
        reference("logIv", "logIv", log_iv, [v, x]),
        reference("iv", "iv", sp.iv, [v[small], x[small]]),
        reference("kv", "kv", sp.kv, [v[small], x[small]]),
        reference("jv", "jv", sp.jv, [v, x], atol=1e-12),
        reference("yv", "yv", sp.yv, [v, x], atol=1e-12),
        reference("jv/negative-order", "jv", sp.jv, [nv, nx], atol=1e-12),
        reference("ive/negative-order", "ive", sp.ive, [nv, nx], atol=1e-12),
        reference("i0e", "i0e", sp.i0e, [ix]),
        reference("i1e", "i1e", sp.i1e, [ix]),
    ]
