"""Zeta functions, exponential integrals, Lambert W and the sine, cosine and Fresnel integrals against mpmath."""

from functools import partial

import mpmath as mp
import numpy as np
from whitebox_oracle.fixture import Case
from whitebox_oracle.helpers import rng
from whitebox_oracle.precise import Mpf, value_and_grads

REFERENCE: dict[str, str] = {"mpmath": mp.__version__}


def d_lambertw(x: Mpf) -> Mpf:
    w = mp.lambertw(x)
    return 1 / (mp.exp(w) * (1 + w))


def expn(n: int, v: Mpf) -> Mpf:
    return mp.expint(n, v)


def cases() -> list[Case]:
    g = rng(7)
    out: list[Case] = []
    s = np.concatenate([g.uniform(1.01, 10, 20), g.uniform(10, 80, 5)])
    out += value_and_grads("zeta", "zeta", mp.zeta, [s], wrt=[0], grad_rtol=1e-12)
    q = np.exp(g.uniform(np.log(1e-3), np.log(1e3), 25))
    out += value_and_grads("hurwitzZeta", "hurwitzZeta", mp.zeta, [s, q], wrt=[0, 1], rtol=1e-13, grad_rtol=1e-12)
    w = np.concatenate([g.uniform(-1 / np.e, 0, 15), np.exp(g.uniform(np.log(1e-10), np.log(1e100), 15))])
    out += value_and_grads(
        "lambertW",
        "lambertW",
        mp.lambertw,
        [w],
        wrt=[0],
        rtol=1e-14,
        atol=1e-300,
        grad_rtol=1e-12,
        exact={0: d_lambertw},
    )
    for n in [0, 1, 2, 5]:
        x = np.exp(g.uniform(np.log(1e-3), np.log(500), 20))
        out += value_and_grads(f"expn/{n}", "expn", partial(expn, n), [x], wrt=[0], rtol=1e-13, atol=1e-300, n=n)
    ei = np.concatenate([g.uniform(-50, 50, 25), np.exp(g.uniform(np.log(1e-8), np.log(700), 10))])
    out += value_and_grads("expi", "expi", mp.ei, [ei], wrt=[0], rtol=1e-13, atol=1e-300)
    sx = np.concatenate([g.uniform(-30, 30, 25), np.exp(g.uniform(np.log(1e-6), np.log(1e4), 10))])
    out += value_and_grads("si", "sici", mp.si, [sx], wrt=[0], rtol=1e-13, atol=1e-15, part=0)
    out += value_and_grads("ci", "sici", mp.ci, [np.abs(sx)], wrt=[0], rtol=1e-13, atol=1e-15, part=1)
    fx = g.uniform(-20, 20, 30)
    out += value_and_grads("fresnelS", "fresnel", mp.fresnels, [fx], wrt=[0], rtol=1e-13, atol=1e-15, part=0)
    out += value_and_grads("fresnelC", "fresnel", mp.fresnelc, [fx], wrt=[0], rtol=1e-13, atol=1e-15, part=1)
    return out
