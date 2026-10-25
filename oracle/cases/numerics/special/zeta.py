"""Zeta functions, exponential integrals, Lambert W and the sine, cosine and Fresnel integrals against scipy."""

from functools import partial

import numpy as np
import scipy
import scipy.special as sp
from mdml_oracle.fixture import Case
from mdml_oracle.helpers import FloatArray, loguniform, reference, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def lambertw(x: FloatArray) -> FloatArray:
    # scipy-stubs types lambertw for scalars only, so apply it elementwise.
    return np.array([sp.lambertw(float(v)).real for v in x], dtype=np.float64)


def si(x: FloatArray) -> FloatArray:
    return sp.sici(x)[0]


def ci(x: FloatArray) -> FloatArray:
    return sp.sici(x)[1]


def fresnel_s(x: FloatArray) -> FloatArray:
    return sp.fresnel(x)[0]


def fresnel_c(x: FloatArray) -> FloatArray:
    return sp.fresnel(x)[1]


def expn(n: int, x: FloatArray) -> FloatArray:
    return sp.expn(n, x)


def cases() -> list[Case]:
    g = rng(7)
    s = g.uniform(1.1, 20, 20)
    q = loguniform(g, 20, 0.01, 100)
    w = np.concatenate([g.uniform(-1 / np.e, 0, 10), loguniform(g, 15, 1e-6, 1e6)])
    x = loguniform(g, 20, 0.01, 50)
    ei = np.concatenate([g.uniform(-30, 30, 20), [1e-6]])
    sx = np.concatenate([g.uniform(-30, 30, 20), [0.0]])
    out = [
        reference("zeta", "zeta", sp.zeta, [s]),
        reference("hurwitzZeta", "hurwitzZeta", sp.zeta, [s, q]),
        reference("lambertW", "lambertW", lambertw, [w]),
        reference("expi", "expi", sp.expi, [ei]),
        reference("si", "sici", si, [sx], part=0),
        reference("ci", "sici", ci, [np.abs(sx[:-1])], part=1),
        reference("fresnelS", "fresnel", fresnel_s, [sx], part=0),
        reference("fresnelC", "fresnel", fresnel_c, [sx], part=1),
    ]
    for n in [0, 1, 2, 5]:
        out.append(reference(f"expn/{n}", "expn", partial(expn, n), [x], n=n))
    return out
