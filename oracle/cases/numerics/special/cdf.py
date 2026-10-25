"""Distribution functions of test statistics against scipy."""

from functools import partial

import numpy as np
import scipy
import scipy.special as sp
from mdml_oracle.fixture import Case
from mdml_oracle.helpers import FloatArray, loguniform, reference, rng
from scipy.stats import kstwobign

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def smirnov(n: int, d: FloatArray) -> FloatArray:
    return sp.smirnov(n, d)


def cases() -> list[Case]:
    g = rng(6)
    df = loguniform(g, 30, 0.5, 500)
    t = np.concatenate([g.uniform(-6, 6, 27), [0.0, 1e-8, -40.0]])
    p = np.concatenate([g.uniform(0.01, 0.99, 28), [0.5, 1e-6]])
    k = loguniform(g, 30, 0.5, 100)
    x = k * loguniform(g, 30, 0.1, 4)
    d1, d2 = loguniform(g, 30, 0.5, 100), loguniform(g, 30, 0.5, 100)
    fx = loguniform(g, 30, 0.05, 20)
    kx = loguniform(g, 20, 0.2, 3)
    out = [
        reference("stdtr", "stdtr", sp.stdtr, [df, t]),
        reference("stdtrit", "stdtrit", sp.stdtrit, [df, p]),
        reference("chdtr", "chdtr", sp.chdtr, [k, x]),
        reference("chdtrc", "chdtrc", sp.chdtrc, [k, x]),
        reference("fdtr", "fdtr", sp.fdtr, [d1, d2, fx]),
        reference("fdtrc", "fdtrc", sp.fdtrc, [d1, d2, fx]),
        reference("kolmogorov", "kolmogorov", sp.kolmogorov, [kx]),
        # scipy's kstwobign.pdf is itself accurate to about 1e-8, so the tolerance follows the reference.
        reference("kolmogorovPdf", "kolmogorovPdf", kstwobign.pdf, [kx], rtol=1e-7),
    ]
    for n in [1, 5, 20, 100]:
        out.append(reference(f"smirnov/{n}", "smirnov", partial(smirnov, n), [g.uniform(0, 1, 15)], n=n))
    return out
