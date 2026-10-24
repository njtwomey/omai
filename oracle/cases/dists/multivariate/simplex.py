"""Distributions on the simplex and count vectors (dists/multivariate/simplex.ts) against scipy.stats."""

import numpy as np
import scipy
from scipy import stats
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(38)
    out: list[Case] = []
    alpha = np.array([0.8, 2.0, 5.0])
    simplex = g.dirichlet(alpha, size=5)
    dir_ = stats.dirichlet(alpha)
    logp = np.array([dir_.logpdf(s) for s in simplex])
    out.append(case("dirichlet/logProb", "dirichlet.logProb", [alpha, simplex], logp, rtol=1e-10))
    out.append(case("dirichlet/mean", "dirichlet.mean", [alpha], dir_.mean()))
    out.append(case("dirichlet/variance", "dirichlet.variance", [alpha], dir_.var()))
    out.append(case("dirichlet/entropy", "dirichlet.entropy", [alpha], np.array(dir_.entropy()), rtol=1e-10))
    probs = np.array([0.2, 0.5, 0.3])
    mult = stats.multinomial(6, probs)
    xs = np.array([[1.0, 3, 2], [0, 6, 0], [2, 2, 2], [6, 0, 0], [1, 1, 1]])
    out.append(case("multinomial/logProb", "multinomial.logProb", [probs, xs], mult.logpmf(xs), rtol=1e-10, n=6))
    out.append(case("multinomial/entropy", "multinomial.entropy", [probs], np.array(mult.entropy()), rtol=1e-10, n=6))
    out.append(case("multinomial/variance", "multinomial.variance", [probs], np.diag(mult.cov()), n=6))
    conc = np.array([1.5, 0.7, 3.0])
    dm = stats.dirichlet_multinomial(conc, 6)
    out.append(
        case("dirichletMultinomial/logProb", "dirichletMultinomial.logProb", [conc, xs], dm.logpmf(xs), rtol=1e-10, n=6)
    )
    out.append(case("dirichletMultinomial/mean", "dirichletMultinomial.mean", [conc], dm.mean(), n=6))
    out.append(
        case("dirichletMultinomial/variance", "dirichletMultinomial.variance", [conc], dm.var(), rtol=1e-12, n=6)
    )
    return out
