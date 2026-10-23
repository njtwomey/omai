"""Multivariate distributions against scipy.stats."""

import numpy as np
import scipy
from scipy import stats
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng, spd

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(32)
    out: list[Case] = []
    mean = np.array([0.5, -1.0, 2.0])
    cov = spd(g, 3) / 3
    x = g.normal(size=(6, 3))
    mvn = stats.multivariate_normal(mean, cov)
    out.append(case("mvn/logProb", "multivariateNormal.logProb", [mean, cov, x], mvn.logpdf(x), rtol=1e-10))
    out.append(case("mvn/entropy", "multivariateNormal.entropy", [mean, cov], np.array(mvn.entropy()), rtol=1e-12))
    out.append(case("mvn/variance", "multivariateNormal.variance", [mean, cov], np.diag(cov), rtol=1e-12))
    # Conditioning on coordinate 2 = 1.5, against the textbook Gaussian conditioning formulas.
    a, b = [0, 1], [2]
    s_ab, s_bb = cov[np.ix_(a, b)], cov[np.ix_(b, b)]
    cond_mean = mean[a] + (s_ab @ np.linalg.solve(s_bb, np.array([1.5]) - mean[b]))
    cond_cov = cov[np.ix_(a, a)] - s_ab @ np.linalg.solve(s_bb, s_ab.T)
    out.append(case("mvn/condition/mean", "multivariateNormal.conditionMean", [mean, cov], cond_mean, rtol=1e-10))
    out.append(case("mvn/condition/cov", "multivariateNormal.conditionCov", [mean, cov], cond_cov, rtol=1e-10))
    alpha = np.array([0.8, 2.0, 5.0])
    simplex = g.dirichlet(alpha, size=5)
    dir_ = stats.dirichlet(alpha)
    out.append(
        case(
            "dirichlet/logProb",
            "dirichlet.logProb",
            [alpha, simplex],
            np.array([dir_.logpdf(s) for s in simplex]),
            rtol=1e-10,
        )
    )
    out.append(case("dirichlet/mean", "dirichlet.mean", [alpha], dir_.mean(), rtol=1e-12))
    out.append(case("dirichlet/variance", "dirichlet.variance", [alpha], dir_.var(), rtol=1e-12))
    out.append(case("dirichlet/entropy", "dirichlet.entropy", [alpha], np.array(dir_.entropy()), rtol=1e-10))
    probs = np.array([0.2, 0.5, 0.3])
    mult = stats.multinomial(6, probs)
    xs = np.array([[1.0, 3, 2], [0, 6, 0], [2, 2, 2], [6, 0, 0]])
    out.append(case("multinomial/logProb", "multinomial.logProb", [probs, xs], mult.logpmf(xs), rtol=1e-10, n=6))
    out.append(case("multinomial/entropy", "multinomial.entropy", [probs], np.array(mult.entropy()), rtol=1e-10, n=6))
    out.append(case("multinomial/variance", "multinomial.variance", [probs], np.diag(mult.cov()), rtol=1e-12, n=6))
    return out
