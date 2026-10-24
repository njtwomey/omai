"""Exponential dispersion families (dists/families/dispersion.ts) against statsmodels.genmod.families."""

from typing import cast

import numpy as np
import scipy
import statsmodels
from scipy import stats
from statsmodels.genmod import families as F
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    g = rng(49)
    out: list[Case] = []
    mu_pos = np.exp(g.normal(size=8) * 0.5)
    families: list[tuple[str, F.Family, np.ndarray, np.ndarray, float]] = [
        ("gaussian", F.Gaussian(), g.normal(size=8), g.normal(size=8), 1.7),
        ("binomial", F.Binomial(), np.array([0.0, 1, 1, 0, 1, 0.0, 1, 1]), g.uniform(0.1, 0.9, 8), 1.0),
        ("poisson", F.Poisson(), np.array([0.0, 1, 3, 2, 0, 5, 1, 4]), mu_pos * 2, 1.0),
        ("gamma", F.Gamma(), np.exp(g.normal(size=8)), mu_pos, 0.6),
        ("inverseGaussian", F.InverseGaussian(), np.exp(g.normal(size=8)), mu_pos, 0.8),
        ("negativeBinomial", F.NegativeBinomial(alpha=0.7), np.array([0.0, 1, 3, 2, 0, 5, 1, 4]), mu_pos * 2, 1.0),
        ("tweedie", F.Tweedie(var_power=1.5), np.array([0.0, 0.3, 1.2, 0.0, 2.5, 0.8, 4.0, 0.1]), mu_pos, 1.3),
    ]
    for name, fam, y, mu, scale in families:
        # statsmodels annotates var_weights as float but accepts arrays.
        weights = cast(float, g.uniform(0.5, 2.0, 8))
        sc = np.array(scale)
        out += [
            case(f"{name}/variance", f"{name}.variance", [mu], fam.variance(mu)),
            case(f"{name}/varianceDerivative", f"{name}.varianceDerivative", [mu], fam.variance.deriv(mu), rtol=1e-12),
            case(
                f"{name}/deviance",
                f"{name}.deviance",
                [y, mu, weights],
                np.array(fam.deviance(y, mu, var_weights=weights)),
                rtol=1e-10,
            ),
            case(
                f"{name}/devianceResiduals",
                f"{name}.devianceResiduals",
                [y, mu, weights],
                fam.resid_dev(y, mu, var_weights=weights),
                rtol=1e-10,
            ),
            case(
                f"{name}/logLikelihoodObs",
                f"{name}.logLikelihoodObs",
                [y, mu, sc, weights],
                fam.loglike_obs(y, mu, var_weights=weights, scale=scale),
                rtol=1e-10,
            ),
            case(f"{name}/startingMu", f"{name}.startingMu", [y], fam.starting_mu(y), rtol=1e-12),
        ]
    # The Tweedie distribution: P(Y = 0) and the density above zero against the family log-likelihood, and moments.
    y = np.array([0.0, 0.2, 1.0, 3.0])
    tw = F.Tweedie(var_power=1.4)
    out.append(
        case(
            "tweedieDistribution/logProb",
            "tweedieDistribution.logProb",
            [np.array(1.2), np.array(0.9), y],
            tw.loglike_obs(y, np.full(4, 1.2), scale=0.9),
            rtol=1e-10,
            power=1.4,
        )
    )
    # Its CDF at a point: the atom plus the Poisson mixture of gamma CDFs, from scipy's distributions.
    lam = 1.2**0.6 / (0.9 * 0.6)
    shape, gscale = 0.6 / 0.4, 0.9 * 0.4 * 1.2**0.4
    ks = np.arange(1, 80)
    cdf = np.exp(-lam) + np.sum(stats.poisson.pmf(ks, lam) * stats.gamma.cdf(1.5, ks * shape, scale=gscale))
    out.append(
        case(
            "tweedieDistribution/cdf",
            "tweedieDistribution.cdf",
            [np.array(1.2), np.array(0.9), np.array(1.5)],
            np.array(cdf),
            rtol=1e-12,
            power=1.4,
        )
    )
    return out
