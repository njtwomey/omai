"""Proper scoring rules for distributions against properscoring (CRPS closed form and quadrature) and scipy.stats
(log and spherical scores); the energy score is its definition over an ensemble.
"""

import numpy as np
import properscoring as ps
import scipy
import scipy.integrate as si
import scipy.stats as st
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"properscoring": "0.1", "scipy": scipy.__version__}

# properscoring integrates with scipy's default quad tolerances, so the closed forms agree with it to about 1e-8.
QUAD = 1e-7


def energy(samples: FloatArray, y: FloatArray) -> float:
    m = samples.shape[1]
    to_obs = np.linalg.norm(samples - y[:, None, :], axis=-1).mean(-1)
    pairs = np.linalg.norm(samples[:, :, None, :] - samples[:, None, :, :], axis=-1).sum((-2, -1)) / (2 * m * m)
    return float((to_obs - pairs).mean())


def gamma_crps(shape: float, y: float) -> float:
    """CRPS of a unit-rate gamma by quadrature of its CDF (scipy.integrate.quad)."""

    def below(x: float) -> float:
        return float(st.gamma.cdf(x, shape) ** 2)

    def above(x: float) -> float:
        return float(st.gamma.sf(x, shape) ** 2)

    b = si.quad(below, 0.0, y, epsabs=1e-13, epsrel=1e-12, limit=200)[0]
    a = si.quad(above, y, np.inf, epsabs=1e-13, epsrel=1e-12, limit=200)[0]
    return float(b + a)


def gamma_spherical(shape: float, y: float) -> float:
    """p(y) / ||p||_2 for a unit-rate gamma, the norm by quadrature."""

    def squared(x: float) -> float:
        return float(st.gamma.pdf(x, shape) ** 2)

    return float(st.gamma.pdf(y, shape) / np.sqrt(si.quad(squared, 0.0, np.inf, epsabs=1e-14, epsrel=1e-12)[0]))


def cases() -> list[Case]:
    g = rng(58)
    n = 12
    mu, sigma, y = g.normal(size=n), np.exp(0.4 * g.normal(size=n)), g.normal(size=n)
    shape = np.exp(0.3 * g.normal(size=n)) + 1.0
    ypos = np.exp(g.normal(size=n))
    samples, obs = g.normal(size=(4, 9, 3)), g.normal(size=(4, 3))
    out: list[Case] = [
        case("logScore/normal", "logScoreNormal", [mu, sigma, y], np.asarray(-np.mean(st.norm.logpdf(y, mu, sigma)))),
        case("crps/normal", "crpsNormal", [mu, sigma, y], np.asarray(np.mean(ps.crps_gaussian(y, mu, sigma)))),
        case(
            "crps/laplace",
            "crpsLaplace",
            [mu, sigma, y],
            np.asarray(
                np.mean([ps.crps_quadrature(v, st.laplace(m, s)) for v, m, s in zip(y, mu, sigma, strict=True)])
            ),
            rtol=QUAD,
        ),
        case(
            "crps/logistic",
            "crpsLogistic",
            [mu, sigma, y],
            np.asarray(
                np.mean([ps.crps_quadrature(v, st.logistic(m, s)) for v, m, s in zip(y, mu, sigma, strict=True)])
            ),
            rtol=QUAD,
        ),
        case(
            "crps/exponential",
            "crpsExponential",
            [sigma, ypos],
            np.asarray(
                np.mean([ps.crps_quadrature(v, st.expon(scale=1 / r)) for v, r in zip(ypos, sigma, strict=True)])
            ),
            rtol=QUAD,
        ),
        case(
            "crps/uniform",
            "crpsUniform",
            [mu, mu + sigma, y],
            np.asarray(
                np.mean([ps.crps_quadrature(v, st.uniform(a, s)) for v, a, s in zip(y, mu, sigma, strict=True)])
            ),
            rtol=QUAD,
        ),
        case(
            "crps/gamma-quadrature",
            "crpsGamma",
            [shape, ypos],
            np.asarray(np.mean([gamma_crps(float(a), float(v)) for a, v in zip(shape, ypos, strict=True)])),
            rtol=1e-8,
        ),
        case(
            "sphericalScore/normal",
            "sphericalScoreNormal",
            [mu, sigma, y],
            np.asarray((st.norm(mu, sigma).pdf(y) / np.sqrt(1 / (2 * sigma * np.sqrt(np.pi)))).mean()),
        ),
        case(
            "sphericalScore/gamma",
            "sphericalScoreGamma",
            [shape, ypos],
            np.asarray(np.mean([gamma_spherical(float(a), float(v)) for a, v in zip(shape, ypos, strict=True)])),
            rtol=1e-8,
        ),
        case("energyScore", "energyScore", [samples, obs], np.asarray(energy(samples, obs))),
    ]
    return out
