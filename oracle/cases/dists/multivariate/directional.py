"""Directional distributions (dists/multivariate/directional.ts) against scipy.stats."""

import numpy as np
import scipy
from mdml_oracle.distributions import method_case
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from scipy import special, stats

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def circular_variance(kappa: float) -> FloatArray:
    """1 - I_1(kappa) / I_0(kappa), the von Mises circular variance."""
    return np.atleast_1d(1 - special.i1e(kappa) / special.i0e(kappa))


def cases() -> list[Case]:
    g = rng(46)
    out: list[Case] = []
    q = np.array([0.05, 0.3, 0.5, 0.8, 0.95])
    for i, (loc, kappa) in enumerate([(0.0, 0.5), (1.0, 4.0), (-2.0, 10.0)]):
        params = [np.array([loc]), np.array([kappa])]
        rv = stats.vonmises(kappa, loc=loc)
        x = loc + g.uniform(-np.pi, np.pi, 6)
        label = f"vonMises/{i}"
        out += [
            method_case(label, "vonMises", "logProb", params, x, rv.logpdf(x), rtol=1e-12),
            # The Fourier series is accurate in absolute terms; far-tail values carry 1e-15 absolute error.
            method_case(label, "vonMises", "cdf", params, x, rv.cdf(x), rtol=1e-10, atol=1e-14),
            method_case(label, "vonMises", "ppf", params, q, rv.ppf(q), rtol=1e-9),
            method_case(label, "vonMises", "entropy", params, None, np.atleast_1d(rv.entropy()), rtol=1e-12),
            # The circular variance 1 - I_1 / I_0 (torch's convention; torch's own Bessel approximations carry about
            # 1e-8, so the reference is scipy.special).
            method_case(label, "vonMises", "variance", params, None, circular_variance(kappa), rtol=1e-12),
        ]
    for i, (d, kappa) in enumerate([(3, 2.0), (5, 30.0)]):
        mu = g.normal(size=d)
        mu /= np.linalg.norm(mu)
        vmf = stats.vonmises_fisher(mu, kappa)
        x = g.normal(size=(6, d))
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        params = [mu, np.array(kappa)]
        out += [
            case(
                f"vonMisesFisher/{i}/logProb",
                "vonMisesFisher.logProb",
                [*params, x],
                vmf.logpdf(x),
                rtol=1e-10,
                arity=2,
            ),
            case(
                f"vonMisesFisher/{i}/entropy",
                "vonMisesFisher.entropy",
                params,
                np.array(vmf.entropy()),
                rtol=1e-10,
                arity=2,
            ),
        ]
    return out
