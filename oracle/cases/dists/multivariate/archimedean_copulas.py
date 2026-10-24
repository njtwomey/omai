"""Archimedean copulas (dists/multivariate/archimedean-copulas.ts) against statsmodels."""

import numpy as np
import statsmodels
from statsmodels.distributions.copula.api import ClaytonCopula, FrankCopula, GumbelCopula
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    g = rng(48)
    out: list[Case] = []
    u2 = g.uniform(0.05, 0.95, size=(6, 2))
    u3 = g.uniform(0.05, 0.95, size=(4, 3))
    for name, family, theta in [
        ("clayton", ClaytonCopula, 2.0),
        ("gumbel", GumbelCopula, 1.8),
        ("frank", FrankCopula, 3.0),
    ]:
        c2 = family(theta)
        c3 = family(theta, k_dim=3)
        th = np.array(theta)
        out += [
            case(f"{name}/logDensity", f"{name}.logDensity", [th, u2], np.log(c2.pdf(u2)), rtol=1e-10, dim=2),
            case(f"{name}/cdf", f"{name}.cdf", [th, u2], c2.cdf(u2), rtol=1e-12, dim=2),
            case(f"{name}/tau", f"{name}.kendallTau", [th], np.array(c2.tau()), rtol=1e-10, dim=2),
            case(f"{name}3/logDensity", f"{name}.logDensity", [th, u3], np.log(c3.pdf(u3)), rtol=1e-8, dim=3),
            case(f"{name}3/cdf", f"{name}.cdf", [th, u3], c3.cdf(u3), rtol=1e-12, dim=3),
        ]
    # Negative dependence for the bivariate Frank copula.
    fn = FrankCopula(-2.5)
    out.append(
        case(
            "frankNegative/logDensity", "frank.logDensity", [np.array(-2.5), u2], np.log(fn.pdf(u2)), rtol=1e-10, dim=2
        )
    )
    return out
