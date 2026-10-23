"""Copulas (dists/multivariate/copulas.ts and its siblings) against statsmodels' copula module."""

import numpy as np
import statsmodels
from statsmodels.distributions.copula.api import (
    ClaytonCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    StudentTCopula,
)
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    g = rng(47)
    out: list[Case] = []
    u2 = g.uniform(0.05, 0.95, size=(6, 2))
    u3 = g.uniform(0.05, 0.95, size=(4, 3))
    r = np.array([[1.0, 0.4], [0.4, 1.0]])
    gauss = GaussianCopula(0.4)
    out += [
        case("gaussian/logDensity", "gaussian.logDensity", [r, u2], np.log(gauss.pdf(u2)), rtol=1e-10),
        case("gaussian/cdf", "gaussian.cdf", [r, u2], gauss.cdf(u2), rtol=1e-8),
    ]
    r3 = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.5], [-0.2, 0.5, 1.0]])
    out.append(
        case(
            "gaussian3/logDensity",
            "gaussian.logDensity",
            [r3, u3],
            np.log(GaussianCopula(r3, k_dim=3).pdf(u3)),
            rtol=1e-10,
        )
    )
    t = StudentTCopula(0.4, df=5)
    out.append(case("studentT/logDensity", "studentT.logDensity", [r, u2], np.log(t.pdf(u2)), rtol=1e-10, df=5))
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
    # Fitting by inversion of Kendall's tau.
    z = g.normal(size=300)
    data = np.stack([z, 0.6 * z + 0.8 * g.normal(size=300)], axis=-1)
    for name, family in [("clayton", ClaytonCopula), ("gumbel", GumbelCopula), ("frank", FrankCopula)]:
        out.append(case(f"fit/{name}", f"fit.{name}", [data], np.array(family().fit_corr_param(data)), rtol=1e-8))
    out.append(
        case("fit/gaussian", "fit.gaussian", [data], np.array(GaussianCopula().fit_corr_param(data)), rtol=1e-10)
    )
    return out
