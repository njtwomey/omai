"""Elliptical copulas (dists/multivariate/copulas.ts) against statsmodels."""

import numpy as np
import statsmodels
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from statsmodels.distributions.copula.api import GaussianCopula, StudentTCopula

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    g = rng(47)
    u2 = g.uniform(0.05, 0.95, size=(6, 2))
    u3 = g.uniform(0.05, 0.95, size=(4, 3))
    r = np.array([[1.0, 0.4], [0.4, 1.0]])
    gauss = GaussianCopula(0.4)
    r3 = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.5], [-0.2, 0.5, 1.0]])
    t = StudentTCopula(0.4, df=5)
    return [
        case("gaussian/logDensity", "gaussian.logDensity", [r, u2], np.log(gauss.pdf(u2)), rtol=1e-10),
        case("gaussian/cdf", "gaussian.cdf", [r, u2], gauss.cdf(u2), rtol=1e-8),
        case(
            "gaussian3/logDensity",
            "gaussian.logDensity",
            [r3, u3],
            np.log(GaussianCopula(r3, k_dim=3).pdf(u3)),
            rtol=1e-10,
        ),
        case("studentT/logDensity", "studentT.logDensity", [r, u2], np.log(t.pdf(u2)), rtol=1e-10, df=5),
    ]
