"""Copula fitting (dists/multivariate/copula-fitting.ts) against statsmodels' fit_corr_param."""

import numpy as np
import statsmodels
from statsmodels.distributions.copula.api import ClaytonCopula, FrankCopula, GaussianCopula, GumbelCopula
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    g = rng(49)
    z = g.normal(size=300)
    data = np.stack([z, 0.6 * z + 0.8 * g.normal(size=300)], axis=-1)
    out: list[Case] = []
    for name, family in [("clayton", ClaytonCopula), ("gumbel", GumbelCopula), ("frank", FrankCopula)]:
        out.append(case(f"fit/{name}", f"fit.{name}", [data], np.array(family().fit_corr_param(data)), rtol=1e-8))
    out.append(
        case("fit/gaussian", "fit.gaussian", [data], np.array(GaussianCopula().fit_corr_param(data)), rtol=1e-10)
    )
    return out
