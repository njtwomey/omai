"""GLM conditionals (dists/compose/conditional.ts) against statsmodels' GLM log-likelihoods."""

import numpy as np
import statsmodels
import statsmodels.api as sm
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    g = rng(51)
    n, p = 12, 3
    x = g.normal(size=(n, p))
    beta = np.array([0.4, -0.3, 0.2])
    intercept = 0.1
    design = np.column_stack([np.ones(n), x])
    params = np.concatenate([[intercept], beta])
    eta = design @ params
    out: list[Case] = []
    responses = {
        "poisson": (sm.families.Poisson(), g.poisson(np.exp(eta)).astype(np.float64), 1.0),
        "binomial": (sm.families.Binomial(), (g.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(np.float64), 1.0),
        "gamma": (sm.families.Gamma(sm.families.links.Log()), g.gamma(2.0, np.exp(eta) / 2.0), 0.5),
        "gaussian": (sm.families.Gaussian(), eta + g.normal(size=n) * 0.7, 0.49),
    }
    for name, (family, y, scale) in responses.items():
        model = sm.GLM(y, design, family=family)
        ll = model.loglike(params, scale=scale)
        out.append(
            case(
                f"glm/{name}",
                f"glm.{name}",
                [x, y, beta, np.array(intercept), np.array(scale)],
                np.array(ll),
                rtol=1e-10,
            )
        )
    return out
