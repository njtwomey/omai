"""Trust-region methods against scipy.optimize.minimize: dogleg, trust-ncg and trust-exact on standard problems."""

from typing import Literal

import numpy as np
import scipy
import scipy.optimize as so
from omai_oracle.fixture import Case, case
from omai_oracle.optim_problems import SMOOTH

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type Method = Literal["dogleg", "trust-ncg", "trust-exact"]

METHODS: dict[Method, str] = {"dogleg": "trustDogleg", "trust-ncg": "trustNcg", "trust-exact": "trustExact"}

# Dogleg needs a positive definite Hessian along the path, so it runs on the convex problems and well-started ones.
PROBLEMS_OF: dict[Method, list[str]] = {
    "dogleg": ["rosen", "spdQuadratic", "powell"],
    "trust-ncg": ["rosen", "rosen5", "beale", "himmelblau", "spdQuadratic", "powell", "wood"],
    "trust-exact": ["rosen", "rosen5", "beale", "himmelblau", "spdQuadratic", "powell", "wood"],
}


def cases() -> list[Case]:
    out: list[Case] = []
    for method, fn in METHODS.items():
        for name in PROBLEMS_OF[method]:
            p = SMOOTH[name]
            r = so.minimize(p.f, p.x0, method=method, jac=p.grad, hess=p.hess, options={"gtol": 1e-8})
            out.append(
                case(
                    f"{fn}/{name}",
                    fn,
                    [p.x0],
                    np.concatenate([np.asarray(r.x), [float(r.fun)]]),
                    rtol=1e-6,
                    atol=1e-7,
                    problem=name,
                    nit=int(r.nit),
                    success=bool(r.success),
                )
            )
    return out
