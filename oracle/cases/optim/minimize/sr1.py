"""SR1 with a trust region against scipy.optimize.minimize(method='trust-constr') with SR1 updates."""

import warnings

import numpy as np
import scipy
import scipy.optimize as so
from omai_oracle.fixture import Case, case
from omai_oracle.optim_problems import SMOOTH

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

NAMES: list[str] = ["rosen", "rosen5", "beale", "himmelblau", "spdQuadratic", "powell", "wood"]


def solution(r: so.OptimizeResult) -> np.ndarray:
    return np.concatenate([np.asarray(r.x, dtype=np.float64), [float(r.fun)]])


def cases() -> list[Case]:
    out: list[Case] = []
    for name in NAMES:
        p = SMOOTH[name]
        # Powell's singular function converges sublinearly to its minimum at the origin, so different methods stop
        # at different points near it: compare those to the minimum's scale rather than digit by digit.
        loose = 2e-3 if name == "powell" else 1e-6
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = so.minimize(
                p.f, p.x0, method="trust-constr", jac=p.grad, hess=so.SR1(), options={"gtol": 1e-10, "maxiter": 5000}
            )
        out.append(case(f"sr1/{name}", "sr1", [p.x0], solution(r), rtol=1e-5, atol=loose, problem=name))
    return out
