"""Quasi-Newton methods against scipy.optimize.minimize: BFGS (a port: solutions and iteration counts), L-BFGS
(against unbounded L-BFGS-B) and SR1 with a trust region (against trust-constr with SR1 updates)."""

import warnings

import numpy as np
import scipy
import scipy.optimize as so
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.optim_problems import PROBLEMS

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

NAMES: list[str] = ["rosen", "rosen5", "beale", "himmelblau", "quadratic", "powell", "wood"]


def solution(r: so.OptimizeResult) -> np.ndarray:
    return np.concatenate([np.asarray(r.x, dtype=np.float64), [float(r.fun)]])


def cases() -> list[Case]:
    out: list[Case] = []
    for name in NAMES:
        p = PROBLEMS[name]
        # Powell's singular function converges sublinearly to its minimum at the origin, so different methods stop at
        # different points near it: compare those to the minimum's scale rather than digit by digit.
        loose = 2e-3 if name == "powell" else 1e-6
        r = so.minimize(p.f, p.x0, method="BFGS", jac=p.grad, options={"gtol": 1e-8})
        out.append(
            case(f"bfgs/{name}", "bfgs", [p.x0], solution(r), rtol=1e-6, atol=1e-7, problem=name, nit=int(r.nit))
        )
        r = so.minimize(p.f, p.x0, method="L-BFGS-B", jac=p.grad, options={"gtol": 1e-10, "ftol": 1e-15})
        out.append(case(f"lbfgs/{name}", "lbfgs", [p.x0], solution(r), rtol=1e-5, atol=loose, problem=name))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = so.minimize(
                p.f, p.x0, method="trust-constr", jac=p.grad, hess=so.SR1(), options={"gtol": 1e-10, "maxiter": 5000}
            )
        out.append(case(f"sr1/{name}", "sr1", [p.x0], solution(r), rtol=1e-5, atol=loose, problem=name))
    return out
