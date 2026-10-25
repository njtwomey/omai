"""Newton methods against scipy.optimize.minimize: Newton-CG on standard problems, and the minimisers the modified
Newton method must reach (scipy's trust-exact solutions on the problems where the minimum is unique)."""

from collections.abc import Callable

import numpy as np
import scipy
import scipy.optimize as so
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray
from mdml_oracle.optim_problems import SMOOTH, Smooth

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def products(p: Smooth) -> Callable[[FloatArray, FloatArray], FloatArray]:
    """Hessian-vector products of a problem."""

    def hessp(x: FloatArray, v: FloatArray) -> FloatArray:
        return p.hess(x) @ v

    return hessp


def newton_cg(p: Smooth, hessian: str) -> so.OptimizeResult:
    if hessian == "products":
        return so.minimize(p.f, p.x0, method="Newton-CG", jac=p.grad, hessp=products(p), options={"xtol": 1e-8})
    return so.minimize(p.f, p.x0, method="Newton-CG", jac=p.grad, hess=p.hess, options={"xtol": 1e-8})


def cases() -> list[Case]:
    out: list[Case] = []
    # Newton-CG reaches its iteration cap on Powell's singular function and Wood's function, so those are left out.
    for name in ["rosen", "rosen5", "beale", "himmelblau", "spdQuadratic"]:
        p = SMOOTH[name]
        for hessian in ["products", "dense"]:
            r = newton_cg(p, hessian)
            out.append(
                case(
                    f"newtonCg/{hessian}/{name}",
                    "newtonCg",
                    [p.x0],
                    np.concatenate([np.asarray(r.x), [float(r.fun)]]),
                    rtol=1e-6,
                    atol=1e-7,
                    problem=name,
                    hessian=hessian,
                    nit=int(r.nit),
                )
            )
    # Unique minimisers (convex, or a single minimum reached from the standard start).
    for name in ["rosen", "rosen5", "spdQuadratic", "wood"]:
        p = SMOOTH[name]
        r = so.minimize(p.f, p.x0, method="trust-exact", jac=p.grad, hess=p.hess, options={"gtol": 1e-10})
        out.append(
            case(
                f"newton/{name}",
                "newton",
                [p.x0],
                np.concatenate([np.asarray(r.x), [float(r.fun)]]),
                rtol=1e-6,
                atol=1e-7,
                problem=name,
            )
        )
    return out
