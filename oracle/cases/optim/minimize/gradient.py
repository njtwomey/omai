"""First-order methods against scipy.optimize.minimize: nonlinear conjugate gradients (a port of method 'CG', compared
on solutions and iteration counts) and the unique minimisers gradient and coordinate descent must reach."""

import numpy as np
import scipy
import scipy.optimize as so
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.optim_problems import PROBLEMS

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    for name in ["rosen", "rosen5", "beale", "himmelblau", "quadratic", "wood"]:
        p = PROBLEMS[name]
        r = so.minimize(p.f, p.x0, method="CG", jac=p.grad, options={"gtol": 1e-8})
        out.append(
            case(
                f"conjugateGradient/{name}",
                "conjugateGradient",
                [p.x0],
                np.concatenate([np.asarray(r.x, dtype=np.float64), [float(r.fun)]]),
                rtol=1e-6,
                atol=1e-7,
                problem=name,
                nit=int(r.nit),
            )
        )
    # Problems whose minimum is reached from the standard start whatever the path (Himmelblau has four minima).
    for name in ["quadratic", "beale"]:
        p = PROBLEMS[name]
        r = so.minimize(p.f, p.x0, method="BFGS", jac=p.grad, options={"gtol": 1e-10})
        for fn in ["gradientDescent", "coordinateDescent"]:
            out.append(
                case(
                    f"{fn}/{name}",
                    fn,
                    [p.x0],
                    np.concatenate([np.asarray(r.x, dtype=np.float64), [float(r.fun)]]),
                    rtol=1e-5,
                    atol=1e-6,
                    problem=name,
                )
            )
    return out
