"""Projected gradients against scipy's L-BFGS-B solutions of box-constrained standard problems."""

import numpy as np
import scipy
import scipy.optimize as so
from omai_oracle.fixture import Case, case
from omai_oracle.optim_problems import SMOOTH

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    for name, bounds in [
        ("rosen", [(-2.0, 0.5), (0.0, 2.0)]),
        ("beale", [(-4.5, 2.5), (-4.5, 0.4)]),
        ("spdQuadratic", [(0.0, np.inf)] * 3),
    ]:
        p = SMOOTH[name]
        r = so.minimize(p.f, p.x0, method="L-BFGS-B", jac=p.grad, bounds=bounds, options={"gtol": 1e-12, "ftol": 1e-15})
        out.append(
            case(
                f"projectedGradient/{name}",
                "projectedGradient",
                [p.x0, np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])],
                np.concatenate([np.asarray(r.x, dtype=np.float64), [float(r.fun)]]),
                rtol=1e-6,
                atol=1e-7,
                problem=name,
            )
        )
    return out
