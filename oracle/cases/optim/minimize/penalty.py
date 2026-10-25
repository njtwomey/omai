"""The penalty, barrier and augmented Lagrangian methods against scipy's SLSQP solutions of the constrained problems."""

import numpy as np
import scipy
from mdml_oracle.fixture import Case, case
from mdml_oracle.optim_problems import CONSTRAINED, slsqp_solution

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    for name, (_f, _eq, _ineq, bounds, x0) in CONSTRAINED.items():
        r = slsqp_solution(name)
        lower = [b[0] for b in bounds] if bounds is not None else []
        upper = [b[1] for b in bounds] if bounds is not None else []
        expected = np.concatenate([np.asarray(r.x, dtype=np.float64), [float(r.fun)]])
        for method, tol in [("augmentedLagrangian", 1e-6), ("penaltyMethod", 1e-4)]:
            out.append(
                case(
                    f"{method}/{name}",
                    method,
                    [np.array(x0), np.array(lower), np.array(upper)],
                    expected,
                    rtol=tol,
                    atol=tol,
                    problem=name,
                )
            )
        if name == "disc":
            out.append(
                case(
                    f"barrierMethod/{name}",
                    "barrierMethod",
                    [np.array(x0)],
                    expected,
                    rtol=1e-6,
                    atol=1e-7,
                    problem=name,
                )
            )
    return out
