"""SLSQP against scipy.optimize.minimize(method='SLSQP') on constrained test problems (unique solutions, compared
directly)."""

import numpy as np
import scipy
from omai_oracle.fixture import Case, case
from omai_oracle.optim_problems import CONSTRAINED, slsqp_solution

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    for name, (_f, _eq, _ineq, bounds, x0) in CONSTRAINED.items():
        r = slsqp_solution(name)
        lower = [b[0] for b in bounds] if bounds is not None else []
        upper = [b[1] for b in bounds] if bounds is not None else []
        out.append(
            case(
                f"slsqp/{name}",
                "slsqp",
                [np.array(x0), np.array(lower), np.array(upper)],
                np.concatenate([np.asarray(r.x, dtype=np.float64), [float(r.fun)]]),
                rtol=1e-6,
                atol=1e-7,
                problem=name,
            )
        )
    return out
