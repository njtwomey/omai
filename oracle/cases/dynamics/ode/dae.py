"""Index-1 DAEs against the equivalent ODE solved by scipy's Radau at tight tolerance.

Robertson's problem with the conservation law y1 + y2 + y3 = 1 as the algebraic equation; whitebox-js solves the
DAE (singular mass matrix) with Radau and BDF, so the comparison is at the DAE solver's tolerance.
"""

import numpy as np
import scipy
import scipy.integrate as si
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def robertson(_t: float, y: FloatArray) -> FloatArray:
    return np.array(
        [
            -0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
            3e7 * y[1] ** 2,
        ]
    )


def cases() -> list[Case]:
    times = [0.4, 4.0, 40.0]
    ode = si.solve_ivp(robertson, (0.0, 40.0), [1.0, 0.0, 0.0], method="Radau", rtol=1e-10, atol=1e-14, t_eval=times)
    return [
        case(f"robertson/{method}", "robertson_dae", [], ode.y.T, rtol=1e-5, atol=1e-9, method=method)
        for method in ["Radau", "BDF"]
    ]
