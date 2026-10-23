"""L-BFGS-B against scipy.optimize.minimize(method='L-BFGS-B') on bounded standard problems.

Both run to tight tolerances, so their solutions agree to the tolerance of the method rather than iterate by iterate.
"""

import numpy as np
import scipy
import scipy.optimize as so
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.optim_problems import PROBLEMS

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

INF = float("inf")

# (problem, start, bounds as [lower, upper] pairs with infinities)
SETUPS: list[tuple[str, list[float], list[tuple[float, float]]]] = [
    ("rosen", [-1.2, 1.0], [(-2.0, 0.5), (0.0, 2.0)]),
    ("rosen", [-1.2, 1.0], [(-INF, INF), (-INF, INF)]),
    ("rosen5", [1.3, 0.7, 0.8, 1.9, 1.2], [(0.0, 2.0), (0.0, 0.9), (-INF, INF), (1.5, 3.0), (-1.0, INF)]),
    ("beale", [1.0, 1.0], [(-4.5, 2.5), (-4.5, 0.4)]),
    ("himmelblau", [0.0, 0.0], [(0.5, 2.5), (0.5, 1.5)]),
    ("quadratic", [1.0, 1.0, 1.0], [(0.0, INF), (0.0, INF), (0.0, INF)]),
    ("quadratic", [1.0, 1.0, 1.0], [(-INF, 0.2), (-0.5, 0.5), (0.0, 0.0)]),
    ("powell", [3.0, -1.0, 0.0, 1.0], [(1.0, 3.0), (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)]),
    ("wood", [-3.0, -1.0, -3.0, -1.0], [(-3.0, 0.8), (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)]),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for i, (name, x0, bounds) in enumerate(SETUPS):
        p = PROBLEMS[name]
        r = so.minimize(
            p.f, np.array(x0), method="L-BFGS-B", jac=p.grad, bounds=bounds, options={"gtol": 1e-10, "ftol": 1e-15}
        )
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        out.append(
            case(
                f"lbfgsb/{i}/{name}",
                "lbfgsb",
                [np.array(x0), lower, upper],
                np.concatenate([np.asarray(r.x, dtype=np.float64), [float(r.fun)]]),
                rtol=1e-5,
                atol=1e-6,
                problem=name,
            )
        )
    return out
