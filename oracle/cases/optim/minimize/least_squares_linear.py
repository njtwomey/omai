"""Linear least squares against scipy.optimize: nnls (Lawson-Hanson) and lsq_linear with method 'bvls'."""

import numpy as np
import scipy
import scipy.optimize as so
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(5)
    for i in range(3):
        a = g.normal(size=(8, 5))
        b = g.normal(size=8)
        x, rnorm = so.nnls(a, b)
        out.append(case(f"nnls/{i}", "nnls", [a, b], np.concatenate([x, [rnorm]]), rtol=1e-10, atol=1e-12))
        lb = np.full(5, -0.3)
        ub = np.full(5, 0.4)
        r = so.lsq_linear(a, b, bounds=(lb, ub), method="bvls", tol=1e-12)
        out.append(
            case(
                f"bvls/{i}",
                "lsqLinear",
                [a, b, lb, ub],
                np.concatenate([np.asarray(r.x, dtype=np.float64), [float(r.cost)]]),
                rtol=1e-10,
                atol=1e-12,
            )
        )
    return out
