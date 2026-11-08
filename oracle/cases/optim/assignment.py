"""Assignment problems against scipy.optimize.linear_sum_assignment."""

import numpy as np
import scipy
import scipy.optimize as so
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def matrices() -> dict[str, tuple[FloatArray, bool]]:
    """name -> (cost matrix, maximize)."""
    g = rng(3)
    return {
        "square": (g.uniform(0.0, 10.0, size=(6, 6)), False),
        "wide": (g.uniform(0.0, 10.0, size=(4, 7)), False),
        "tall": (g.uniform(0.0, 10.0, size=(7, 4)), False),
        "maximize": (g.uniform(-5.0, 5.0, size=(5, 5)), True),
        "integer": (g.integers(0, 50, size=(8, 8)).astype(np.float64) + 0.001 * np.arange(64).reshape(8, 8), False),
        "single": (np.array([[3.0]]), False),
    }


def cases() -> list[Case]:
    out: list[Case] = []
    for name, (cost, maximize) in matrices().items():
        rows, cols = so.linear_sum_assignment(cost, maximize=maximize)
        output = {
            "rowInd": np.asarray(rows, dtype=np.int32),
            "colInd": np.asarray(cols, dtype=np.int32),
            "cost": np.array(cost[rows, cols].sum()),
        }
        out.append(case(name, "linearSumAssignment", [cost], output, rtol=1e-12, atol=1e-12, maximize=maximize))
    return out
