"""Linear programs against scipy.optimize.linprog (HiGHS): optimal points, values, statuses and marginals."""

import numpy as np
import scipy
import scipy.optimize as so
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type Problem = tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]


def empty(n: int) -> tuple[FloatArray, FloatArray]:
    return np.zeros((0, n)), np.zeros(0)


def problems() -> dict[str, Problem]:
    """name -> (c, A_ub, b_ub, A_eq, b_eq, bounds as an (n, 2) array with infinities)."""
    inf = np.inf
    out: dict[str, Problem] = {}
    a0, b0 = empty(2)
    out["textbook"] = (
        np.array([-1.0, -2.0]),
        np.array([[1.0, 1.0], [1.0, 3.0]]),
        np.array([4.0, 6.0]),
        a0,
        b0,
        np.array([[0.0, inf], [0.0, inf]]),
    )
    out["mixed-bounds"] = (
        np.array([2.0, -3.0, 1.0, 0.5]),
        np.array([[1.0, 1.0, 1.0, 1.0], [-1.0, 2.0, 0.0, 1.0], [0.0, 1.0, -1.0, 0.0]]),
        np.array([10.0, 8.0, 3.0]),
        np.array([[1.0, 0.0, 1.0, -1.0]]),
        np.array([1.0]),
        np.array([[-2.0, 5.0], [-inf, 4.0], [-inf, inf], [0.0, 3.0]]),
    )
    out["equalities"] = (
        np.array([1.0, 2.0, 3.0, 1.0]),
        np.zeros((0, 4)),
        np.zeros(0),
        np.array([[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 2.0, 0.0]]),
        np.array([4.0, 1.0]),
        np.array([[0.0, inf]] * 4),
    )
    out["redundant"] = (
        np.array([1.0, 1.0, 2.0]),
        np.zeros((0, 3)),
        np.zeros(0),
        np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, -1.0, 0.0]]),
        np.array([3.0, 6.0, 0.5]),
        np.array([[0.0, inf]] * 3),
    )
    g = rng(11)
    n, m = 6, 4
    a = g.uniform(0.1, 2.0, size=(m, n))
    out["random"] = (
        -g.uniform(0.5, 2.0, size=n),
        a,
        a @ g.uniform(0.5, 1.5, size=n),
        g.uniform(-1.0, 1.0, size=(1, n)),
        np.array([0.3]),
        np.array([[0.0, 3.0]] * n),
    )
    out["negative-rhs"] = (
        np.array([3.0, 2.0]),
        np.array([[-1.0, -1.0], [-2.0, -1.0]]),
        np.array([-3.0, -4.0]),
        a0,
        b0,
        np.array([[0.0, inf], [0.0, inf]]),
    )
    out["infeasible"] = (
        np.array([1.0, 1.0]),
        np.array([[1.0, 1.0]]),
        np.array([1.0]),
        np.array([[1.0, 1.0]]),
        np.array([3.0]),
        np.array([[0.0, inf], [0.0, inf]]),
    )
    out["unbounded"] = (
        np.array([-1.0, 0.0]),
        np.array([[-1.0, 1.0]]),
        np.array([1.0]),
        a0,
        b0,
        np.array([[0.0, inf], [0.0, inf]]),
    )
    return out


STATUS = {0: "optimal", 1: "iteration limit", 2: "infeasible", 3: "unbounded"}


def cases() -> list[Case]:
    out: list[Case] = []
    for name, (c, aub, bub, aeq, beq, bounds) in problems().items():
        r = so.linprog(
            c,
            A_ub=aub if aub.size else None,
            b_ub=bub if bub.size else None,
            A_eq=aeq if aeq.size else None,
            b_eq=beq if beq.size else None,
            bounds=[(None if np.isinf(lo) else lo, None if np.isinf(hi) else hi) for lo, hi in bounds],
            method="highs",
        )
        status = STATUS[int(r.status)]
        output: dict[str, object] = {"status": status}
        if status == "optimal":
            output |= {
                "x": np.asarray(r.x),
                "fun": np.array(r.fun),
                "ineqlin": np.asarray(r.ineqlin.marginals) if aub.size else np.zeros(0),
                "eqlin": np.asarray(r.eqlin.marginals) if aeq.size else np.zeros(0),
                "lower": np.asarray(r.lower.marginals),
                "upper": np.asarray(r.upper.marginals),
            }
        out.append(case(name, "linprog", [c, aub, bub, aeq, beq, bounds], output, rtol=1e-9, atol=1e-9))
    return out
