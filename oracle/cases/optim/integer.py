"""Mixed-integer programs, knapsacks and set covers against scipy.optimize.milp (HiGHS)."""

import numpy as np
import scipy
import scipy.optimize as so
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def solve(
    c: FloatArray,
    aub: FloatArray,
    bub: FloatArray,
    aeq: FloatArray,
    beq: FloatArray,
    bounds: FloatArray,
    integrality: FloatArray,
) -> tuple[int, FloatArray, float]:
    constraints: list[so.LinearConstraint] = []
    if aub.size:
        constraints.append(so.LinearConstraint(aub, -np.inf, bub))
    if aeq.size:
        constraints.append(so.LinearConstraint(aeq, beq, beq))
    r = so.milp(
        c,
        integrality=integrality.astype(np.int64),
        bounds=so.Bounds(bounds[:, 0], bounds[:, 1]),
        constraints=constraints,
        options={"mip_rel_gap": 1e-12},
    )
    x = np.asarray(r.x) if r.x is not None else np.zeros(0)
    return int(r.status), x, float(r.fun) if r.fun is not None else float("nan")


def programs() -> dict[str, tuple[FloatArray, ...]]:
    """name -> (c, A_ub, b_ub, A_eq, b_eq, bounds, integrality)."""
    inf = np.inf
    out: dict[str, tuple[FloatArray, ...]] = {}
    out["textbook"] = (
        np.array([-5.0, -4.0]),
        np.array([[6.0, 4.0], [1.0, 2.0]]),
        np.array([24.0, 6.0]),
        np.zeros((0, 2)),
        np.zeros(0),
        np.array([[0.0, inf], [0.0, inf]]),
        np.array([1.0, 1.0]),
    )
    g = rng(5)
    n, m = 7, 4
    a = g.uniform(0.5, 3.0, size=(m, n))
    out["random-mixed"] = (
        -g.uniform(1.0, 3.0, size=n),
        a,
        np.round(a.sum(axis=1) * 1.7, 2),
        np.zeros((0, n)),
        np.zeros(0),
        np.array([[0.0, 4.0]] * n),
        np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
    )
    out["equality-negative"] = (
        np.array([1.0, -2.0, 3.0, 0.5]),
        np.array([[1.0, 1.0, 0.0, 1.0], [0.0, 1.0, -1.0, 2.0]]),
        np.array([6.5, 4.2]),
        np.array([[1.0, 0.0, 1.0, -1.0]]),
        np.array([0.5]),
        np.array([[-3.0, 3.0], [-2.0, 5.0], [-inf, inf], [0.0, 2.5]]),
        np.array([1.0, 1.0, 1.0, 0.0]),
    )
    out["infeasible"] = (
        np.array([1.0]),
        np.zeros((0, 1)),
        np.zeros(0),
        np.zeros((0, 1)),
        np.zeros(0),
        np.array([[0.2, 0.8]]),
        np.array([1.0]),
    )
    return out


def cases() -> list[Case]:
    out: list[Case] = []
    for name, (c, aub, bub, aeq, beq, bounds, integ) in programs().items():
        status, x, fun = solve(c, aub, bub, aeq, beq, bounds, integ)
        output: dict[str, object] = {"status": "optimal" if status == 0 else "infeasible"}
        if status == 0:
            output |= {"x": x, "fun": np.array(fun)}
        out.append(case(name, "milp", [c, aub, bub, aeq, beq, bounds, integ], output, rtol=1e-9, atol=1e-9))

    g = rng(9)
    for k, n in enumerate([10, 16]):
        values = np.round(g.uniform(1.0, 20.0, size=n), 3)
        weights = g.integers(1, 15, size=n).astype(np.float64)
        capacity = float(np.floor(weights.sum() / 2.5))
        status, x, fun = solve(
            -values,
            weights[None, :],
            np.array([capacity]),
            np.zeros((0, n)),
            np.zeros(0),
            np.array([[0.0, 1.0]] * n),
            np.ones(n),
        )
        items = np.flatnonzero(x > 0.5).astype(np.int32)
        out.append(
            case(
                f"knapsack/{k}",
                "knapsack",
                [values, weights],
                {"value": np.array(-fun), "items": items},
                capacity=capacity,
            )
        )

    for k, (n_sets, universe) in enumerate([(10, 8), (14, 12)]):
        incidence = (g.uniform(size=(universe, n_sets)) < 0.3).astype(np.float64)
        for e in range(universe):
            incidence[e, e % n_sets] = 1.0
        costs = np.round(g.uniform(1.0, 5.0, size=n_sets), 3)
        status, x, fun = solve(
            costs,
            -incidence,
            -np.ones(universe),
            np.zeros((0, n_sets)),
            np.zeros(0),
            np.array([[0.0, 1.0]] * n_sets),
            np.ones(n_sets),
        )
        out.append(case(f"set-cover/{k}", "setCover", [incidence, costs], {"cost": np.array(fun)}, rtol=1e-9))
    return out
