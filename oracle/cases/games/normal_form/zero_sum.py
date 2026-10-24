"""Zero-sum matrix games against scipy.optimize.linprog (HiGHS): the value and both optimal strategies (unique for
the generic random games used here), from the textbook linear program max v subject to M^T x >= v, x in the simplex;
and security (maxmin) levels of general-sum games the same way."""

import numpy as np
import scipy
import scipy.optimize as so
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def maximin(rows: FloatArray) -> tuple[FloatArray, float]:
    """max_x min_s sum_a x_a rows[a, s] by linprog: variables (x, v), minimise -v."""
    k, cols = rows.shape
    c = np.append(np.zeros(k), -1.0)
    a_ub = np.hstack([-rows.T, np.ones((cols, 1))])
    a_eq = np.append(np.ones(k), 0.0).reshape(1, -1)
    bounds = [(0.0, None)] * k + [(None, None)]
    r = so.linprog(c, A_ub=a_ub, b_ub=np.zeros(cols), A_eq=a_eq, b_eq=np.array([1.0]), bounds=bounds)
    z = np.asarray(r.x, dtype=np.float64)
    return z[:k], float(z[k])


def cases() -> list[Case]:
    out: list[Case] = []
    games: dict[str, FloatArray] = {
        "matching-pennies": np.array([[1.0, -1.0], [-1.0, 1.0]]),
        "rock-paper-scissors": np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]),
        "saddle-point": np.array([[3.0, 1.0, 4.0], [2.0, 0.0, 1.0], [5.0, 1.5, 2.0]]),
        "weighted-rps": np.array([[0.0, -2.0, 1.0], [2.0, 0.0, -3.0], [-1.0, 3.0, 0.0]]),
    }
    for seed, (m, n) in enumerate([(2, 3), (3, 3), (4, 4), (3, 5), (6, 4), (8, 8)]):
        games[f"random-{m}x{n}"] = rng(200 + seed).uniform(-1, 1, size=(m, n))
    for name, a in games.items():
        x, v = maximin(a)
        y, w = maximin(-a.T)
        out.append(
            case(name, "solveZeroSum", [a], np.array(v), row=x.tolist(), column=y.tolist(), negated=w, atol=1e-8)
        )
    general: dict[str, tuple[FloatArray, FloatArray]] = {
        "battle-of-the-sexes": (np.array([[3.0, 0.0], [0.0, 2.0]]), np.array([[2.0, 0.0], [0.0, 3.0]])),
        "chicken": (np.array([[6.0, 2.0], [7.0, 0.0]]), np.array([[6.0, 7.0], [2.0, 0.0]])),
        "random-3x4": (rng(300).uniform(-1, 1, size=(3, 4)), rng(301).uniform(-1, 1, size=(3, 4))),
    }
    for name, (a, b) in general.items():
        _, v0 = maximin(a)
        _, v1 = maximin(b.T)
        out.append(case(f"maximin/{name}", "maximin", [a, b], np.array([v0, v1]), atol=1e-8))
    return out
