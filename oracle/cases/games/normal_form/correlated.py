"""Correlated and coarse correlated equilibria against scipy.optimize.linprog (HiGHS): the optimal welfare and
egalitarian values over the equilibrium polytope (the optimal distribution need not be unique, the value is), for
Aumann's chicken (whose welfare-optimal correlated equilibrium is unique), the battle of the sexes and random two- and
three-player games."""

import itertools

import numpy as np
import scipy
import scipy.optimize as so
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def incentive_rows(payoffs: list[FloatArray], coarse: bool) -> list[FloatArray]:
    """One row per (player, recommended action or none, deviation): the gain of deviating at each profile."""
    shape = payoffs[0].shape
    profiles = list(itertools.product(*[range(k) for k in shape]))
    rows: list[FloatArray] = []
    for p, u in enumerate(payoffs):
        recommended: list[int | None] = [None] if coarse else list(range(shape[p]))
        for a in recommended:
            for d in range(shape[p]):
                if d == a:
                    continue
                row = np.zeros(len(profiles))
                for k, s in enumerate(profiles):
                    if a is not None and s[p] != a:
                        continue
                    t = list(s)
                    t[p] = d
                    row[k] = u[tuple(t)] - u[s]
                rows.append(row)
    return rows


def optimum(payoffs: list[FloatArray], coarse: bool, objective: str) -> float:
    """The best welfare or egalitarian value over the (coarse) correlated equilibria."""
    flat = [u.ravel() for u in payoffs]
    total = flat[0].size
    rows = incentive_rows(payoffs, coarse)
    if objective == "welfare":
        c = -np.sum(flat, axis=0)
        a_ub = np.array(rows)
        a_eq = np.ones((1, total))
        bounds: list[tuple[float | None, float | None]] = [(0.0, None) for _ in range(total)]
    else:
        c = np.append(np.zeros(total), -1.0)
        a_ub = np.vstack([np.hstack([np.array(rows), np.zeros((len(rows), 1))])] + [np.append(-u, 1.0) for u in flat])
        a_eq = np.append(np.ones(total), 0.0).reshape(1, -1)
        bounds = [(0.0, None) for _ in range(total)]
        bounds.append((None, None))
    r = so.linprog(c, A_ub=a_ub, b_ub=np.zeros(a_ub.shape[0]), A_eq=a_eq, b_eq=np.array([1.0]), bounds=bounds)
    return -float(np.asarray(r.fun, dtype=np.float64))


def cases() -> list[Case]:
    out: list[Case] = []
    games: dict[str, list[FloatArray]] = {
        "chicken": [np.array([[6.0, 2.0], [7.0, 0.0]]), np.array([[6.0, 7.0], [2.0, 0.0]])],
        "battle-of-the-sexes": [np.array([[3.0, 0.0], [0.0, 2.0]]), np.array([[2.0, 0.0], [0.0, 3.0]])],
        "random-3x3": [rng(400).uniform(-1, 1, size=(3, 3)), rng(401).uniform(-1, 1, size=(3, 3))],
        "random-2x2x2": [rng(402 + p).uniform(-1, 1, size=(2, 2, 2)) for p in range(3)],
    }
    for name, payoffs in games.items():
        for coarse in (False, True):
            kind = "coarse" if coarse else "correlated"
            for objective in ("welfare", "egalitarian"):
                value = optimum(payoffs, coarse, objective)
                out.append(
                    case(
                        f"{name}/{kind}/{objective}",
                        "correlatedEquilibrium",
                        payoffs,
                        np.array(value),
                        coarse=coarse,
                        objective=objective,
                        atol=1e-8,
                    )
                )
    return out
