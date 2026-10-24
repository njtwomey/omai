"""MaxSAT against python-sat's RC2 (core-guided, exact): optimum costs of random weighted partial MaxSAT instances
and of unweighted MaxSAT over unsatisfiable random 3-CNF, plus an instance whose hard part is unsatisfiable."""

from importlib.metadata import version

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"python-sat": version("python-sat")}


def clause(g: object, n: int, k: int) -> list[int]:
    """A random clause of k distinct variables."""
    import numpy as np

    assert isinstance(g, np.random.Generator)
    vs = g.choice(n, size=k, replace=False) + 1
    signs = g.integers(0, 2, size=k) * 2 - 1
    return [int(v * s) for v, s in zip(vs, signs, strict=True)]


def optimum(hard: list[list[int]], soft: list[tuple[list[int], int]]) -> int | None:
    """RC2's optimal cost, or None when the hard clauses are unsatisfiable."""
    w = WCNF()
    for c in hard:
        w.append(c)
    for c, weight in soft:
        w.append(c, weight=weight)
    with RC2(w) as r:
        model = r.compute()
        return None if model is None else int(r.cost)


def cases() -> list[Case]:
    out: list[Case] = []
    for seed, n, nh, ns in [(0, 6, 4, 10), (1, 8, 6, 14), (2, 10, 8, 18), (3, 12, 10, 22), (4, 14, 6, 30)]:
        g = rng(seed)
        hard = [clause(g, n, 3) for _ in range(nh)]
        soft = [(clause(g, n, int(g.integers(1, 3))), int(g.integers(1, 6))) for _ in range(ns)]
        cost = optimum(hard, soft)
        out.append(
            case(
                f"weighted/{seed}",
                "maxsat",
                [],
                cost,
                numVars=n,
                hard=hard,
                soft=[{"clause": c, "weight": w} for c, w in soft],
                cost=cost,
            )
        )
    for seed, n, m in [(5, 6, 40), (6, 8, 50), (7, 10, 60)]:
        g = rng(seed)
        soft = [(clause(g, n, 3), 1) for _ in range(m)]
        cost = optimum([], soft)
        out.append(
            case(
                f"unweighted/{seed}",
                "maxsat",
                [],
                cost,
                numVars=n,
                hard=[],
                soft=[{"clause": c, "weight": w} for c, w in soft],
                cost=cost,
            )
        )
    hard = [[1], [-1, 2], [-2]]
    out.append(
        case("hard-unsat", "maxsat", [], None, numVars=2, hard=hard, soft=[{"clause": [1, 2], "weight": 1}], cost=None)
    )
    return out
