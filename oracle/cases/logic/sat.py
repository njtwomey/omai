"""SAT against python-sat: satisfiability (Glucose 3) of random 3-CNF near the phase transition, pigeonhole and graph
colouring formulas; exact model counts by enumeration for small formulas; every minimal unsatisfiable subset of small
unsatisfiable formulas (by brute force over clause subsets, checked with Glucose) and one found by MUSX; DIMACS
parsing by pysat's CNF reader."""

from importlib.metadata import version
from itertools import combinations

from pysat.examples.musx import MUSX
from pysat.formula import CNF, WCNF
from pysat.solvers import Solver
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"python-sat": version("python-sat")}


def random_3cnf(seed: int, n: int, m: int) -> list[list[int]]:
    """m random clauses of 3 distinct variables with random signs."""
    g = rng(seed)
    out: list[list[int]] = []
    for _ in range(m):
        vs = g.choice(n, size=3, replace=False) + 1
        signs = g.integers(0, 2, size=3) * 2 - 1
        out.append([int(v * s) for v, s in zip(vs, signs, strict=True)])
    return out


def pigeonhole(holes: int) -> tuple[int, list[list[int]]]:
    """holes + 1 pigeons in `holes` holes (unsatisfiable): x(p, h) = p * holes + h + 1."""
    pigeons = holes + 1

    def x(p: int, h: int) -> int:
        return p * holes + h + 1

    clauses = [[x(p, h) for h in range(holes)] for p in range(pigeons)]
    for h in range(holes):
        for p, q in combinations(range(pigeons), 2):
            clauses.append([-x(p, h), -x(q, h)])
    return pigeons * holes, clauses


def colouring(edges: list[tuple[int, int]], nodes: int, colours: int) -> tuple[int, list[list[int]]]:
    """Proper k-colouring: c(v, k) = v * colours + k + 1."""

    def c(v: int, k: int) -> int:
        return v * colours + k + 1

    clauses = [[c(v, k) for k in range(colours)] for v in range(nodes)]
    for v in range(nodes):
        for k, j in combinations(range(colours), 2):
            clauses.append([-c(v, k), -c(v, j)])
    for a, b in edges:
        for k in range(colours):
            clauses.append([-c(a, k), -c(b, k)])
    return nodes * colours, clauses


def satisfiable(clauses: list[list[int]]) -> bool:
    """Glucose 3."""
    with Solver(name="g3", bootstrap_with=clauses) as s:
        return bool(s.solve())


def count(n: int, clauses: list[list[int]]) -> int:
    """Models over variables 1..n by enumeration (free variables are fixed by a tautology on each)."""
    padded = [*clauses, *([v, -v] for v in range(1, n + 1))]
    with Solver(name="g3", bootstrap_with=padded) as s:
        total = 0
        for model in s.enum_models() or []:
            total += 1
            if len(model) < n:
                raise RuntimeError("partial model")
        return total


def all_muses(clauses: list[list[int]]) -> list[list[int]]:
    """Every minimal unsatisfiable subset (0-based indices), by size then lexicographically."""
    out: list[list[int]] = []
    for k in range(1, len(clauses) + 1):
        for subset in combinations(range(len(clauses)), k):
            if any(set(m) <= set(subset) for m in out):
                continue
            if not satisfiable([clauses[i] for i in subset]):
                out.append(list(subset))
    return out


def musx(clauses: list[list[int]]) -> list[int]:
    """One MUS by MUSX (deletion-based), 0-based."""
    w = WCNF()
    for c in clauses:
        w.append(c, weight=1)
    return sorted(i - 1 for i in MUSX(w, verbosity=0).compute() or [])


PETERSEN = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 0),
    (0, 5),
    (1, 6),
    (2, 7),
    (3, 8),
    (4, 9),
    (5, 7),
    (7, 9),
    (9, 6),
    (6, 8),
    (8, 5),
]
K4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def cases() -> list[Case]:
    out: list[Case] = []
    for seed, n, m in [
        (0, 8, 20),
        (1, 10, 43),
        (2, 10, 50),
        (3, 12, 51),
        (4, 14, 60),
        (5, 20, 85),
        (6, 30, 128),
        (7, 40, 170),
        (8, 50, 213),
        (9, 12, 70),
    ]:
        clauses = random_3cnf(seed, n, m)
        sat = satisfiable(clauses)
        extra = {"count": count(n, clauses)} if n <= 14 else {}
        out.append(case(f"random3/{seed}/{n}x{m}", "sat", [], sat, numVars=n, clauses=clauses, sat=sat, **extra))
    for holes in [2, 3, 4]:
        n, clauses = pigeonhole(holes)
        out.append(case(f"pigeonhole/{holes}", "sat", [], False, numVars=n, clauses=clauses, sat=False))
    for name, edges, nodes, k in [
        ("petersen3", PETERSEN, 10, 3),
        ("petersen2", PETERSEN, 10, 2),
        ("k4-3", K4, 4, 3),
        ("k4-4", K4, 4, 4),
    ]:
        n, clauses = colouring(edges, nodes, k)
        out.append(
            case(
                f"colouring/{name}",
                "sat",
                [],
                satisfiable(clauses),
                numVars=n,
                clauses=clauses,
                sat=satisfiable(clauses),
                count=count(n, clauses),
            )
        )
    for seed, n, m in [(10, 4, 9), (11, 5, 11), (12, 4, 10), (13, 3, 9), (14, 5, 12)]:
        clauses = random_3cnf(seed, n, m) if n >= 3 else []
        clauses = [c[: 1 + (i % 3)] for i, c in enumerate(clauses)]
        sat = satisfiable(clauses)
        out.append(
            case(
                f"cores/{seed}",
                "core",
                [],
                sat,
                numVars=n,
                clauses=clauses,
                sat=sat,
                muses=[] if sat else all_muses(clauses),
                musx=[] if sat else musx(clauses),
            )
        )
    for text in ["c tiny\np cnf 3 2\n1 -3 0\n2 3 -1 0\n", "p cnf 5 3\n1 2 0\nc mid\n-5 4 0\n3 0\n"]:
        f = CNF(from_string=text)
        out.append(case(f"dimacs/{len(out)}", "dimacs", [], f.clauses, text=text, numVars=f.nv, clauses=f.clauses))
    return out
