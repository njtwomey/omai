"""Constraint satisfaction against python-constraint (every solution of n-queens, Australia map colouring, a
cryptarithm and 4x4 Sudoku; the unique solution of aima-python's harder1 Sudoku) and against aima-python's AC-3 (a
typed port in whitebox_oracle.aima_csp: arc-consistent domains of partially assigned map colourings, n-queens and the
easy1 Sudoku, which AC-3 solves outright)."""

from collections.abc import Callable
from importlib.metadata import version

import constraint as pc
import numpy as np
from whitebox_oracle.aima_csp import (
    AUSTRALIA_NEIGHBOURS,
    EASY1,
    HARDER1,
    Value,
    ac3,
    different_values,
    queens_constraint,
    sudoku,
    sudoku_units,
)
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {
    "python-constraint2": version("python-constraint2"),
    "aima-python": "port of csp.py (2024)",
}

COLOURS = ["red", "green", "blue"]
REGIONS = ["WA", "NT", "SA", "Q", "NSW", "V", "T"]
BORDERS = sorted({tuple(sorted((a, b))) for a, ns in AUSTRALIA_NEIGHBOURS.items() for b in ns})


def queens_pair(d: int) -> Callable[[int, int], bool]:
    """Two queens d columns apart do not attack each other."""

    def safe(a: int, b: int) -> bool:
        return a != b and abs(a - b) != d

    return safe


def differ(x: str, y: str) -> bool:
    """Neighbouring regions take different colours."""
    return x != y


def nonzero(x: int) -> bool:
    """A leading digit is not zero."""
    return x != 0


def column(a: int, b: int, carry_in: int, digit: int, carry_out: int) -> bool:
    """One column of an addition: a + b + carry in = digit + 10 carry out."""
    return a + b + carry_in == digit + 10 * carry_out


def units_column(d: int, e: int, y: int, carry: int) -> bool:
    """The units column: D + E = Y + 10 C1."""
    return d + e == y + 10 * carry


def leading_column(s: int, m: int, carry: int, o: int) -> bool:
    """The thousands column, whose carry is M: S + M + C3 = O + 10 M."""
    return s + m + carry == o + 10 * m


def queens_solutions(n: int) -> list[list[int]]:
    """Every placement of n queens, one per column, by python-constraint."""
    p = pc.Problem()
    cols = list(range(n))
    p.addVariables(cols, list(range(n)))
    for i in cols:
        for j in cols:
            if i < j:
                p.addConstraint(queens_pair(j - i), (i, j))
    return sorted([s[c] for c in cols] for s in p.getSolutions())


def australia_solutions() -> list[list[str]]:
    """Every three-colouring of Australia by python-constraint, in REGIONS order."""
    p = pc.Problem()
    p.addVariables(REGIONS, COLOURS)
    for a, b in BORDERS:
        p.addConstraint(differ, (a, b))
    return sorted([s[r] for r in REGIONS] for s in p.getSolutions())


def cryptarithm() -> list[dict[str, int]]:
    """SEND + MORE = MONEY by python-constraint, in column form with carries."""
    p = pc.Problem()
    letters = "SENDMORY"
    p.addVariables(list(letters), list(range(10)))
    p.addVariables(["C1", "C2", "C3"], [0, 1])
    p.addConstraint(pc.AllDifferentConstraint(), list(letters))
    p.addConstraint(nonzero, ["S"])
    p.addConstraint(nonzero, ["M"])
    p.addConstraint(units_column, ["D", "E", "Y", "C1"])
    p.addConstraint(column, ["N", "R", "C1", "E", "C2"])
    p.addConstraint(column, ["E", "O", "C2", "N", "C3"])
    p.addConstraint(leading_column, ["S", "M", "C3", "O"])
    return [dict(sorted((str(k), int(v)) for k, v in s.items())) for s in p.getSolutions()]


def sudoku4_solutions(grid: str) -> list[str]:
    """Every completion of a 4x4 Sudoku by python-constraint."""
    p = pc.Problem()
    for i, ch in enumerate(grid):
        p.addVariable(i, [int(ch)] if ch != "." else [1, 2, 3, 4])
    rows = [[4 * r + c for c in range(4)] for r in range(4)]
    cols = [[4 * r + c for r in range(4)] for c in range(4)]
    boxes = [[4 * (2 * br + r) + 2 * bc + c for r in range(2) for c in range(2)] for br in range(2) for bc in range(2)]
    for unit in rows + cols + boxes:
        p.addConstraint(pc.AllDifferentConstraint(), unit)
    return sorted("".join(str(s[i]) for i in range(16)) for s in p.getSolutions())


def sudoku9_solution(grid: str) -> str:
    """The unique completion of a 9x9 Sudoku by python-constraint."""
    p = pc.Problem()
    for i, ch in enumerate(grid):
        p.addVariable(i, [int(ch)] if ch in "123456789" else list(range(1, 10)))
    for unit in sudoku_units():
        p.addConstraint(pc.AllDifferentConstraint(), unit)
    s = p.getSolution()
    if s is None:
        raise ValueError("the Sudoku has no solution")
    return "".join(str(s[i]) for i in range(81))


def cases() -> list[Case]:
    out: list[Case] = []
    for n in range(4, 9):
        sols = queens_solutions(n)
        out.append(
            case(
                f"queens/{n}/solutions",
                "queens",
                [],
                np.array(0.0),
                n=n,
                count=len(sols),
                solutions=sols if n <= 6 else [],
                first=sols[0],
            )
        )
    sols = australia_solutions()
    out.append(
        case(
            "australia/solutions",
            "australia",
            [],
            np.array(0.0),
            regions=REGIONS,
            borders=[list(b) for b in BORDERS],
            colours=COLOURS,
            solutions=sols,
        )
    )
    out.append(case("send-more-money", "cryptarithm", [], np.array(0.0), solutions=cryptarithm()))
    for grid in ["1...3.1...4..2..", "....3412........", "1..4.4..2..3...2"]:
        out.append(case(f"sudoku4/{grid}", "sudoku4", [], np.array(0.0), grid=grid, solutions=sudoku4_solutions(grid)))
    out.append(case("sudoku9/harder1", "sudoku9", [], np.array(0.0), grid=HARDER1, solution=sudoku9_solution(HARDER1)))
    neighbours = {
        r: sorted(set(AUSTRALIA_NEIGHBOURS[r]) | {a for a, ns in AUSTRALIA_NEIGHBOURS.items() if r in ns})
        for r in REGIONS
    }
    for fixed in [{"WA": "red"}, {"WA": "red", "Q": "green"}, {"WA": "red", "V": "blue"}, {"SA": "blue", "NT": "red"}]:
        domains: dict[str, list[Value]] = {r: [fixed[r]] if r in fixed else list(COLOURS) for r in REGIONS}
        consistent, pruned = ac3(REGIONS, domains, neighbours, different_values)
        out.append(
            case(
                "australia/ac3/" + ",".join(f"{k}={v}" for k, v in fixed.items()),
                "ac3Australia",
                [],
                np.array(0.0),
                regions=REGIONS,
                borders=[list(b) for b in BORDERS],
                colours=COLOURS,
                fixed=fixed,
                consistent=consistent,
                domains=pruned if consistent else None,
            )
        )
    for n, row in [(4, 0), (5, 0), (6, 1), (8, 3)]:
        cols = [str(c) for c in range(n)]
        domains = {c: [row] if c == "0" else list(range(n)) for c in cols}
        nbrs = {c: [d for d in cols if d != c] for c in cols}
        consistent, pruned = ac3(cols, domains, nbrs, queens_constraint)
        out.append(
            case(
                f"queens/{n}/ac3/row{row}",
                "ac3Queens",
                [],
                np.array(0.0),
                n=n,
                row=row,
                consistent=consistent,
                domains=pruned if consistent else None,
            )
        )
    variables, domains, nbrs = sudoku(EASY1)
    consistent, pruned = ac3(variables, domains, nbrs, different_values)
    out.append(
        case(
            "sudoku9/easy1/ac3",
            "ac3Sudoku",
            [],
            np.array(0.0),
            grid=EASY1,
            consistent=consistent,
            solution="".join(str(pruned[v][0]) if len(pruned[v]) == 1 else "." for v in variables),
        )
    )
    return out
