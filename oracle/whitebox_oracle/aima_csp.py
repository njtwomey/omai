"""A typed port of the arc consistency of aima-python (csp.py, github.com/aimacode/aima-python): `AC3` with `revise`
over a CSP given by variables, domains, neighbours and a binary constraint function, and the problem instances
`australia_csp`, `Sudoku` with aima-python's `easy1` and `harder1` grids, and `NQueensCSP`. AC-3 reaches the unique
largest arc-consistent domains whatever its queue order, so the fixed point is what is compared."""

from collections.abc import Callable

type Value = str | int
"""A value of a ported CSP: a colour, a digit character or a row."""

type ConstraintFn = Callable[[str, Value, str, Value], bool]
"""aima-python's `constraints(A, a, B, b)`."""


def ac3(
    variables: list[str],
    domains: dict[str, list[Value]],
    neighbours: dict[str, list[str]],
    constraints: ConstraintFn,
) -> tuple[bool, dict[str, list[Value]]]:
    """aima-python's `AC3` (with `revise`): whether the CSP is arc consistent, and the pruned domains."""
    current = {v: list(domains[v]) for v in variables}
    queue = [(xi, xk) for xi in variables for xk in neighbours[xi]]
    while queue:
        xi, xj = queue.pop()
        revised = False
        for x in current[xi][:]:
            if all(not constraints(xi, x, xj, y) for y in current[xj]):
                current[xi].remove(x)
                revised = True
        if revised:
            if not current[xi]:
                return False, current
            for xk in neighbours[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True, current


def different_values(a: str, x: Value, b: str, y: Value) -> bool:
    """aima-python's `different_values_constraint`."""
    return x != y


AUSTRALIA_NEIGHBOURS: dict[str, list[str]] = {
    "SA": ["WA", "NT", "Q", "NSW", "V"],
    "WA": ["SA", "NT"],
    "NT": ["SA", "WA", "Q"],
    "Q": ["SA", "NT", "NSW"],
    "NSW": ["SA", "Q", "V"],
    "V": ["SA", "NSW"],
    "T": [],
}
"""aima-python's `australia_csp` neighbours (parse_neighbors('SA: WA NT Q NSW V; NT: WA Q; NSW: Q V; T: '))."""

EASY1 = "..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3.."
"""aima-python's `easy1` Sudoku (solved by AC-3 alone)."""

HARDER1 = "4173698.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"
"""aima-python's `harder1` Sudoku."""


def sudoku_units() -> list[list[int]]:
    """The 27 rows, columns and boxes of cells 0 .. 80."""
    rows = [[9 * r + c for c in range(9)] for r in range(9)]
    cols = [[9 * r + c for r in range(9)] for c in range(9)]
    boxes = [[9 * (3 * br + r) + 3 * bc + c for r in range(3) for c in range(3)] for br in range(3) for bc in range(3)]
    return rows + cols + boxes


def sudoku(grid: str) -> tuple[list[str], dict[str, list[Value]], dict[str, list[str]]]:
    """aima-python's `Sudoku(grid)`: cell variables, digit-character domains and unit neighbours."""
    variables = [str(i) for i in range(81)]
    domains: dict[str, list[Value]] = {
        str(i): [ch] if ch in "123456789" else list("123456789") for i, ch in enumerate(grid)
    }
    neighbours: dict[str, set[str]] = {v: set() for v in variables}
    for unit in sudoku_units():
        for a in unit:
            for b in unit:
                if a != b:
                    neighbours[str(a)].add(str(b))
    return variables, domains, {v: sorted(n, key=int) for v, n in neighbours.items()}


def queens_constraint(a: str, x: Value, b: str, y: Value) -> bool:
    """aima-python's `queen_constraint` on column names and rows."""
    ca, cb = int(a), int(b)
    return ca == cb or (x != y and int(x) + ca != int(y) + cb and int(x) - ca != int(y) - cb)
