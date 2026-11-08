"""Answer sets against clingo: every stable model of small normal programs with default negation and integrity
constraints (even and odd loops, graph colouring, Hamiltonian cycles, the win-move game, a stratified program whose
single answer set is its perfect model)."""

from importlib.metadata import version

import clingo
from omai_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"clingo": version("clingo")}


def stable_models(src: str) -> list[list[str]]:
    """Every answer set, each sorted, sorted."""
    ctl = clingo.Control(["0", "--warn=none"])
    ctl.add("base", [], src.replace("\\=", "!=").replace("=<", "<="))
    ctl.ground([("base", [])])
    out: list[list[str]] = []

    def on_model(m: clingo.Model) -> None:
        out.append(sorted(str(s) for s in m.symbols(atoms=True)))

    ctl.solve(on_model=on_model)
    return sorted(out)


COLOUR = (
    "col(X, red) :- node(X), not col(X, green), not col(X, blue). "
    "col(X, green) :- node(X), not col(X, red), not col(X, blue). "
    "col(X, blue) :- node(X), not col(X, red), not col(X, green). "
    ":- edge(X, Y), col(X, C), col(Y, C)."
)

HAMILTON = (
    "in(X, Y) :- edge(X, Y), not out(X, Y). out(X, Y) :- edge(X, Y), not in(X, Y). "
    ":- in(X, Y), in(X, Z), Y \\= Z. :- in(X, Y), in(Z, Y), X \\= Z. "
    "reached(Y) :- in(1, Y). reached(Y) :- reached(X), in(X, Y). :- node(X), not reached(X)."
)

PROGRAMS: list[tuple[str, str]] = [
    ("even-loop", "a :- not b. b :- not a."),
    ("odd-loop", "p :- not q. q :- not r. r :- not p."),
    ("self-negation", "p :- not p."),
    ("constraint", "a :- not b. b :- not a. c :- a. :- c."),
    (
        "facts-and-defaults",
        "bird(tweety). bird(sam). penguin(sam). flies(X) :- bird(X), not ab(X). ab(X) :- penguin(X).",
    ),
    ("triangle-3col", "node(1). node(2). node(3). edge(1, 2). edge(2, 3). edge(1, 3). " + COLOUR),
    ("path-3col", "node(1). node(2). node(3). edge(1, 2). edge(2, 3). " + COLOUR),
    (
        "k4-3col",
        "node(1). node(2). node(3). node(4). edge(1, 2). edge(1, 3). edge(1, 4). edge(2, 3). edge(2, 4). edge(3, 4). "
        + COLOUR,
    ),
    (
        "hamilton",
        "node(1). node(2). node(3). node(4). edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 1). edge(1, 3). edge(3, 1). "
        + HAMILTON,
    ),
    ("win-move", "move(a, b). move(b, a). move(b, c). move(c, d). win(X) :- move(X, Y), not win(Y)."),
    ("stratified", "a(1). a(2). a(3). b(X) :- a(X), not c(X). c(2). d(X) :- a(X), not b(X)."),
    ("choice-pair", "p(1). p(2). in(X) :- p(X), not out(X). out(X) :- p(X), not in(X). :- in(1), in(2)."),
]


def cases() -> list[Case]:
    return [
        case(name, "stableModels", [], len(m := stable_models(src)), program=src, models=m) for name, src in PROGRAMS
    ]
