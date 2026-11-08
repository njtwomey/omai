"""Datalog against clingo: least Herbrand models and perfect models of stratified programs (transitive closure of
seeded random graphs, same generation, family relations, a points-to analysis, stratified negation, comparisons),
and per-stratum round counts of the immediate consequence operator from a reference naive evaluator
(oracle/cases/logic/_datalog.py). Unstratifiable programs are recorded as errors."""

from importlib.metadata import version

import clingo
from cases.logic._datalog import evaluate
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"clingo": version("clingo")}


def asp(src: str) -> str:
    """The program in clingo's syntax."""
    return src.replace("\\=", "!=").replace("=<", "<=").replace("\\+", "not")


def clingo_models(src: str) -> list[list[str]]:
    """Every stable model, each sorted (a stratified program has exactly one)."""
    ctl = clingo.Control(["0", "--warn=none"])
    ctl.add("base", [], asp(src))
    ctl.ground([("base", [])])
    out: list[list[str]] = []

    def on_model(m: clingo.Model) -> None:
        out.append(sorted(str(s) for s in m.symbols(atoms=True)))

    ctl.solve(on_model=on_model)
    return sorted(out)


def graph(seed: int, n: int, m: int) -> str:
    """Edge facts of a seeded random directed graph."""
    g = rng(seed)
    edges = {(int(a), int(b)) for a, b in zip(g.integers(0, n, m), g.integers(0, n, m), strict=True) if a != b}
    return " ".join(f"edge({a},{b})." for a, b in sorted(edges)) + " "


TC = "path(X, Y) :- edge(X, Y). path(X, Z) :- path(X, Y), edge(Y, Z)."
TC_RIGHT = "path(X, Y) :- edge(X, Y). path(X, Z) :- edge(X, Y), path(Y, Z)."
TC_DOUBLE = "path(X, Y) :- edge(X, Y). path(X, Z) :- path(X, Y), path(Y, Z)."

PROGRAMS: list[tuple[str, str]] = [
    ("tc-left/0", graph(0, 8, 12) + TC),
    ("tc-right/1", graph(1, 10, 14) + TC_RIGHT),
    ("tc-double/2", graph(2, 12, 18) + TC_DOUBLE),
    (
        "same-generation",
        "par(b, a). par(c, a). par(d, b). par(e, b). par(f, c). par(g, f). "
        "person(X) :- par(X, _). person(X) :- par(_, X). "
        "sg(X, X) :- person(X). sg(X, Y) :- par(X, XP), sg(XP, YP), par(Y, YP).",
    ),
    (
        "family",
        "parent(tom, bob). parent(tom, liz). parent(bob, ann). parent(bob, pat). parent(pat, jim). "
        "male(tom). male(bob). male(jim). "
        "anc(X, Y) :- parent(X, Y). anc(X, Z) :- parent(X, Y), anc(Y, Z). "
        "grandfather(X, Z) :- male(X), parent(X, Y), parent(Y, Z). "
        "cousin_free(X) :- parent(_, X), not has_sibling(X). has_sibling(X) :- parent(P, X), parent(P, Y), X \\= Y.",
    ),
    (
        "points-to",
        "new(a, h1). new(b, h2). assign(c, a). assign(d, c). assign(c, b). store(d, f, a). load(e, c, f). "
        "pt(V, H) :- new(V, H). pt(V, H) :- assign(V, W), pt(W, H). "
        "hpt(H1, F, H2) :- store(V, F, W), pt(V, H1), pt(W, H2). "
        "pt(V, H2) :- load(V, W, F), pt(W, H1), hpt(H1, F, H2).",
    ),
    (
        "negation",
        graph(3, 9, 10) + "node(X) :- edge(X, _). node(Y) :- edge(_, Y). reach(0). reach(Y) :- reach(X), edge(X, Y). "
        "unreachable(X) :- node(X), not reach(X). sink(X) :- node(X), not has_out(X). has_out(X) :- edge(X, _). "
        "lonely(X) :- unreachable(X), sink(X).",
    ),
    (
        "comparisons",
        "num(1). num(2). num(3). num(4). less(X, Y) :- num(X), num(Y), X < Y. "
        "between(X, Y, Z) :- less(X, Y), less(Y, Z). notsucc(X, Y) :- less(X, Y), Y > X, not succ(X, Y). "
        "succ(1, 2). succ(2, 3). succ(3, 4). leq(X, Y) :- num(X), num(Y), X =< Y, X \\= 4.",
    ),
    ("strata-chain", "a(1). a(2). a(3). b(X) :- a(X), not c(X). c(2). d(X) :- a(X), not b(X). e(X) :- d(X), not c(X)."),
]

UNSTRATIFIABLE = [
    ("win-move", "move(a, b). move(b, a). move(b, c). win(X) :- move(X, Y), not win(Y)."),
    ("odd-loop", "p :- not q. q :- p."),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for name, src in PROGRAMS:
        models = clingo_models(src)
        model, rounds = evaluate(src)
        if len(models) != 1 or models[0] != model:
            raise RuntimeError(f"{name}: reference evaluator disagrees with clingo")
        out.append(case(name, "datalog", [], len(model), program=src, model=model, rounds=rounds))
    for name, src in UNSTRATIFIABLE:
        out.append(case(name, "unstratifiable", [], None, program=src, stableModels=clingo_models(src)))
    return out
