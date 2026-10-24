"""Propositional logic against sympy.logic: models (satisfiable with all_models), validity, entailment
(the AIMA wumpus-world knowledge base among others), CNF, DNF and Tseitin conversion (checked against sympy's models),
and minimal sum and product forms (simplify_logic, whose term and literal counts the engine's exact minimisation must
not exceed)."""

from collections.abc import Iterable
from importlib.metadata import version
from typing import cast

import numpy as np
from sympy import Symbol
from sympy.logic.boolalg import (
    And,
    Boolean,
    Equivalent,
    Implies,
    Not,
    Or,
    Xor,
    simplify_logic,
)
from sympy.logic.inference import satisfiable
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"sympy": version("sympy")}

NAMES = ["p", "q", "r", "s"]

type Tree = str | tuple[str, list["Tree"]]


def random_tree(g: np.random.Generator, depth: int, names: list[str]) -> Tree:
    """A random formula tree."""
    if depth == 0 or g.random() < 0.25:
        return names[int(g.integers(0, len(names)))]
    op = ["not", "and", "or", "implies", "iff", "xor"][int(g.integers(0, 6))]
    if op == "not":
        return ("not", [random_tree(g, depth - 1, names)])
    k = int(g.integers(2, 4)) if op in ("and", "or") else 2
    return (op, [random_tree(g, depth - 1, names) for _ in range(k)])


def text(t: Tree) -> str:
    """The engine's ascii syntax, fully bracketed."""
    if isinstance(t, str):
        return t
    op, args = t
    if op == "not":
        return f"~{text(args[0])}"
    sym = {"and": " & ", "or": " | ", "implies": " -> ", "iff": " <-> ", "xor": " ^ "}[op]
    return "(" + sym.join(text(a) for a in args) + ")"


def sym(t: Tree) -> Boolean:
    """The sympy expression."""
    if isinstance(t, str):
        return Symbol(t)
    op, args = t
    xs = [sym(a) for a in args]
    if op == "not":
        return Not(xs[0])
    if op == "and":
        return And(*xs)
    if op == "or":
        return Or(*xs)
    if op == "implies":
        return Implies(xs[0], xs[1])
    if op == "iff":
        return Equivalent(xs[0], xs[1])
    return Xor(xs[0], xs[1])


def models_of(e: Boolean, names: list[str]) -> list[dict[str, bool]]:
    """Every model over `names` (variables absent from e range freely), sorted in truth-table order."""
    found = cast(Iterable[object], satisfiable(e, all_models=True))
    out: list[dict[str, bool]] = []
    for m in found:
        if m is False:
            return []
        if m is True:
            base: dict[str, bool] = {}
        else:
            assert isinstance(m, dict)
            base = {str(k): bool(v) for k, v in m.items() if str(k) in names}
        free = [n for n in names if n not in base]
        for bits in range(2 ** len(free)):
            full = dict(base)
            for i, n in enumerate(free):
                full[n] = bool((bits >> (len(free) - 1 - i)) & 1)
            out.append({n: full[n] for n in names})
    return sorted(out, key=lambda d: [d[n] for n in names])


def names_in(t: Tree) -> list[str]:
    """The variables of a tree, sorted (sympy may simplify some away on construction)."""
    if isinstance(t, str):
        return [t]
    return sorted({n for a in t[1] for n in names_in(a)})


def counts(e: Boolean) -> tuple[int, int]:
    """(terms, literals) of a sum or product form."""
    terms = list(e.args) if isinstance(e, And | Or) else [e]
    lits = 0
    for t in terms:
        lits += len(t.args) if isinstance(t, And | Or) else 1
    return len(terms), lits


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(0)
    for i in range(40):
        names = NAMES[: 2 + (i % 3)]
        t = random_tree(g, 3 + (i % 2), names)
        e = sym(t)
        variables = names_in(t)
        ms = models_of(e, variables)
        valid = not satisfiable(Not(e))
        if e.free_symbols:
            sop = cast(Boolean, simplify_logic(e, form="dnf", force=True))
            pos = cast(Boolean, simplify_logic(e, form="cnf", force=True))
            sop_counts: list[int] = list(counts(sop)) if sop not in (True, False) else [0, 0]
            pos_counts: list[int] = list(counts(pos)) if pos not in (True, False) else [0, 0]
        else:
            sop_counts, pos_counts = [0, 0], [0, 0]
        out.append(
            case(
                f"random/{i}",
                "formula",
                [],
                len(ms),
                formula=text(t),
                variables=variables,
                models=ms,
                satisfiable=bool(ms),
                valid=bool(valid),
                minimalDnf=sop_counts,
                minimalCnf=pos_counts,
            )
        )
    # AIMA's wumpus world (section 7.4.3): R1..R5 entail ~P12 but not P22.
    p11, p12, p21, p22, p31, b11, b21 = (Symbol(n) for n in ["P11", "P12", "P21", "P22", "P31", "B11", "B21"])
    kb = And(Not(p11), Equivalent(b11, Or(p12, p21)), Equivalent(b21, Or(p11, p22, p31)), Not(b11), b21)
    kb_text = "~P11 & (B11 <-> (P12 | P21)) & (B21 <-> (P11 | P22 | P31)) & ~B11 & B21"
    for query, qtext in [(Not(p12), "~P12"), (p22, "P22"), (Or(p22, p31), "P22 | P31"), (Not(p21), "~P21")]:
        entailed = not satisfiable(And(kb, Not(query)))
        out.append(
            case(
                f"wumpus/{qtext}",
                "entails",
                [],
                bool(entailed),
                premises=[kb_text],
                conclusion=qtext,
                entailed=bool(entailed),
            )
        )
    for premises, conclusion in [
        (["p -> q", "q -> r"], "p -> r"),
        (["p | q", "~p | r"], "q | r"),
        (["p -> q", "q"], "p"),
        (["(p & q) -> r", "p", "~r"], "~q"),
    ]:
        from sympy.parsing.sympy_parser import parse_expr

        def conv(s: str) -> Boolean:
            expr = s.replace("->", ">>")
            return parse_expr(expr, evaluate=False)

        entailed = not satisfiable(And(*[conv(x) for x in premises], Not(conv(conclusion))))
        out.append(
            case(
                f"entails/{' ; '.join(premises)} |= {conclusion}",
                "entails",
                [],
                bool(entailed),
                premises=premises,
                conclusion=conclusion,
                entailed=bool(entailed),
            )
        )
    return out
