"""Generalisation against a reference port of Plotkin's anti-unification (oracle/cases/logic/ilp/_ilp.py): lggs of term
pairs, and relative least general generalisations (head-connected, reduced) on Lavrač and Džeroski's daughter task and
a grandparent task. Clauses are compared up to theta-equivalence."""

from cases.logic.ilp._ilp import Lgg, parse, rlgg, show
from omai_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"plotkin-1970": "reference port of anti-unification (oracle/cases/logic/ilp/_ilp.py)"}

PAIRS = [
    ("p(f(a),a,g(b))", "p(f(b),b,g(c))"),
    ("f(a,b,a)", "f(c,d,c)"),
    ("g(h(a,b),h(a,b))", "g(h(c,d),h(c,d))"),
    ("p(a)", "p(a)"),
    ("p(f(X),Y)", "p(g(X),Y)"),
    ("q(s(s(z)),s(z))", "q(s(z),z)"),
]

DAUGHTER = [
    "parent(ann,mary)",
    "parent(ann,tom)",
    "parent(tom,eve)",
    "parent(tom,ian)",
    "female(ann)",
    "female(mary)",
    "female(eve)",
]
FAMILY = [
    "parent(tom,bob)",
    "parent(tom,liz)",
    "parent(bob,ann)",
    "parent(bob,pat)",
    "parent(pat,jim)",
    "male(tom)",
    "male(bob)",
    "male(jim)",
]


def cases() -> list[Case]:
    out: list[Case] = []
    for a, b in PAIRS:
        g = show(Lgg().of(parse(a), parse(b)))
        out.append(case(f"lgg/{a}/{b}", "lgg", [], g, a=a, b=b, lgg=g))
    for name, examples, bk in [
        ("daughter", ["daughter(mary,ann)", "daughter(eve,tom)"], DAUGHTER),
        ("grandparent", ["grandparent(tom,ann)", "grandparent(bob,jim)"], FAMILY),
        ("grandfather", ["grandfather(tom,ann)", "grandfather(tom,pat)"], FAMILY),
    ]:
        clause = rlgg(examples, bk)
        out.append(case(f"rlgg/{name}", "rlgg", [], clause, examples=examples, background=bk, clause=clause))
    return out
