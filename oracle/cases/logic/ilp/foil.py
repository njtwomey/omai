"""FOIL against a reference port (oracle/cases/logic/ilp/_ilp.py) on Quinlan's tasks: grandparent and daughter from
family relations, and the can-reach network of Quinlan (1990, figure 2) learned recursively. The learned clauses and
the gain of every chosen literal are compared."""

from cases.logic.ilp._ilp import foil
from omai_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"quinlan-1990": "Learning logical definitions from relations; reference port"}

PARENT = [("tom", "bob"), ("tom", "liz"), ("bob", "ann"), ("bob", "pat"), ("pat", "jim"), ("liz", "sue")]
FEMALE = [("liz",), ("ann",), ("pat",), ("sue",)]
LINKED = [
    ("0", "1"),
    ("0", "3"),
    ("1", "2"),
    ("3", "2"),
    ("3", "4"),
    ("4", "5"),
    ("4", "6"),
    ("6", "8"),
    ("7", "6"),
    ("7", "8"),
]


def closure(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Transitive closure, sorted."""
    reach = set(edges)
    while True:
        new = {(a, d) for a, b in reach for c, d in reach if b == c} - reach
        if not new:
            return sorted(reach)
        reach |= new


TASKS: list[tuple[str, str, int, list[tuple[str, ...]], dict[str, list[tuple[str, ...]]], bool]] = [
    (
        "grandparent",
        "grandparent",
        2,
        [("tom", "ann"), ("tom", "pat"), ("bob", "jim"), ("tom", "sue")],
        {"parent": list(PARENT)},
        False,
    ),
    (
        "daughter",
        "daughter",
        2,
        [("liz", "tom"), ("ann", "bob"), ("pat", "bob"), ("sue", "liz")],
        {"parent": list(PARENT), "female": list(FEMALE)},
        False,
    ),
    ("can-reach", "can_reach", 2, list(closure(LINKED)), {"linked_to": list(LINKED)}, True),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for name, target, arity, positive, background, recursion in TASKS:
        clauses, gains = foil(target, arity, positive, background, recursion=recursion)
        out.append(
            case(
                name,
                "foil",
                [],
                clauses,
                target=target,
                arity=arity,
                positive=[list(t) for t in positive],
                background={k: [list(t) for t in v] for k, v in background.items()},
                recursion=recursion,
                clauses=clauses,
                gains=gains,
                rtol=1e-9,
                atol=1e-9,
            )
        )
    return out
