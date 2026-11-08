"""Meta-interpretive learning and learning from failures on the textbook tasks of Metagol and Popper: grandparent
from parent (the chain metarule), grandparent from mother and father with an invented predicate (Metagol's family
example), and the recursive ancestor relation. The expected programs are the systems' published answers; their
coverage (every positive, no negative) is confirmed with the reference Prolog interpreter."""

from cases.logic._prolog import Machine
from omai_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {
    "metagol": "github.com/metagol/metagol examples; checked with oracle/cases/logic/_prolog.py",
    "popper": "github.com/logic-and-learning-lab/Popper examples; checked with oracle/cases/logic/_prolog.py",
}

PARENT = "parent(ann, amy). parent(ann, andy). parent(amy, amelia). parent(andy, spongebob). parent(amelia, ben)."
KIN = (
    "mother(ann, amy). mother(ann, andy). mother(amy, amelia). mother(linda, gavin). "
    "father(steve, amy). father(steve, andy). father(gavin, amelia). father(andy, spongebob)."
)

TASKS: list[tuple[str, str, str, list[str], list[str], list[str], list[str], list[str]]] = [
    (
        "grandparent",
        "both",
        PARENT,
        ["parent/2"],
        ["grandparent(ann, amelia)", "grandparent(ann, spongebob)", "grandparent(amy, ben)"],
        ["grandparent(amy, amelia)", "grandparent(ann, amy)", "grandparent(ann, ben)"],
        ["ident", "chain"],
        ["grandparent(A,B) :- parent(A,C), parent(C,B)."],
    ),
    (
        "kinship-invention",
        "metagol",
        KIN,
        ["mother/2", "father/2"],
        [
            "grandparent(ann, amelia)",
            "grandparent(steve, amelia)",
            "grandparent(ann, spongebob)",
            "grandparent(steve, spongebob)",
            "grandparent(linda, amelia)",
        ],
        ["grandparent(amy, amelia)"],
        ["ident", "chain"],
        [
            "grandparent(A,B) :- grandparent_1(A,C), grandparent_1(C,B).",
            "grandparent_1(A,B) :- father(A,B).",
            "grandparent_1(A,B) :- mother(A,B).",
        ],
    ),
    (
        "ancestor",
        "both",
        PARENT,
        ["parent/2"],
        ["ancestor(ann, amy)", "ancestor(ann, amelia)", "ancestor(ann, ben)", "ancestor(andy, spongebob)"],
        ["ancestor(amy, ann)", "ancestor(amelia, andy)", "ancestor(spongebob, ann)", "ancestor(amy, andy)"],
        ["ident", "tailrec"],
        ["ancestor(A,B) :- parent(A,B).", "ancestor(A,B) :- parent(A,C), ancestor(C,B)."],
    ),
]


def check(background: str, program: list[str], positive: list[str], negative: list[str]) -> bool:
    """Every positive provable and no negative."""
    text = background + "\n" + "\n".join(program)
    return all(Machine(text).run(e, 1)["answers"] for e in positive) and not any(
        Machine(text).run(e, 1)["answers"] for e in negative
    )


def cases() -> list[Case]:
    out: list[Case] = []
    for name, systems, bk, preds, pos, neg, metarules, program in TASKS:
        if not check(bk, program, pos, neg):
            raise RuntimeError(f"{name}: the expected program does not fit the data")
        out.append(
            case(
                name,
                "mil",
                [],
                program,
                systems=systems,
                background=bk,
                predicates=preds,
                positive=pos,
                negative=neg,
                metarules=metarules,
                program=program,
            )
        )
    return out
