"""Inverse entailment on textbook tasks: a Michalski-style trains problem (ten trains; eastbound exactly when a train
has a short closed car, the classic answer) and Lavrač and Džeroski's daughter task. The expected theories are the
textbook ones; their coverage (every positive, no negative) is confirmed with the reference Prolog interpreter, and the
bottom clause of the first example is computed by a reference port of Muggleton's saturation over the same modes."""

from itertools import product

from cases.logic._prolog import Machine
from mdml_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {
    "muggleton-1995": "Inverse entailment and Progol; textbook theories checked with oracle/cases/logic/_prolog.py",
}

# Cars: (train, car, length, roof, shape, wheels, load).
CARS = [
    ("east1", "c11", "short", "closed", "rectangle", 2, "circle"),
    ("east1", "c12", "long", "open", "rectangle", 3, "hexagon"),
    ("east1", "c13", "short", "open", "u_shaped", 2, "triangle"),
    ("east2", "c21", "long", "closed", "rectangle", 2, "rectangle"),
    ("east2", "c22", "short", "closed", "bucket", 2, "triangle"),
    ("east3", "c31", "short", "closed", "hexagon", 2, "circle"),
    ("east3", "c32", "long", "open", "rectangle", 3, "rectangle"),
    ("east4", "c41", "short", "open", "u_shaped", 2, "circle"),
    ("east4", "c42", "short", "closed", "rectangle", 3, "rectangle"),
    ("east4", "c43", "long", "closed", "rectangle", 2, "hexagon"),
    ("east5", "c51", "long", "open", "bucket", 3, "triangle"),
    ("east5", "c52", "short", "closed", "ellipse", 2, "diamond"),
    ("west6", "c61", "long", "closed", "rectangle", 2, "circle"),
    ("west6", "c62", "short", "open", "rectangle", 2, "triangle"),
    ("west7", "c71", "short", "open", "u_shaped", 2, "circle"),
    ("west7", "c72", "long", "closed", "hexagon", 3, "diamond"),
    ("west8", "c81", "long", "open", "rectangle", 2, "hexagon"),
    ("west8", "c82", "short", "open", "bucket", 3, "rectangle"),
    ("west9", "c91", "long", "closed", "ellipse", 3, "triangle"),
    ("west9", "c92", "short", "open", "rectangle", 2, "diamond"),
    ("west9", "c93", "long", "closed", "bucket", 2, "circle"),
    ("west10", "c101", "short", "open", "hexagon", 3, "rectangle"),
    ("west10", "c102", "long", "closed", "u_shaped", 2, "triangle"),
]


def trains_background() -> str:
    """The trains as Prolog facts."""
    out: list[str] = []
    for train, car, length, roof, shape, wheels, load in CARS:
        out.append(f"has_car({train}, {car}). {length}({car}). {roof}({car}). shape({car}, {shape}). ")
        out.append(f"wheels({car}, {wheels}). load({car}, {load}).")
    return "\n".join(out)


TRAIN_MODES = [
    "modeh(1, eastbound(+train))",
    "modeb(*, has_car(+train, -car))",
    "modeb(1, short(+car))",
    "modeb(1, long(+car))",
    "modeb(1, closed(+car))",
    "modeb(1, open(+car))",
    "modeb(1, shape(+car, #shape))",
    "modeb(1, wheels(+car, #int))",
    "modeb(1, load(+car, #shape))",
]

DAUGHTER_BK = (
    "parent(ann, mary). parent(ann, tom). parent(tom, eve). parent(tom, ian). female(ann). female(mary). female(eve)."
)
DAUGHTER_MODES = [
    "modeh(1, daughter(+person, +person))",
    "modeb(*, parent(+person, -person))",
    "modeb(*, parent(-person, +person))",
    "modeb(1, female(+person))",
]


def parse_mode(text: str) -> tuple[str, str, list[tuple[str, str]], float]:
    """(kind, predicate, [(io, type)], recall) of a mode written as above."""
    kind, rest = text.split("(", 1)
    recall_text, atom = rest.split(",", 1)
    atom = atom.strip()[:-1]
    name, args = atom.split("(", 1)
    parts = [a.strip() for a in args[:-1].split(",")]
    recall = float("inf") if recall_text.strip() == "*" else float(recall_text)
    return kind, name, [(p[0], p[1:]) for p in parts], recall


def letter(i: int) -> str:
    """A, B, C, ..."""
    return chr(65 + i)


def bottom_clause(background: str, example: str, modes: list[str], depth: int = 2) -> str:
    """Muggleton's saturation, ported: the bottom clause of a ground example as text."""
    m = [parse_mode(x) for x in modes]
    head_mode = m[0]
    body_modes = m[1:]
    ex_name, ex_args = example.split("(", 1)
    ex_terms = [a.strip() for a in ex_args[:-1].split(",")]
    names: dict[str, str] = {}

    def var(t: str) -> str:
        if t not in names:
            names[t] = letter(len(names))
        return names[t]

    known: list[tuple[str, str]] = []

    def know(t: str, ty: str) -> None:
        if (t, ty) not in known:
            known.append((t, ty))

    head_args: list[str] = []
    for (io, ty), t in zip(head_mode[2], ex_terms, strict=True):
        if io == "#":
            head_args.append(t)
            continue
        head_args.append(var(t))
        if io == "+":
            know(t, ty)
    body: list[str] = []
    for _ in range(depth):
        snapshot = list(known)
        for _kind, name, args, recall in body_modes:
            in_pos = [i for i, (io, _) in enumerate(args) if io == "+"]
            pools = [[t for t, ty in snapshot if ty == args[i][1]] for i in in_pos]
            for combo in product(*pools):
                query_args = [combo[in_pos.index(i)] if i in in_pos else f"Q{i}" for i in range(len(args))]
                query = f"{name}({', '.join(query_args)})"
                result = Machine(background).run(query, max_solutions=int(min(recall, 1000)))
                answers = result["answers"]
                assert isinstance(answers, list)
                for ans in answers:
                    assert isinstance(ans, list)
                    binding = {str(k): str(v) for k, v in ans}
                    ground = [binding.get(f"Q{i}", query_args[i]) for i in range(len(args))]
                    lit_args = [g if args[i][0] == "#" else var(g) for i, g in enumerate(ground)]
                    lit = f"{name}({','.join(lit_args)})"
                    if lit in body:
                        continue
                    body.append(lit)
                    for i, g in enumerate(ground):
                        if args[i][0] == "-":
                            know(g, args[i][1])
    return f"{ex_name}({','.join(head_args)}) :- {', '.join(body)}."


def covers_exactly(background: str, theory: str, positive: list[str], negative: list[str]) -> bool:
    """Whether the theory proves every positive and no negative (reference interpreter)."""
    program = background + "\n" + theory
    for e in positive:
        if not Machine(program).run(e, 1)["answers"]:
            return False
    return all(not Machine(program).run(e, 1)["answers"] for e in negative)


TASKS: list[tuple[str, str, list[str], list[str], list[str], str]] = [
    (
        "trains",
        trains_background(),
        [f"eastbound(east{i})" for i in range(1, 6)],
        [f"eastbound(west{i})" for i in range(6, 11)],
        TRAIN_MODES,
        "eastbound(A) :- has_car(A,B), short(B), closed(B).",
    ),
    (
        "daughter",
        DAUGHTER_BK,
        ["daughter(mary, ann)", "daughter(eve, tom)"],
        ["daughter(tom, ann)", "daughter(eve, ann)", "daughter(ian, tom)"],
        DAUGHTER_MODES,
        "daughter(A,B) :- parent(B,A), female(A).",
    ),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for name, bk, pos, neg, modes, theory in TASKS:
        if not covers_exactly(bk, theory, pos, neg):
            raise RuntimeError(f"{name}: the textbook theory does not fit the data")
        bottom = bottom_clause(bk, pos[0].replace(" ", ""), modes)
        out.append(
            case(
                name,
                "progol",
                [],
                [theory],
                background=bk,
                positive=pos,
                negative=neg,
                modes=modes,
                theory=[theory],
                bottom=bottom,
            )
        )
    return out
