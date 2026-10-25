"""First-order resolution against textbook ground truth: Pelletier's problems without equality (1-33, 35, 36, 39-42,
44-46; all theorems), AIMA's crime and curiosity examples, the drinker and barber puzzles, and countersatisfiable
conjectures whose clause sets saturate (hand-checked: each has a small countermodel).

No packaged prover serves as a reference: NLTK's ResolutionProver lacks factoring (it fails the barber puzzle) and
reads single-letter names as variables (it "proves" rel(a, c) from rel(a, b) and rel(b, c)), and SWI-Prolog, Prover9
and E cannot be installed with uv."""

from mdml_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {
    "pelletier-1986": "F. J. Pelletier, Seventy-five problems for testing automatic theorem provers, JAR 2 (1986)",
    "aima": "Russell and Norvig, AIMA 4th ed., chapter 9",
}

THEOREMS: list[tuple[str, list[str], str]] = [
    ("pelletier/1", [], "(p -> q) <-> (-q -> -p)"),
    ("pelletier/2", [], "--p <-> p"),
    ("pelletier/3", [], "-(p -> q) -> (q -> p)"),
    ("pelletier/4", [], "(-p -> q) <-> (-q -> p)"),
    ("pelletier/5", [], "((p | q) -> (p | r)) -> (p | (q -> r))"),
    ("pelletier/6", [], "p | -p"),
    ("pelletier/7", [], "p | ---p"),
    ("pelletier/8", [], "((p -> q) -> p) -> p"),
    ("pelletier/9", [], "((p | q) & (-p | q) & (p | -q)) -> -(-p | -q)"),
    ("pelletier/10", ["q -> r", "r -> (p & q)", "p -> (q | r)"], "p <-> q"),
    ("pelletier/11", [], "p <-> p"),
    ("pelletier/12", [], "((p <-> q) <-> r) <-> (p <-> (q <-> r))"),
    ("pelletier/13", [], "(p | (q & r)) <-> ((p | q) & (p | r))"),
    ("pelletier/14", [], "(p <-> q) <-> ((q | -p) & (-q | p))"),
    ("pelletier/15", [], "(p -> q) <-> (-p | q)"),
    ("pelletier/16", [], "(p -> q) | (q -> p)"),
    ("pelletier/17", [], "((p & (q -> r)) -> s) <-> ((-p | q | s) & (-p | -r | s))"),
    ("pelletier/18", [], "exists y. all x.(F(y) -> F(x))"),
    ("pelletier/19", [], "exists x. all y. all z.((P(y) -> Q(z)) -> (P(x) -> Q(x)))"),
    (
        "pelletier/20",
        [],
        "(all x. all y. exists z. all w.((P(x) & Q(y)) -> (R(z) & S(w)))) -> "
        "((exists x. exists y.(P(x) & Q(y))) -> exists z. R(z))",
    ),
    ("pelletier/21", ["exists x.(p -> F(x))", "exists x.(F(x) -> p)"], "exists x.(p <-> F(x))"),
    ("pelletier/22", [], "(all x.(p <-> F(x))) -> (p <-> all x. F(x))"),
    ("pelletier/23", [], "(all x.(p | F(x))) <-> (p | all x. F(x))"),
    (
        "pelletier/24",
        [
            "-(exists x.(S(x) & Q(x)))",
            "all x.(P(x) -> (Q(x) | R(x)))",
            "(-(exists x. P(x))) -> exists x. Q(x)",
            "all x.((Q(x) | R(x)) -> S(x))",
        ],
        "exists x.(P(x) & R(x))",
    ),
    (
        "pelletier/25",
        [
            "exists x. P(x)",
            "all x.(F(x) -> (-G(x) & R(x)))",
            "all x.(P(x) -> (G(x) & F(x)))",
            "(all x.(P(x) -> Q(x))) | exists x.(P(x) & R(x))",
        ],
        "exists x.(Q(x) & P(x))",
    ),
    (
        "pelletier/26",
        ["(exists x. P(x)) <-> (exists x. Q(x))", "all x. all y.((P(x) & Q(y)) -> (R(x) <-> S(y)))"],
        "(all x.(P(x) -> R(x))) <-> (all x.(Q(x) -> S(x)))",
    ),
    (
        "pelletier/27",
        [
            "exists x.(F(x) & -G(x))",
            "all x.(F(x) -> H(x))",
            "all x.((J(x) & I(x)) -> F(x))",
            "(exists x.(H(x) & -G(x))) -> all x.(I(x) -> -H(x))",
        ],
        "all x.(J(x) -> -I(x))",
    ),
    (
        "pelletier/28",
        [
            "all x.(P(x) -> all x. Q(x))",
            "(all x.(Q(x) | R(x))) -> exists x.(Q(x) & S(x))",
            "(exists x. S(x)) -> all x.(F(x) -> G(x))",
        ],
        "all x.((P(x) & F(x)) -> G(x))",
    ),
    (
        "pelletier/29",
        ["(exists x. F(x)) & (exists x. G(x))"],
        "((all x.(F(x) -> H(x))) & (all x.(G(x) -> J(x)))) <-> all x. all y.((F(x) & G(y)) -> (H(x) & J(y)))",
    ),
    ("pelletier/30", ["all x.((F(x) | G(x)) -> -H(x))", "all x.((G(x) -> -I(x)) -> (F(x) & H(x)))"], "all x. I(x)"),
    (
        "pelletier/31",
        ["-(exists x.(F(x) & (G(x) | H(x))))", "exists x.(I(x) & F(x))", "all x.(-H(x) -> J(x))"],
        "exists x.(I(x) & J(x))",
    ),
    (
        "pelletier/32",
        ["all x.((F(x) & (G(x) | H(x))) -> I(x))", "all x.((I(x) & H(x)) -> J(x))", "all x.(K(x) -> H(x))"],
        "all x.((F(x) & K(x)) -> J(x))",
    ),
    (
        "pelletier/33",
        [],
        "(all x.((P(a) & (P(x) -> P(b))) -> P(c))) <-> all x.((-P(a) | P(x) | P(c)) & (-P(a) | -P(b) | P(c)))",
    ),
    ("pelletier/35", [], "exists x. exists y.(P(x, y) -> all x. all y. P(x, y))"),
    (
        "pelletier/36",
        [
            "all x. exists y. F(x, y)",
            "all x. exists y. G(x, y)",
            "all x. all y.((F(x, y) | G(x, y)) -> all z.((F(y, z) | G(y, z)) -> H(x, z)))",
        ],
        "all x. exists y. H(x, y)",
    ),
    ("pelletier/39", [], "-(exists x. all y.(F(y, x) <-> -F(y, y)))"),
    (
        "pelletier/40",
        [],
        "(exists y. all x.(F(x, y) <-> F(x, x))) -> -(all x. exists y. all z.(F(z, y) <-> -F(z, x)))",
    ),
    ("pelletier/42", [], "-(exists y. all x.(F(x, y) <-> -(exists z.(F(x, z) & F(z, x)))))"),
    (
        "pelletier/44",
        [
            "all x.(F(x) -> ((exists y.(G(y) & H(x, y))) & (exists y.(G(y) & -H(x, y)))))",
            "exists x.(J(x) & all y.(G(y) -> H(x, y)))",
        ],
        "exists x.(J(x) & -F(x))",
    ),
    ("pelletier/41", ["all z. exists y. all x.(F(x, y) <-> (F(x, z) & -F(x, x)))"], "-(exists z. all x. F(x, z))"),
    (
        "pelletier/45",
        [
            "all x.((F(x) & all y.((G(y) & H(x, y)) -> J(x, y))) -> all y.((G(y) & H(x, y)) -> K(y)))",
            "-(exists y.(L(y) & K(y)))",
            "exists x.(F(x) & all y.(H(x, y) -> L(y)) & all y.((G(y) & H(x, y)) -> J(x, y)))",
        ],
        "exists x.(F(x) & -(exists y.(G(y) & H(x, y))))",
    ),
    (
        "pelletier/46",
        [
            "all x.((F(x) & all y.((F(y) & H(y, x)) -> G(y))) -> G(x))",
            "(exists x.(F(x) & -G(x))) -> exists x.(F(x) & -G(x) & all y.((F(y) & -G(y)) -> J(x, y)))",
            "all x. all y.((F(x) & F(y) & H(x, y)) -> -J(y, x))",
        ],
        "all x.(F(x) -> G(x))",
    ),
    ("drinker", [], "exists x.(drinks(x) -> all y. drinks(y))"),
    ("barber", [], "-(exists x. all y.(shaves(x, y) <-> -shaves(y, y)))"),
    ("socrates", ["all x.(man(x) -> mortal(x))", "man(socrates)"], "mortal(socrates)"),
    (
        "aima/crime",
        [
            "all x. all y. all z.((american(x) & weapon(y) & sells(x, y, z) & hostile(z)) -> criminal(x))",
            "exists x.(owns(nono, x) & missile(x))",
            "all x.((missile(x) & owns(nono, x)) -> sells(west, x, nono))",
            "all x.(missile(x) -> weapon(x))",
            "all x.(enemy(x, america) -> hostile(x))",
            "american(west)",
            "enemy(nono, america)",
        ],
        "criminal(west)",
    ),
    (
        "aima/curiosity",
        [
            "all x.((all y.(animal(y) -> loves(x, y))) -> exists y. loves(y, x))",
            "all x.((exists z.(animal(z) & kills(x, z))) -> all y. -loves(y, x))",
            "all x.(animal(x) -> loves(jack, x))",
            "kills(jack, tuna) | kills(curiosity, tuna)",
            "cat(tuna)",
            "all x.(cat(x) -> animal(x))",
        ],
        "kills(curiosity, tuna)",
    ),
]

COUNTERSATISFIABLE: list[tuple[str, list[str], str]] = [
    ("converse", ["all x.(prop(x) -> qual(x))", "qual(a)"], "prop(a)"),
    ("some-all", ["exists x. prop(x)"], "all x. prop(x)"),
    ("quantifier-swap", ["all x. exists y. rel(x, y)"], "exists y. all x. rel(x, y)"),
    ("split", ["all x.(prop(x) | qual(x))"], "all x. prop(x)"),
    ("affirm-consequent", ["pp -> qq", "qq"], "pp"),
    ("transitivity-missing", ["rel(a, b)", "rel(b, c)"], "rel(a, c)"),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for name, axioms, conjecture in THEOREMS:
        out.append(case(name, "prove", [], "theorem", axioms=axioms, conjecture=conjecture, status="theorem"))
    for name, axioms, conjecture in COUNTERSATISFIABLE:
        out.append(
            case(
                f"countersatisfiable/{name}",
                "prove",
                [],
                "countersatisfiable",
                axioms=axioms,
                conjecture=conjecture,
                status="countersatisfiable",
            )
        )
    return out
