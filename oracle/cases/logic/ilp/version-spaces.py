"""Version spaces against brute force (oracle/cases/logic/ilp/_ilp.py): the boundaries S and G as the maximally
specific and general consistent conjunctions found by enumerating the whole hypothesis space, the version space's size,
and Find-S, on Mitchell's EnjoySport (table 2.1) and seeded random tasks."""

from cases.logic.ilp._ilp import brute_version_space, find_s
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"mitchell-1997": "Machine Learning, chapter 2; brute-force reference"}

ENJOY_DOMAINS = [
    ["sunny", "cloudy", "rainy"],
    ["warm", "cold"],
    ["normal", "high"],
    ["strong", "weak"],
    ["warm", "cool"],
    ["same", "change"],
]
ENJOY = [
    (["sunny", "warm", "normal", "strong", "warm", "same"], True),
    (["sunny", "warm", "high", "strong", "warm", "same"], True),
    (["rainy", "cold", "high", "strong", "warm", "change"], False),
    (["sunny", "warm", "high", "strong", "cool", "change"], True),
]


def cases() -> list[Case]:
    out: list[Case] = []
    tasks: list[tuple[str, list[tuple[list[str], bool]], list[list[str]]]] = [
        ("enjoysport", ENJOY, ENJOY_DOMAINS),
        ("enjoysport/first-two", ENJOY[:2], ENJOY_DOMAINS),
        ("enjoysport/negative-first", [ENJOY[2], ENJOY[0], ENJOY[1]], ENJOY_DOMAINS),
    ]
    for seed in range(6):
        g = rng(seed)
        domains = [[f"v{j}" for j in range(int(g.integers(2, 4)))] for _ in range(4)]
        target = [d[int(g.integers(0, len(d)))] if g.random() < 0.6 else "?" for d in domains]
        examples: list[tuple[list[str], bool]] = []
        for k in range(int(g.integers(3, 7))):
            x = [d[int(g.integers(0, len(d)))] for d in domains]
            if k % 2 == 0:
                # Half the examples are drawn from the target concept, so most tasks have positives.
                x = [v if t == "?" else t for v, t in zip(x, target, strict=True)]
            examples.append((x, all(t in ("?", v) for t, v in zip(target, x, strict=True))))
        tasks.append((f"random/{seed}", examples, domains))
    for name, examples, domains in tasks:
        s, gb, size = brute_version_space(examples, domains)
        out.append(
            case(
                name,
                "versionSpace",
                [],
                size,
                examples=[{"values": x, "positive": y} for x, y in examples],
                domains=domains,
                specific=s,
                general=gb,
                size=size,
                findS=find_s(examples),
            )
        )
    return out
