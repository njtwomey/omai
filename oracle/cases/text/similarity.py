"""String similarity against rapidfuzz (Levenshtein with weights, OSA, Damerau-Levenshtein, Hamming, Jaro,
Jaro-Winkler, Indel, LCSseq, fuzz.ratio, token_sort_ratio, token_set_ratio) and Python sets (Jaccard, containment)."""

from importlib.metadata import version

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from rapidfuzz import fuzz
from rapidfuzz.distance import (
    OSA,
    DamerauLevenshtein,
    Hamming,
    Indel,
    Jaro,
    JaroWinkler,
    LCSseq,
    Levenshtein,
)

REFERENCE: dict[str, str] = {"rapidfuzz": version("rapidfuzz")}

PAIRS: list[tuple[str, str]] = [
    ("kitten", "sitting"),
    ("MARTHA", "MARHTA"),
    ("DIXON", "DICKSONX"),
    ("DWAYNE", "DUANE"),
    ("ca", "abc"),
    ("abcdef", "badcfe"),
    ("", ""),
    ("", "abc"),
    ("a", "a"),
    ("a", "b"),
    ("ab", "ba"),
    ("naïve café", "naive cafe"),
    ("😀ab", "a😀b"),
    ("fuzzy was a bear", "fuzzy fuzzy was a bear"),
    ("fuzzy was a bear but not a dog", "fuzzy was a bear but not a cat"),
    ("new york mets vs atlanta braves", "atlanta braves vs new york mets"),
    ("this is a test", "this is a test!"),
    ("the quick brown fox", "the  quick   brown fox jumps"),
]


def random_pairs() -> list[tuple[str, str]]:
    """Random strings over a small alphabet, so that transpositions and repeats are common."""
    g = rng(2101)
    out: list[tuple[str, str]] = []
    for _ in range(12):
        n, m = int(g.integers(0, 12)), int(g.integers(0, 12))
        a = "".join(g.choice(list("abcd"), size=n))
        b = "".join(g.choice(list("abcd"), size=m))
        out.append((a, b))
    return out


def cases() -> list[Case]:
    out: list[Case] = []
    for i, (a, b) in enumerate(PAIRS + random_pairs()):
        row = [
            Levenshtein.distance(a, b),
            Levenshtein.distance(a, b, weights=(1, 1, 2)),
            Levenshtein.distance(a, b, weights=(2, 3, 4)),
            Levenshtein.normalized_similarity(a, b),
            OSA.distance(a, b),
            DamerauLevenshtein.distance(a, b),
            Indel.distance(a, b),
            LCSseq.similarity(a, b),
            Jaro.similarity(a, b),
            JaroWinkler.similarity(a, b),
            JaroWinkler.similarity(a, b, prefix_weight=0.2),
            fuzz.ratio(a, b),
            fuzz.token_sort_ratio(a, b),
            fuzz.token_set_ratio(a, b),
            Hamming.distance(a, b) if len(a) == len(b) else -1,
        ]
        out.append(case(f"pair/{i}", "similarity", [a, b], np.array(row, dtype=np.float64), rtol=1e-12, atol=1e-12))
    sets = [
        (["a", "b", "c"], ["b", "c", "d"]),
        (["a"], ["a", "b", "c", "d"]),
        ([], []),
        (["x", "y"], []),
    ]
    for i, (sa, sb) in enumerate(sets):
        a, b = set(sa), set(sb)
        jac = len(a & b) / len(a | b) if a | b else 1.0
        cont = len(a & b) / len(a) if a else 1.0
        out.append(case(f"sets/{i}", "sets", [sa, sb], np.array([jac, cont])))
    return out
