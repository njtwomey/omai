"""Global (Needleman-Wunsch) and local (Smith-Waterman) alignment scores against Biopython's PairwiseAligner, with
linear and affine (Gotoh) gaps."""

from importlib.metadata import version

import numpy as np
from Bio.Align import PairwiseAligner
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"biopython": version("biopython")}

PAIRS: list[tuple[str, str]] = [
    ("GATTACA", "GCATGCU"),
    ("TGTTACGG", "GGTTGACTA"),
    ("ACACACTA", "AGCACACA"),
    ("AAAA", "AA"),
    ("ACGT", "TGCA"),
    ("HEAGAWGHEE", "PAWHEAE"),
]

SCHEMES: list[tuple[str, float, float, float, float]] = [
    ("unit", 1, -1, -1, -1),
    ("dna", 3, -3, -2, -2),
    ("affine", 2, -1, -3, -0.5),
    ("affine2", 5, -4, -10, -1),
]


def random_pairs() -> list[tuple[str, str]]:
    g = rng(2102)
    out: list[tuple[str, str]] = []
    for _ in range(6):
        n, m = int(g.integers(1, 14)), int(g.integers(1, 14))
        out.append(("".join(g.choice(list("ACGT"), size=n)), "".join(g.choice(list("ACGT"), size=m))))
    return out


def cases() -> list[Case]:
    out: list[Case] = []
    for i, (a, b) in enumerate(PAIRS + random_pairs()):
        for name, match, mismatch, gap_open, gap_extend in SCHEMES:
            scores: list[float] = []
            for mode in ["global", "local"]:
                aligner = PairwiseAligner()
                aligner.mode = mode
                aligner.match_score = match
                aligner.mismatch_score = mismatch
                aligner.open_gap_score = gap_open
                aligner.extend_gap_score = gap_extend
                scores.append(float(aligner.score(a, b)))
            out.append(
                case(
                    f"{name}/{i}",
                    "align",
                    [a, b],
                    np.array(scores),
                    rtol=1e-12,
                    atol=1e-12,
                    match=match,
                    mismatch=mismatch,
                    gapOpen=gap_open,
                    gapExtend=gap_extend,
                )
            )
    return out
