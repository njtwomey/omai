"""Identification against pgmpy's CausalInference: back-door adjustment sets (all minimal ones), the minimal
adjustment set, front-door sets, and their validity checks, on the student and Asia graphs and classic textbook
graphs (the M-bias graph, a front-door graph with a latent confounder, and a graph with two back-door routes)."""

import warnings
from collections.abc import Hashable
from typing import cast

import numpy as np
import pgmpy
from pgmpy.base import DAG
from pgmpy.inference import CausalInference
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.pgm import ASIA, STUDENT

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__}

GRAPHS: dict[str, tuple[list[tuple[str, str]], list[str], list[tuple[str, str]]]] = {
    # name: (edges, latent variables, (treatment, outcome) pairs)
    "student": (
        [(p, n) for n, _, ps, _ in STUDENT for p in ps],
        [],
        [("grade", "letter"), ("intelligence", "letter"), ("difficulty", "sat")],
    ),
    "asia": (
        [(p, n) for n, _, ps, _ in ASIA for p in ps],
        [],
        [("lung", "dysp"), ("either", "dysp"), ("smoke", "xray")],
    ),
    "m-bias": ([("a", "x"), ("a", "m"), ("b", "m"), ("b", "y"), ("x", "y")], [], [("x", "y")]),
    "two-routes": ([("z1", "x"), ("z1", "z2"), ("z2", "y"), ("z3", "x"), ("z3", "y"), ("x", "y")], [], [("x", "y")]),
    "frontdoor": ([("u", "x"), ("u", "y"), ("x", "m"), ("m", "y")], ["u"], [("x", "y")]),
}


def sets(fs: object) -> list[list[str]]:
    """pgmpy's frozenset of frozensets (or a single empty frozenset) as sorted lists."""
    if (
        isinstance(fs, frozenset)
        and all(isinstance(e, frozenset) for e in cast(frozenset[object], fs))
        and len(cast(frozenset[object], fs)) > 0
    ):
        return sorted(sorted(str(v) for v in s) for s in cast(frozenset[frozenset[str]], fs))
    return [[]]


def cases() -> list[Case]:
    out: list[Case] = []
    zero = np.array(0.0)
    warnings.simplefilter("ignore", FutureWarning)
    for label, (edges, latent, pairs) in GRAPHS.items():
        g = DAG(edges, latents=set(latent))
        ci = CausalInference(g)
        for x, y in pairs:
            try:
                back = sets(ci.get_all_backdoor_adjustment_sets(x, y))
            except ValueError:
                back = []
            front = sorted(sorted(str(v) for v in s) for s in ci.get_all_frontdoor_adjustment_sets(x, y))
            minimal = ci.get_minimal_adjustment_set(x, y)
            out.append(
                case(
                    f"{label}/{x}-{y}",
                    "sets",
                    [],
                    zero,
                    edges=[list(e) for e in edges],
                    latent=latent,
                    x=x,
                    y=y,
                    backdoor=back,
                    frontdoor=front,
                    minimalSize=-1 if cast(object, minimal) is None else len(minimal),
                    valid=[bool(ci.is_valid_backdoor_adjustment_set(x, y, cast(list[Hashable], s))) for s in back],
                )
            )
    return out
