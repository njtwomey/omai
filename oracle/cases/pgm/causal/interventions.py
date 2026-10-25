"""Interventional distributions against pgmpy's CausalInference.query with `do` (and evidence), on the Asia, student
and sprinkler networks, checked against and stored as the exact distribution of the mutilated network (a fixed-order
einsum), since pgmpy's results vary in the last bits with set order. The evidence is never a descendant of the
treatment: pgmpy adjusts for it as a pre-treatment covariate, which is the mutilated network's answer only then."""

from collections.abc import Hashable
from typing import cast

import pgmpy
from mdml_oracle.fixture import Case, case
from mdml_oracle.pgm import NETWORKS, NodeSpec, checked, definition, exact_posterior, ordered, pgmpy_model
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import CausalInference

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__}

QUERIES: dict[str, list[tuple[list[str], dict[str, str], dict[str, str]]]] = {
    "asia": [
        (["dysp"], {"smoke": "yes"}, {}),
        (["xray"], {"either": "no"}, {}),
        (["dysp"], {"lung": "yes"}, {"asia": "yes"}),
    ],
    "student": [(["letter"], {"grade": "A"}, {}), (["letter"], {"intelligence": "high"}, {"difficulty": "hard"})],
    "sprinkler": [
        (["wet"], {"sprinkler": "on"}, {}),
        (["wet"], {"sprinkler": "on"}, {"cloudy": "yes"}),
        (["wet"], {"rain": "yes"}, {}),
    ],
}


def mutilate(spec: list[NodeSpec], do: dict[str, str]) -> list[NodeSpec]:
    return [
        (n, s, [], [[1.0 if x == do[n] else 0.0 for x in s]]) if n in do else (n, s, ps, rows)
        for n, s, ps, rows in spec
    ]


def cases() -> list[Case]:
    out: list[Case] = []
    for label, spec in NETWORKS.items():
        model = pgmpy_model(spec)
        engine = CausalInference(model)
        for i, (variables, do, evidence) in enumerate(QUERIES[label]):
            result = engine.query(
                cast(list[Hashable], variables),
                do=cast(dict[Hashable, int], do),
                evidence=cast(dict[Hashable, int], evidence) if evidence else None,
                show_progress=False,
            )
            assert isinstance(result, DiscreteFactor)
            values = ordered(result, variables)
            exact = exact_posterior(pgmpy_model(mutilate(spec, do)), variables, evidence)
            out.append(
                case(
                    f"{label}/{i}",
                    "do",
                    [],
                    checked(values / values.sum(), exact),
                    rtol=1e-12,
                    atol=1e-14,
                    network=definition(spec),
                    query=variables,
                    do=do,
                    evidence=evidence,
                )
            )
    return out
