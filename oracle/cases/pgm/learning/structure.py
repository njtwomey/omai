"""Structure learning against pgmpy: Chow-Liu trees and tree-augmented naive Bayes (TreeSearch), the PC algorithm
(stable variant, chi-square and Fisher z tests) and hill climbing (BIC, K2, BDeu) on data sampled from the student and
sprinkler networks and from a linear-Gaussian network.

Hill climbing breaks ties between equally scored moves by set order, so it runs in a child interpreter with a fixed
hash seed."""

import json
import os
import subprocess
import sys
from typing import cast

import networkx as nx
import numpy as np
import pandas as pd
import pgmpy
from pgmpy.base import DAG, PDAG
from pgmpy.causal_discovery import PC
from pgmpy.estimators import TreeSearch
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.pgm import SPRINKLER, STUDENT, sample_rows

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__, "networkx": nx.__version__}


def edges_of(g: DAG) -> list[list[str]]:
    return sorted([[str(a), str(b)] for a, b in g.edges()])


HILL_CLIMB = """
import json, sys
import pandas as pd
from pgmpy.causal_discovery import HillClimbSearch
from pgmpy.structure_score import BDeu, BIC, K2
spec = json.load(sys.stdin)
frame = pd.DataFrame(spec["rows"])
states = spec["states"]
out = {}
for name, score in [("bic", BIC(frame, state_names=states)), ("k2", K2(frame, state_names=states)),
                    ("bdeu", BDeu(frame, equivalent_sample_size=10, state_names=states))]:
    hc = HillClimbSearch(scoring_method=score, show_progress=False, return_type="dag")
    hc.fit(frame)
    g = hc.causal_graph_
    out[name] = [float(score.score(g)), sorted([[str(a), str(b)] for a, b in g.edges()])]
print(json.dumps(out))
"""


def hill_climb(rows: list[dict[str, str]], states: dict[str, list[str]]) -> dict[str, tuple[float, list[list[str]]]]:
    """pgmpy's HillClimbSearch with each score, in a child interpreter with a fixed hash seed: pgmpy breaks ties
    between equally scored moves by set order, which otherwise varies between Python runs."""
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    result = subprocess.run(
        [sys.executable, "-c", HILL_CLIMB],
        input=json.dumps({"rows": rows, "states": states}),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    parsed: dict[str, list[object]] = json.loads(result.stdout.strip().splitlines()[-1])
    return {k: (float(cast(float, v[0])), cast(list[list[str]], v[1])) for k, v in parsed.items()}


def cases() -> list[Case]:
    out: list[Case] = []
    zero = np.array(0.0)
    for label, spec, n, seed in [("student", STUDENT, 1500, 31), ("sprinkler", SPRINKLER, 1500, 32)]:
        rows = sample_rows(spec, n, seed)
        frame = pd.DataFrame(rows)
        states = {name: s for name, s, _, _ in spec}
        names = [name for name, _, _, _ in spec]
        tree = TreeSearch(frame, root_node=names[0]).estimate(show_progress=False)
        assert isinstance(tree, DAG)
        out.append(case(f"{label}/chowLiu", "chowLiu", [], zero, data=rows, root=names[0], edges=edges_of(tree)))
        tan = TreeSearch(frame, root_node=names[1]).estimate(
            estimator_type="tan", class_node=names[-1], show_progress=False
        )
        assert isinstance(tan, DAG)
        out.append(
            case(
                f"{label}/tan", "tan", [], zero, data=rows, root=names[1], classVariable=names[-1], edges=edges_of(tan)
            )
        )
        est = PC(
            variant="stable", ci_test="chi_square", significance_level=0.01, show_progress=False, return_type="pdag"
        )
        est.fit(frame)
        p = est.causal_graph_
        assert isinstance(p, PDAG)
        directed = sorted([[str(a), str(b)] for a, b in p.directed_edges])
        undirected = sorted([sorted([str(a), str(b)]) for a, b in p.undirected_edges])
        out.append(
            case(f"{label}/pc", "pc", [], zero, data=rows, test="chiSquare", directed=directed, undirected=undirected)
        )
        for name, (score_value, edges) in hill_climb(rows, states).items():
            out.append(
                case(
                    f"{label}/hillClimb/{name}",
                    "hillClimb",
                    [],
                    np.array(score_value),
                    rtol=1e-10,
                    data=rows,
                    states=states,
                    score=name,
                    edges=edges,
                )
            )
    g = rng(17)
    n = 1000
    a = g.normal(size=n)
    b = g.normal(size=n)
    c = 0.8 * a + 0.6 * b + g.normal(size=n) * 0.5
    d = -0.7 * c + g.normal(size=n) * 0.5
    e = 0.5 * d + g.normal(size=n) * 0.7
    cont = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    data = [{k: float(v) for k, v in r.items()} for r in cont.to_dict(orient="records")]
    est = PC(variant="stable", ci_test="fisher_z", significance_level=0.01, show_progress=False, return_type="pdag")
    est.fit(cont)
    p = est.causal_graph_
    assert isinstance(p, PDAG)
    directed = sorted([[str(x), str(y)] for x, y in p.directed_edges])
    undirected = sorted([sorted([str(x), str(y)]) for x, y in p.undirected_edges])
    out.append(case("gaussian/pc", "pc", [], zero, data=data, test="fisherZ", directed=directed, undirected=undirected))
    return out
