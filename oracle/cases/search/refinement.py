"""Refinement search against pysubgroup: exhaustive subgroup discovery over conjunctions of attribute=value selectors
with weighted relative accuracy (WRAcc), by pysubgroup's DFS (optimistic-estimate pruning) and Apriori, on seeded
nominal datasets. The top-k qualities are compared (subgroups of equal quality may come in either order)."""

from importlib.metadata import version

import numpy as np
import pandas as pd
import pysubgroup as ps
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"pysubgroup": version("pysubgroup")}


def dataset(seed: int, n: int) -> pd.DataFrame:
    """Nominal attributes a, b, c, d and a binary target that depends on a and b."""
    r = rng(seed)
    df = pd.DataFrame(
        {
            "a": r.choice(["x", "y", "z"], size=n),
            "b": r.choice(["p", "q"], size=n),
            "c": r.choice(["u", "v", "w"], size=n),
            "d": r.choice(["s", "t"], size=n),
        }
    )
    signal = (df["a"] == "x") & (df["b"] == "p")
    noise = r.random(n) < 0.2
    df["target"] = (signal ^ noise).astype(int)
    return df


def cases() -> list[Case]:
    out: list[Case] = []
    for seed, n, depth, k in [(0, 40, 2, 5), (1, 60, 2, 8), (2, 50, 3, 6)]:
        df = dataset(seed, n)
        target = ps.BinaryTarget("target", 1)
        space = ps.create_selectors(df, ignore=["target"])
        qualities: list[list[float]] = []
        for algorithm in (ps.DFS(), ps.Apriori()):
            task = ps.SubgroupDiscoveryTask(df, target, space, result_set_size=k, depth=depth, qf=ps.WRAccQF())
            result = algorithm.execute(task)
            qualities.append(sorted((float(r[0]) for r in result.results), reverse=True))
        if not np.allclose(qualities[0], qualities[1]):
            raise ValueError("pysubgroup's DFS and Apriori disagree")
        out.append(
            case(
                f"wracc/seed{seed}/depth{depth}/k{k}",
                "wracc",
                [],
                np.array(0.0),
                rtol=1e-12,
                atol=1e-12,
                columns={c: [str(v) for v in df[c]] for c in ["a", "b", "c", "d"]},
                target=[int(v) for v in df["target"]],
                depth=depth,
                k=k,
                qualities=qualities[0],
            )
        )
    return out
