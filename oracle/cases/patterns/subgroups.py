"""Subgroup discovery against pysubgroup: exhaustive search (DFS with optimistic estimates) over conjunctions of
attribute=value selectors, for the Klosgen family of quality functions on a binary target (StandardQF with a = 0, 0.5
and 1, where a = 1 is weighted relative accuracy) and the size-weighted mean shift of a numeric target
(StandardQFNumeric), on seeded nominal datasets. The top-k qualities are compared (subgroups of equal quality may come
in either order)."""

from importlib.metadata import version

import numpy as np
import pandas as pd
import pysubgroup as ps
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"pysubgroup": version("pysubgroup")}


def dataset(seed: int, n: int) -> pd.DataFrame:
    """Nominal attributes a, b, c, d, a binary target depending on a and b, and a numeric target with a shifted mean."""
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
    df["value"] = np.round(2.0 * signal.astype(float) + r.normal(size=n), 6)
    return df


def cases() -> list[Case]:
    out: list[Case] = []
    for seed, n, depth, k in [(0, 40, 2, 5), (1, 60, 2, 8), (2, 50, 3, 6)]:
        df = dataset(seed, n)
        columns = {c: [str(v) for v in df[c]] for c in ["a", "b", "c", "d"]}
        space = ps.create_selectors(df, ignore=["target", "value"])
        for a in (0.0, 0.5, 1.0):
            task = ps.SubgroupDiscoveryTask(
                df,
                ps.BinaryTarget("target", 1),
                space,
                result_set_size=k,
                depth=depth,
                qf=ps.StandardQF(a),
            )
            result = ps.DFS().execute(task)
            qualities = sorted((float(r[0]) for r in result.results), reverse=True)
            out.append(
                case(
                    f"binary/a{a}/seed{seed}/depth{depth}/k{k}",
                    "standard-quality",
                    [],
                    np.array(0.0),
                    rtol=1e-12,
                    atol=1e-12,
                    columns=columns,
                    target=[int(v) for v in df["target"]],
                    a=a,
                    depth=depth,
                    k=k,
                    qualities=qualities,
                )
            )
        task = ps.SubgroupDiscoveryTask(
            df,
            ps.NumericTarget("value"),
            space,
            result_set_size=k,
            depth=depth,
            qf=ps.StandardQFNumeric(1.0, False, "sum"),
        )
        result = ps.DFS().execute(task)
        out.append(
            case(
                f"numeric/seed{seed}/depth{depth}/k{k}",
                "mean-shift",
                [],
                np.array(0.0),
                rtol=1e-12,
                atol=1e-12,
                columns=columns,
                value=[float(v) for v in df["value"]],
                depth=depth,
                k=k,
                qualities=sorted((float(r[0]) for r in result.results), reverse=True),
            )
        )
    return out
