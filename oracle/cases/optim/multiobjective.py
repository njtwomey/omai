"""Pareto machinery against pymoo and moocore: non-dominated sorting, crowding distance, hypervolume and contributions.

The NSGA-II cases compare the hypervolume of the final front with pymoo's NSGA-II on the same problem and budget:
random streams differ, so the fronts are compared by their quality, not point by point.
"""

from typing import Any, cast

import moocore  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pymoo  # pyright: ignore[reportMissingTypeStubs]
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from pymoo.algorithms.moo.nsga2 import NSGA2  # pyright: ignore[reportMissingTypeStubs]
from pymoo.operators.survival.rank_and_crowding.metrics import (  # pyright: ignore[reportMissingTypeStubs]
    calc_crowding_distance,
)
from pymoo.optimize import minimize  # pyright: ignore[reportMissingTypeStubs]
from pymoo.problems import get_problem  # pyright: ignore[reportMissingTypeStubs]
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting  # pyright: ignore[reportMissingTypeStubs]

REFERENCE: dict[str, str] = {"pymoo": str(pymoo.__version__), "moocore": str(moocore.__version__)}


def points(seed: int, n: int, m: int, ties: bool = False) -> FloatArray:
    g = rng(seed)
    out = np.round(g.uniform(0, 1, size=(n, m)), 1 if ties else 6)
    return np.asarray(out, dtype=np.float64)


def front_points(seed: int, n: int, m: int) -> FloatArray:
    """Mutually non-dominated points on the unit simplex surface."""
    g = rng(seed)
    p = g.dirichlet(np.ones(m), size=n)
    return np.asarray(p, dtype=np.float64)


def sorting(name: str, f: FloatArray) -> Case:
    fronts = cast(list[Any], NonDominatedSorting().do(f))
    ranks = np.zeros(len(f), dtype=np.int32)
    for r, front in enumerate(fronts):
        ranks[np.asarray(front)] = r
    return case(name, "nonDominatedSort", [f], ranks, rtol=0, atol=0)


def crowding(name: str, f: FloatArray) -> Case:
    # pymoo divides the summed normalised gaps by the number of objectives; Deb's definition (ours) does not.
    cd = np.asarray(calc_crowding_distance(f), dtype=np.float64) * f.shape[1]
    return case(name, "crowdingDistance", [f], cd, rtol=1e-12, atol=1e-12)


def volume(name: str, f: FloatArray, ref: list[float]) -> Case:
    v = float(moocore.hypervolume(f, ref=ref))
    return case(name, "hypervolume", [f], np.array(v), rtol=1e-12, atol=1e-14, reference=ref)


def contributions(name: str, f: FloatArray, ref: list[float]) -> Case:
    c = np.asarray(moocore.hv_contributions(f, ref=ref), dtype=np.float64)
    return case(name, "hypervolumeContributions", [f], c, rtol=1e-10, atol=1e-14, reference=ref)


def nsga(name: str, problem: str, n_var: int, pop: int, gens: int, ref: list[float]) -> Case:
    res = minimize(get_problem(problem, n_var=n_var), NSGA2(pop_size=pop), ("n_gen", gens), seed=1, verbose=False)
    f = np.asarray(res.F, dtype=np.float64)
    v = float(moocore.hypervolume(f, ref=ref))
    # Fronts from different random streams differ by a few thousandths in hypervolume at this budget.
    return case(
        name,
        "nsga2",
        [],
        np.array(v),
        rtol=0,
        atol=0.01,
        problem=problem,
        nVar=n_var,
        popSize=pop,
        generations=gens,
        reference=ref,
    )


def cases() -> list[Case]:
    return [
        sorting("sort/random-2d", points(0, 30, 2)),
        sorting("sort/random-3d", points(1, 30, 3)),
        sorting("sort/ties-2d", points(2, 25, 2, ties=True)),
        sorting("sort/ties-3d", points(3, 25, 3, ties=True)),
        crowding("crowding/2d", front_points(4, 12, 2)),
        crowding("crowding/3d", front_points(5, 12, 3)),
        volume("hv/2d", points(6, 20, 2), [1.1, 1.1]),
        volume("hv/3d", points(7, 15, 3), [1.0, 1.0, 1.0]),
        volume("hv/4d", points(8, 12, 4), [1.1, 1.1, 1.1, 1.1]),
        volume("hv/5d", points(9, 10, 5), [1.0, 1.0, 1.0, 1.0, 1.0]),
        volume("hv/front-3d", front_points(10, 15, 3), [1.0, 1.0, 1.0]),
        contributions("hvc/2d", front_points(11, 8, 2), [1.1, 1.1]),
        contributions("hvc/3d", front_points(12, 8, 3), [1.1, 1.1, 1.1]),
        nsga("nsga2/zdt1", "zdt1", 6, 40, 60, [1.1, 1.1]),
        nsga("nsga2/zdt2", "zdt2", 6, 40, 80, [1.1, 1.1]),
    ]
