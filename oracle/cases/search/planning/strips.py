"""STRIPS planning against pyperplan: the grounded initial state and goal of PDDL instances (blocks world, gripper,
the Sussman anomaly) and the optimal plan length found by pyperplan's breadth-first search, confirmed by its A* with
h-max; omai grounds the same PDDL and plans forwards, backwards, by GraphPlan and partial-order planning."""

from importlib.metadata import version

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.planning import INSTANCES, task
from pyperplan.heuristics.relaxation import hMaxHeuristic
from pyperplan.search import astar_search, breadth_first_search

REFERENCE: dict[str, str] = {"pyperplan": version("pyperplan")}


def cases() -> list[Case]:
    out: list[Case] = []
    for name, domain, problem in INSTANCES:
        t = task(domain, problem)
        bfs = breadth_first_search(t)
        astar = astar_search(t, hMaxHeuristic(t))
        if bfs is None or astar is None:
            raise ValueError(f"{name}: pyperplan found no plan")
        if len(bfs) != len(astar):
            raise ValueError(f"{name}: breadth-first and A* plan lengths differ")
        out.append(
            case(
                name,
                "plan",
                [],
                np.array(0.0),
                domain=domain,
                problem=problem,
                initial=sorted(t.initial_state),
                goal=sorted(t.goals),
                length=len(bfs),
                plan=[op.name for op in bfs],
            )
        )
    return out
