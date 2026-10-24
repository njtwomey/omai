"""Relaxation heuristics against pyperplan's hAddHeuristic and hMaxHeuristic (Bonet and Geffner's h-add and h-max):
their values on every state along an optimal plan of each PDDL instance. pyperplan's FF heuristic is not compared:
its relaxed plan breaks cost ties by the iteration order of a Python set, which varies between runs; whitebox-js's
h-FF is checked by properties (between h-max and h-add, a valid relaxed plan)."""

from importlib.metadata import version

import numpy as np
from pyperplan.heuristics.relaxation import hAddHeuristic, hMaxHeuristic
from pyperplan.search import breadth_first_search
from pyperplan.search.searchspace import make_root_node
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.planning import INSTANCES, task

REFERENCE: dict[str, str] = {"pyperplan": version("pyperplan")}


def cases() -> list[Case]:
    out: list[Case] = []
    for name, domain, problem in INSTANCES:
        t = task(domain, problem)
        plan = breadth_first_search(t)
        if plan is None:
            raise ValueError(f"{name}: pyperplan found no plan")
        states = [t.initial_state]
        for op in plan:
            states.append(op.apply(states[-1]))
        hadd = hAddHeuristic(t)
        hmax = hMaxHeuristic(t)
        out.append(
            case(
                name,
                "relaxation",
                [],
                np.array(0.0),
                domain=domain,
                problem=problem,
                states=[sorted(s) for s in states],
                hAdd=[float(hadd(make_root_node(s))) for s in states],
                hMax=[float(hmax(make_root_node(s))) for s in states],
            )
        )
    return out
