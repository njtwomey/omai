"""Informed search against aima-python (a typed port in whitebox_oracle.aima_search: A*, greedy best-first and
recursive best-first search) on the Romania map with aima-python's truncated straight-line distance and on the eight
puzzle with its misplaced-squares heuristic and with the Manhattan distance, comparing solutions, costs and expansion,
goal-test and generation counts. A* costs on the Romania map are also checked against NetworkX's `astar_path_length`."""

import math
from collections.abc import Callable

import networkx as nx
import numpy as np
from whitebox_oracle.aima_search import (
    LOCATIONS,
    ROMANIA,
    EightPuzzle,
    GraphProblem,
    ManhattanEightPuzzle,
    Node,
    Problem,
    astar_search,
    greedy_best_first_graph_search,
    recursive_best_first_search,
    run,
    undirected,
)
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"aima-python": "port of search.py (2024)", "networkx": nx.__version__}

SEARCHES: dict[str, Callable[[Problem], Node | str | None]] = {
    "aStar": astar_search,
    "greedy": greedy_best_first_graph_search,
    "recursiveBestFirst": recursive_best_first_search,
}

ROUTES: list[tuple[str, str]] = [("Arad", "Bucharest"), ("Oradea", "Neamt"), ("Timisoara", "Eforie"), ("Iasi", "Arad")]

PUZZLES: list[tuple[int, ...]] = [
    (4, 1, 3, 7, 2, 6, 0, 5, 8),
    (2, 4, 3, 1, 5, 6, 7, 8, 0),
    (1, 6, 2, 5, 3, 0, 4, 7, 8),
    (5, 1, 3, 2, 0, 6, 4, 7, 8),
]


def straight_line(goal: str) -> dict[str, float]:
    """aima-python's GraphProblem.h for every city: the truncated distance between locations."""
    b = LOCATIONS[goal]
    return {c: float(int(math.hypot(a[0] - b[0], a[1] - b[1]))) for c, a in LOCATIONS.items()}


def table_heuristic(h: dict[str, float]) -> Callable[[str, str], float]:
    """A NetworkX A* heuristic reading a table of estimates to the goal."""

    def estimate(u: str, v: str) -> float:
        return h[u]

    return estimate


def cases() -> list[Case]:
    out: list[Case] = []
    graph = undirected(ROMANIA)
    neighbours = [[city, [[b, d] for b, d in nbrs.items()]] for city, nbrs in graph.items()]
    nxg: nx.Graph[str] = nx.Graph()
    for a, nbrs in graph.items():
        for b, d in nbrs.items():
            nxg.add_edge(a, b, weight=d)
    for start, goal in ROUTES:
        h = straight_line(goal)
        for name, search in SEARCHES.items():
            out.append(
                case(
                    f"romania/{start}-{goal}/{name}",
                    name,
                    [],
                    np.array(0.0),
                    problem="romania",
                    start=start,
                    goal=goal,
                    neighbours=neighbours,
                    h=h,
                    expected=run(search, GraphProblem(start, goal, graph)),
                )
            )
        out.append(
            case(
                f"romania/{start}-{goal}/networkx",
                "networkx",
                [],
                np.array(0.0),
                problem="romania",
                start=start,
                goal=goal,
                neighbours=neighbours,
                h=h,
                cost=float(nx.astar_path_length(nxg, start, goal, heuristic=table_heuristic(h))),
            )
        )
    for tiles in PUZZLES:
        for heuristic, cls in (("misplaced", EightPuzzle), ("manhattan", ManhattanEightPuzzle)):
            for name, search in SEARCHES.items():
                if name == "greedy" and heuristic == "misplaced":
                    continue
                out.append(
                    case(
                        f"eight-puzzle/{''.join(map(str, tiles))}/{heuristic}/{name}",
                        name,
                        [],
                        np.array(0.0),
                        problem="eight-puzzle",
                        tiles=list(tiles),
                        heuristic=heuristic,
                        expected=run(search, cls(tiles)),
                    )
                )
    return out
