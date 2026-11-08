"""Uninformed search against aima-python (a typed port in omai_oracle.aima_search: breadth-first and depth-first
graph search, depth-limited and iterative deepening search, uniform-cost search) on the Romania map, the eight puzzle
and incremental n-queens, comparing solutions, costs and aima-python's expansion, goal-test and generation counts.
Uniform-cost costs on the Romania map are also checked against NetworkX's Dijkstra, the cost bidirectional search
must reach."""

from collections.abc import Callable

import networkx as nx
import numpy as np
from omai_oracle.aima_search import (
    ROMANIA,
    EightPuzzle,
    GraphProblem,
    Node,
    NQueensProblem,
    Problem,
    breadth_first_graph_search,
    depth_first_graph_search,
    depth_limited_search,
    iterative_deepening_search,
    run,
    undirected,
    uniform_cost_search,
)
from omai_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"aima-python": "port of search.py (2024)", "networkx": nx.__version__}

type Search = Callable[[Problem], Node | str | None]

SEARCHES: dict[str, Search] = {
    "breadthFirst": breadth_first_graph_search,
    "depthFirst": depth_first_graph_search,
    "iterativeDeepening": iterative_deepening_search,
    "uniformCost": uniform_cost_search,
}

ROUTES: list[tuple[str, str]] = [("Arad", "Bucharest"), ("Oradea", "Neamt"), ("Timisoara", "Eforie")]

PUZZLES: list[tuple[int, ...]] = [
    (1, 2, 3, 4, 5, 6, 0, 7, 8),
    (1, 2, 3, 0, 4, 6, 7, 5, 8),
    (4, 1, 3, 7, 2, 6, 0, 5, 8),
    (1, 3, 0, 4, 2, 5, 7, 8, 6),
]


def romania() -> dict[str, dict[str, int]]:
    """The undirected Romania map in aima-python's neighbour order."""
    return undirected(ROMANIA)


def limited(limit: int) -> Search:
    """Depth-limited search with a fixed limit."""

    def search(problem: Problem) -> Node | str | None:
        return depth_limited_search(problem, limit)

    return search


def road_map() -> list[list[object]]:
    """The map as a fixture parameter: each city with its neighbours in order and their road lengths."""
    return [[city, [[b, d] for b, d in nbrs.items()]] for city, nbrs in romania().items()]


def cases() -> list[Case]:
    out: list[Case] = []
    graph = romania()
    nxg: nx.Graph[str] = nx.Graph()
    for a, nbrs in graph.items():
        for b, d in nbrs.items():
            nxg.add_edge(a, b, weight=d)
    for start, goal in ROUTES:
        for name, search in SEARCHES.items():
            if name == "iterativeDeepening" and start != "Arad":
                continue
            out.append(
                case(
                    f"romania/{start}-{goal}/{name}",
                    name,
                    [],
                    np.array(0.0),
                    problem="romania",
                    start=start,
                    goal=goal,
                    neighbours=road_map(),
                    expected=run(search, GraphProblem(start, goal, graph)),
                )
            )
        for limit in (2, 3, 4):
            out.append(
                case(
                    f"romania/{start}-{goal}/depthLimited{limit}",
                    "depthLimited",
                    [],
                    np.array(0.0),
                    problem="romania",
                    start=start,
                    goal=goal,
                    limit=limit,
                    neighbours=road_map(),
                    expected=run(limited(limit), GraphProblem(start, goal, graph)),
                )
            )
        out.append(
            case(
                f"romania/{start}-{goal}/networkx",
                "bidirectional",
                [],
                np.array(0.0),
                problem="romania",
                start=start,
                goal=goal,
                neighbours=road_map(),
                cost=float(nx.dijkstra_path_length(nxg, start, goal)),
            )
        )
    for tiles in PUZZLES:
        for name in ("breadthFirst", "uniformCost", "iterativeDeepening"):
            out.append(
                case(
                    f"eight-puzzle/{''.join(map(str, tiles))}/{name}",
                    name,
                    [],
                    np.array(0.0),
                    problem="eight-puzzle",
                    tiles=list(tiles),
                    expected=run(SEARCHES[name], EightPuzzle(tiles)),
                )
            )
    for n in (4, 5, 6, 8):
        for name in ("breadthFirst", "depthFirst"):
            out.append(
                case(
                    f"queens/{n}/{name}",
                    name,
                    [],
                    np.array(0.0),
                    problem="queens",
                    n=n,
                    expected=run(SEARCHES[name], NQueensProblem(n)),
                )
            )
    return out
