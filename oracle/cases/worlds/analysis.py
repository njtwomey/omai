"""The cell graph of worlds against NetworkX: connected regions, articulation points (choke points), bridges, dead
ends (degree one) and shortest-path lengths (breadth-first and by terrain cost, Dijkstra), on random square maps under
4- and 8-connectivity (diagonals need both orthogonal neighbours open) and on random hexagonal maps."""

import networkx as nx
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.graphs import NxGraph
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"networkx": nx.__version__}

COST = {".": 1.0, "~": 5.0}
SQUARE4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIAGONALS = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
HEX_EVEN = [(0, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]
HEX_ODD = [(0, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (-1, 1)]


def random_map(g: np.random.Generator, rows: int, cols: int, wall: float, water: float) -> list[str]:
    """A map of floor `.`, water `~` and walls `#`."""
    u = g.uniform(size=(rows, cols))
    return ["".join("#" if x < wall else "~" if x < wall + water else "." for x in row) for row in u]


def moves(rows: list[str], topology: str, connectivity: int) -> list[tuple[int, int]]:
    """Every legal move `(from, to)` between passable cells."""
    nr, nc = len(rows), len(rows[0])
    free = [[ch != "#" for ch in row] for row in rows]
    out: list[tuple[int, int]] = []
    for r in range(nr):
        for c in range(nc):
            if not free[r][c]:
                continue
            offsets = (HEX_ODD if r % 2 == 1 else HEX_EVEN) if topology == "hex" else SQUARE4
            if topology == "square" and connectivity == 8:
                offsets = SQUARE4 + DIAGONALS
            for dr, dc in offsets:
                r2, c2 = r + dr, c + dc
                if not (0 <= r2 < nr and 0 <= c2 < nc) or not free[r2][c2]:
                    continue
                if dr != 0 and dc != 0 and topology == "square" and not (free[r + dr][c] and free[r][c + dc]):
                    continue
                out.append((r * nc + c, r2 * nc + c2))
    return out


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(20261024)
    shapes = [("square", 4), ("square", 8), ("hex", 6)]
    for i in range(12):
        topology, connectivity = shapes[i % 3]
        rows = random_map(g, int(g.integers(3, 8)), int(g.integers(3, 9)), 0.3, 0.15)
        nc = len(rows[0])
        cells = [ch for row in rows for ch in row]
        passable = [k for k, ch in enumerate(cells) if ch != "#"]
        if not passable:
            continue
        edges = moves(rows, topology, connectivity)
        und: NxGraph = nx.Graph()
        und.add_nodes_from(passable)
        und.add_edges_from(edges)
        weighted: NxGraph = nx.DiGraph()
        weighted.add_nodes_from(passable)
        for a, b in edges:
            diagonal = topology == "square" and a // nc != b // nc and a % nc != b % nc
            weighted.add_edge(a, b, weight=COST[cells[b]] * (np.sqrt(2.0) if diagonal else 1.0))
        source = passable[int(g.integers(0, len(passable)))]
        hops = nx.single_source_shortest_path_length(und, source)
        cost = nx.single_source_dijkstra_path_length(weighted, source)
        n = len(cells)
        out.append(
            case(
                f"{topology}{connectivity}/{i}",
                "cellGraph",
                [],
                np.array(0.0),
                rows=rows,
                topology=topology,
                connectivity=connectivity,
                source=source,
                regions=sorted(sorted(int(v) for v in comp) for comp in nx.connected_components(und)),
                chokePoints=sorted(int(v) for v in nx.articulation_points(und)),
                bridges=sorted(sorted([int(a), int(b)]) for a, b in nx.bridges(und)),
                deadEnds=sorted(int(v) for v in und.nodes() if und.degree(v) == 1),
                hops=[float(hops[v]) if v in hops else float("inf") for v in range(n)],
                cost=[float(cost[v]) if v in cost else float("inf") for v in range(n)],
            )
        )
    return out
