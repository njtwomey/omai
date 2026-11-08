"""Spanning-tree counts of small grids by the matrix-tree theorem (NetworkX's `number_of_spanning_trees`), against which
the uniformity of Wilson's and the Aldous-Broder algorithms is tested, and a check of the theorem itself with numpy's
determinant of a reduced Laplacian."""

import networkx as nx
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.graphs import NxGraph

REFERENCE: dict[str, str] = {"networkx": nx.__version__, "numpy": np.__version__}


def hex_grid(rows: int, cols: int) -> NxGraph:
    """The odd-r hexagonal grid graph on cells `r * cols + c` (odd rows shifted right)."""
    g: NxGraph = nx.Graph()
    g.add_nodes_from(range(rows * cols))
    for r in range(rows):
        for c in range(cols):
            offsets = [(0, 1), (1, 1), (1, 0)] if r % 2 == 1 else [(0, 1), (1, 0), (1, -1)]
            for dr, dc in offsets:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < rows and 0 <= c2 < cols:
                    g.add_edge(r * cols + c, r2 * cols + c2)
    return g


def cases() -> list[Case]:
    out: list[Case] = []
    for topology, rows, cols in [("square", 2, 2), ("square", 2, 3), ("square", 3, 3), ("hex", 2, 3)]:
        g: NxGraph = (
            nx.convert_node_labels_to_integers(nx.grid_2d_graph(rows, cols), ordering="sorted")
            if topology == "square"
            else hex_grid(rows, cols)
        )
        count = round(float(nx.number_of_spanning_trees(g)))
        lap = nx.laplacian_matrix(g, nodelist=sorted(g.nodes())).toarray().astype(np.float64)
        det = round(float(np.linalg.det(lap[1:, 1:])))
        out.append(
            case(
                f"{topology}/{rows}x{cols}",
                "spanningTrees",
                [],
                np.array(float(count)),
                topology=topology,
                rows=rows,
                cols=cols,
                determinant=det,
            )
        )
    return out
