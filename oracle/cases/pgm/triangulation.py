"""Graph machinery against networkx: chordality, maximal cliques, connected components, and the min-degree and min-fill
treewidth upper bounds, on seeded random graphs, grids and cycles."""

import networkx as nx
import numpy as np
from mdml_oracle.fixture import Case, case
from networkx.algorithms.approximation import treewidth_min_degree, treewidth_min_fill_in

REFERENCE: dict[str, str] = {"networkx": nx.__version__}


def named(g: "nx.Graph[int]") -> "nx.Graph[str]":
    """Relabel integer vertices as v00, v01, ... so they sort the same way in both languages."""
    h: nx.Graph[str] = nx.Graph()
    h.add_nodes_from(f"v{v:02d}" for v in g.nodes())
    h.add_edges_from((f"v{a:02d}", f"v{b:02d}") for a, b in g.edges())
    return h


def graphs() -> "list[tuple[str, nx.Graph[str]]]":
    out: list[tuple[str, nx.Graph[str]]] = []
    for seed, (n, p) in enumerate([(7, 0.35), (8, 0.4), (9, 0.3), (10, 0.25), (8, 0.6)]):
        out.append((f"gnp-{n}-{seed}", named(nx.gnp_random_graph(n, p, seed=seed))))
    out.append(("grid-3x3", named(nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3)))))
    out.append(("cycle-6", named(nx.cycle_graph(6))))
    out.append(("chordal", named(nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4), (4, 5)]))))
    out.append(("tree", named(nx.balanced_tree(2, 2))))
    return out


def cases() -> list[Case]:
    out: list[Case] = []
    for label, g in graphs():
        vertices = sorted(str(v) for v in g.nodes())
        edges = sorted(sorted((str(a), str(b))) for a, b in g.edges())
        cliques = sorted(sorted(str(v) for v in c) for c in nx.find_cliques(g))
        components = sorted(sorted(str(v) for v in c) for c in nx.connected_components(g))
        width_degree = treewidth_min_degree(g)[0]
        width_fill = treewidth_min_fill_in(g)[0]
        out.append(
            case(
                label,
                "graph",
                [],
                np.array(0.0),
                vertices=vertices,
                edges=edges,
                chordal=bool(nx.is_chordal(g)),
                cliques=cliques,
                components=components,
                minDegreeWidth=int(width_degree),
                minFillWidth=int(width_fill),
            )
        )
    return out
