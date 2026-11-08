"""Colouring and cliques against NetworkX: greedy_color with the largest_first and saturation_largest_first (DSATUR)
strategies (whose tie-breaking is deterministic, so the colourings match exactly), the number of colours of the other
strategies (an upper bound comparison), find_cliques as sorted sets, and the clique number."""

import networkx as nx
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.graphs import spec, undirected_family

REFERENCE: dict[str, str] = {"networkx": nx.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    for label, g in undirected_family():
        n = g.number_of_nodes()
        largest = nx.greedy_color(g, strategy="largest_first")
        dsatur = nx.greedy_color(g, strategy="saturation_largest_first")
        counts = {
            s: max(nx.greedy_color(g, strategy=s).values(), default=-1) + 1
            for s in ["smallest_last", "independent_set", "connected_sequential_bfs", "connected_sequential_dfs"]
        }
        cliques = sorted(sorted(int(v) for v in c) for c in nx.find_cliques(g))
        out.append(
            case(
                label,
                "colouring",
                [],
                np.array(0.0),
                **spec(g),
                largestFirst=[int(largest[v]) for v in range(n)],
                dsatur=[int(dsatur[v]) for v in range(n)],
                counts={k.replace("_", "-"): int(v) for k, v in counts.items()},
                cliques=cliques,
                cliqueNumber=max(len(c) for c in cliques),
            )
        )
    return out
