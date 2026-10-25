"""Graph matrices against NetworkX (to_numpy_array, incidence_matrix, laplacian_matrix, normalized_laplacian_matrix,
modularity_matrix) and scipy.sparse.csgraph.laplacian. The random-walk Laplacian, the transition matrix and the
positional encodings have no library function, so they are formed from those matrices with numpy (eigh for the
Laplacian encodings, with the sign convention of mdml-js), on graphs with distinct eigenvalues; so are the GCN
normalised adjacency D~^{-1/2} (A + I) D~^{-1/2} and the mean aggregation D~^{-1} (A + I)."""

import networkx as nx
import numpy as np
import scipy
from mdml_oracle.fixture import Case, case
from mdml_oracle.graphs import dense, directed_family, gnp, spec, undirected_family, weighted_family
from mdml_oracle.helpers import FloatArray, sign_fixed
from scipy.sparse.csgraph import laplacian as csgraph_laplacian

REFERENCE: dict[str, str] = {"networkx": nx.__version__, "scipy": scipy.__version__}


def transition(a: FloatArray) -> FloatArray:
    """D^{-1} A with uniform rows for dangling nodes."""
    d = a.sum(axis=1, keepdims=True)
    n = a.shape[0]
    return np.where(d > 0, a / np.where(d > 0, d, 1.0), 1.0 / n)


def cases() -> list[Case]:
    out: list[Case] = []
    for label, g in undirected_family() + weighted_family() + directed_family():
        a = dense(g)
        nodes = sorted(g.nodes())
        out.append(case(f"{label}/adjacency", "adjacency", [], a, **spec(g)))
        out.append(
            case(
                f"{label}/incidence",
                "incidence",
                [],
                np.asarray(nx.incidence_matrix(g, nodelist=nodes, edgelist=sorted(g.edges()), oriented=True).toarray()),
                **spec(g),
            )
        )
        out.append(case(f"{label}/transition", "transition", [], transition(a), **spec(g)))
        if g.is_directed():
            continue
        lap = np.asarray(nx.laplacian_matrix(g, nodelist=nodes).toarray(), dtype=np.float64)
        out.append(case(f"{label}/laplacian", "laplacian", [], lap, **spec(g)))
        out.append(
            case(
                f"{label}/csgraph-laplacian",
                "laplacian",
                [],
                np.asarray(csgraph_laplacian(a), dtype=np.float64),
                **spec(g),
            )
        )
        sym = np.asarray(nx.normalized_laplacian_matrix(g, nodelist=nodes).toarray(), dtype=np.float64)
        out.append(case(f"{label}/normalised", "normalised", [], sym, **spec(g)))
        d = a.sum(axis=1)
        inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
        out.append(case(f"{label}/random-walk", "randomWalk", [], inv[:, None] * lap, **spec(g)))
        out.append(
            case(
                f"{label}/modularity",
                "modularityMatrix",
                [],
                np.asarray(nx.modularity_matrix(g, nodelist=nodes, weight="weight")),
                **spec(g),
            )
        )
        tilde = a + np.eye(a.shape[0])
        dt = tilde.sum(axis=1)
        out.append(case(f"{label}/gcn", "gcn", [], tilde / np.sqrt(np.outer(dt, dt)), **spec(g)))
        out.append(case(f"{label}/mean", "meanAggregation", [], tilde / dt[:, None], **spec(g)))
        p = transition(a) * (d[:, None] > 0)
        powers = [np.diag(np.linalg.matrix_power(p, k)) for k in range(1, 5)]
        out.append(case(f"{label}/rwpe", "rwpe", [], np.stack(powers, axis=1), **spec(g)))
    for seed in range(3):
        g = gnp(9, 0.5, 40 + seed, weights=(1, 9))
        if not nx.is_connected(g):
            continue
        sym = np.asarray(nx.normalized_laplacian_matrix(g, nodelist=sorted(g.nodes())).toarray(), dtype=np.float64)
        w, v = np.linalg.eigh(sym)
        gaps = np.diff(w)
        out.append(
            case(
                f"lappe-{seed}",
                "lappe",
                [],
                sign_fixed(v)[:, 1:4],
                rtol=1e-9,
                atol=1e-10,
                **spec(g),
                minGap=float(gaps.min()),
            )
        )
    return out
