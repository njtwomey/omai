"""Graph measures against NetworkX: density, eccentricity, diameter, radius, centre, periphery, triangles, clustering
(unweighted, weighted, directed), transitivity, assortativity (degree, numeric, attribute), conductance, edge expansion
and algebraic connectivity; the random-walk spectral gap from numpy's eigvalsh of the normalised Laplacian."""

from collections.abc import Mapping
from typing import cast

import networkx as nx
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.graphs import NxGraph, digraph, directed_family, spec, undirected_family, weighted_family
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"networkx": nx.__version__}


def vector(g: NxGraph, values: object) -> FloatArray:
    """A node-indexed mapping (NetworkX returns dicts typed as unions) as an array in node order."""
    table = cast(Mapping[int, float], values)
    return np.array([float(table[v]) for v in range(g.number_of_nodes())])


def cases() -> list[Case]:
    out: list[Case] = []
    for label, g in undirected_family() + weighted_family() + directed_family():
        n = g.number_of_nodes()
        params = spec(g)
        out.append(case(f"{label}/density", "density", [], np.array(nx.density(g)), **params))
        connected = nx.is_strongly_connected(digraph(g)) if g.is_directed() else nx.is_connected(g)
        if connected:
            ecc = nx.eccentricity(g)
            out.append(
                case(
                    f"{label}/distances",
                    "distances",
                    [],
                    vector(g, ecc),
                    **params,
                    diameter=int(nx.diameter(g)),
                    radius=int(nx.radius(g)),
                    center=sorted(int(v) for v in nx.center(g)),
                    periphery=sorted(int(v) for v in nx.periphery(g)),
                    weightedDiameter=float(nx.diameter(g, weight="weight")),
                )
            )
        out.append(case(f"{label}/clustering", "clustering", [], vector(g, nx.clustering(g)), rtol=1e-12, **params))
        out.append(
            case(
                f"{label}/clustering-weighted",
                "clusteringWeighted",
                [],
                vector(g, nx.clustering(g, weight="weight")),
                rtol=1e-12,
                **params,
            )
        )
        out.append(
            case(f"{label}/average-clustering", "averageClustering", [], np.array(nx.average_clustering(g)), **params)
        )
        # Regular graphs have no degree variance (the coefficient is 0 / 0).
        if g.number_of_edges() > 1 and len({d for _, d in g.degree()}) > 1:
            out.append(
                case(
                    f"{label}/assortativity",
                    "assortativity",
                    [],
                    np.array(nx.degree_assortativity_coefficient(g)),
                    rtol=1e-10,
                    **params,
                )
            )
        values = [float(x) for x in rng(n).normal(size=n)]
        cats = [int(x) for x in rng(n + 1).integers(0, 3, size=n)]
        for v in range(n):
            g.nodes[v]["value"] = values[v]
            g.nodes[v]["cat"] = cats[v]
        out.append(
            case(
                f"{label}/attribute-assortativity",
                "attributeAssortativity",
                [],
                np.array(
                    [nx.numeric_assortativity_coefficient(g, "value"), nx.attribute_assortativity_coefficient(g, "cat")]
                ),
                rtol=1e-10,
                **params,
                values=values,
                categories=cats,
            )
        )
        subset = [v for v in range(n) if cats[v] == 0]
        if 0 < len(subset) < n:
            out.append(
                case(
                    f"{label}/expansion",
                    "expansion",
                    [],
                    np.array([nx.conductance(g, subset), nx.edge_expansion(g, subset)]),
                    **params,
                    subset=subset,
                )
            )
        if not g.is_directed():
            out.append(
                case(
                    f"{label}/triangles",
                    "triangles",
                    [],
                    vector(g, nx.triangles(g)),
                    **params,
                    transitivity=float(nx.transitivity(g)),
                )
            )
            if connected:
                lap = np.asarray(nx.normalized_laplacian_matrix(g).toarray(), dtype=np.float64)
                mu = np.linalg.eigvalsh(lap)
                out.append(
                    case(
                        f"{label}/spectrum",
                        "spectrum",
                        [],
                        # NetworkX's Lanczos run is reproducible only to rounding, so the fixture keeps 10 decimals.
                        np.round(
                            np.array(
                                [
                                    nx.algebraic_connectivity(g, tol=1e-12, method="lanczos", seed=0),
                                    nx.algebraic_connectivity(g, normalized=True, tol=1e-12, method="lanczos", seed=0),
                                    mu[1],
                                    1 - max(abs(1 - mu[1]), abs(1 - mu[-1])),
                                ]
                            ),
                            10,
                        ),
                        rtol=1e-7,
                        atol=1e-9,
                        **params,
                    )
                )
    return out
