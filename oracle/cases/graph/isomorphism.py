"""Isomorphism against NetworkX: is_isomorphic and the number of isomorphisms (GraphMatcher, DiGraphMatcher), the
numbers of induced subgraph isomorphisms and of monomorphisms of small patterns, and which pairs of graphs share a
Weisfeiler-Lehman hash (the digests themselves differ: NetworkX uses blake2b)."""

import warnings
from collections.abc import Iterable

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism as iso
from omai_oracle.fixture import Case, case
from omai_oracle.graphs import NxGraph, build, digraph, edges_of, gnp, relabelled, spec
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"networkx": nx.__version__}


def permuted(g: NxGraph, seed: int) -> NxGraph:
    """The same graph with its nodes renumbered by a random permutation."""
    perm = rng(seed).permutation(g.number_of_nodes())
    edges = [(int(perm[u]), int(perm[v]), w) for u, v, w in edges_of(g)]
    return build(g.number_of_nodes(), edges, g.is_directed())


def matcher(g1: NxGraph, g2: NxGraph) -> "iso.GraphMatcher | iso.DiGraphMatcher":
    return iso.DiGraphMatcher(g1, g2) if g1.is_directed() else iso.GraphMatcher(g1, g2)


def count(it: Iterable[object]) -> int:
    return sum(1 for _ in it)


def cases() -> list[Case]:
    out: list[Case] = []
    petersen = relabelled(nx.petersen_graph())
    two_triangles = relabelled(nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3)))
    hexagon = relabelled(nx.cycle_graph(6))
    pairs: list[tuple[str, NxGraph, NxGraph]] = [
        ("petersen", petersen, permuted(petersen, 1)),
        ("hexagon-vs-triangles", hexagon, two_triangles),
        ("grid", relabelled(nx.grid_2d_graph(3, 3)), permuted(relabelled(nx.grid_2d_graph(3, 3)), 2)),
    ]
    for seed in range(3):
        g = gnp(9, 0.4, 70 + seed)
        pairs.append((f"gnp-{seed}", g, permuted(g, seed)))
        h = gnp(9, 0.4, 80 + seed)
        pairs.append((f"gnp-different-{seed}", g, h))
        d = gnp(7, 0.3, 90 + seed, directed=True)
        pairs.append((f"digraph-{seed}", d, permuted(d, seed)))
        pairs.append((f"digraph-reversed-{seed}", d, relabelled(digraph(d).reverse())))
    for label, g1, g2 in pairs:
        same = bool(nx.is_isomorphic(g1, g2))
        out.append(
            case(
                label,
                "isomorphism",
                [],
                np.array(0.0),
                **spec(g1),
                n2=g2.number_of_nodes(),
                directed2=g2.is_directed(),
                edges2=spec(g2)["edges"],
                isomorphic=same,
                count=count(matcher(g1, g2).isomorphisms_iter()),
            )
        )
    patterns: list[tuple[str, NxGraph]] = [
        ("path3", relabelled(nx.path_graph(3))),
        ("triangle", relabelled(nx.cycle_graph(3))),
        ("square", relabelled(nx.cycle_graph(4))),
        ("star3", relabelled(nx.star_graph(3))),
    ]
    hosts: list[tuple[str, NxGraph]] = [
        ("gnp-10", gnp(10, 0.45, 5)),
        ("petersen", petersen),
        ("grid-3x3", relabelled(nx.grid_2d_graph(3, 3))),
    ]
    for hname, host in hosts:
        for pname, pattern in patterns:
            m = matcher(host, pattern)
            induced = count(m.subgraph_isomorphisms_iter())
            mono = count(matcher(host, pattern).subgraph_monomorphisms_iter())
            out.append(
                case(
                    f"{hname}/{pname}",
                    "subgraph",
                    [],
                    np.array([float(induced), float(mono)]),
                    **spec(host),
                    n2=pattern.number_of_nodes(),
                    directed2=False,
                    edges2=spec(pattern)["edges"],
                )
            )
    dhost = gnp(9, 0.35, 11, directed=True)
    for pname, pattern in [
        ("dipath", build(3, [(0, 1, 1.0), (1, 2, 1.0)], True)),
        ("dicycle", build(3, [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)], True)),
        ("reciprocal", build(2, [(0, 1, 1.0), (1, 0, 1.0)], True)),
    ]:
        induced = count(matcher(dhost, pattern).subgraph_isomorphisms_iter())
        mono = count(matcher(dhost, pattern).subgraph_monomorphisms_iter())
        out.append(
            case(
                f"digraph/{pname}",
                "subgraph",
                [],
                np.array([float(induced), float(mono)]),
                **spec(dhost),
                n2=pattern.number_of_nodes(),
                directed2=True,
                edges2=spec(pattern)["edges"],
            )
        )
    family = [hexagon, two_triangles, petersen, permuted(petersen, 3), gnp(9, 0.4, 70), permuted(gnp(9, 0.4, 70), 5)]
    family += [gnp(9, 0.4, 71), relabelled(nx.path_graph(6)), relabelled(nx.star_graph(5))]
    with warnings.catch_warnings():
        # NetworkX notes that its digests of unlabelled graphs changed in 3.5; only equalities are compared here.
        warnings.simplefilter("ignore", UserWarning)
        hashes = [nx.weisfeiler_lehman_graph_hash(g, iterations=3) for g in family]
    equal = [[hashes[i] == hashes[j] for j in range(len(family))] for i in range(len(family))]
    out.append(
        case(
            "wl-hash",
            "wlHash",
            [],
            np.array(0.0),
            graphs=[spec(g) for g in family],
            equal=equal,
        )
    )
    labelled = [relabelled(nx.cycle_graph(6)), permuted(hexagon, 1), relabelled(nx.cycle_graph(6))]
    labels = [[v % 2 for v in range(6)], [0] * 6, [v % 3 for v in range(6)]]
    for g, lab in zip(labelled, labels, strict=True):
        for v in range(6):
            g.nodes[v]["label"] = str(lab[v])
    lhashes = [nx.weisfeiler_lehman_graph_hash(g, node_attr="label", iterations=2) for g in labelled]
    out.append(
        case(
            "wl-hash-labelled",
            "wlHashLabelled",
            [],
            np.array(0.0),
            graphs=[spec(g) for g in labelled],
            labels=[[str(x) for x in lab] for lab in labels],
            equal=[[lhashes[i] == lhashes[j] for j in range(3)] for i in range(3)],
        )
    )
    return out
