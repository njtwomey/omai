"""Structured kernels: scikit-learn's WhiteKernel and ConstantKernel (with eval_gradient), chi2_kernel and
additive_chi2_kernel (minus the chi-square distance); the spectrum kernel from CountVectorizer character k-gram
counts; set kernels as means of rbf_kernel blocks; the Jaccard kernel from scipy's boolean jaccard distance; graph
diffusion kernels from scipy.linalg.expm of NetworkX Laplacians, regularised Laplacian kernels by numpy inversion, and
Weisfeiler-Lehman subtree kernels from NetworkX's weisfeiler_lehman_subgraph_hashes (round 0 is the initial labels).

The additive chi-square, skewed chi-square and histogram intersection kernels have no library function; they are
formed with numpy from their definitions, and the subsequence kernel by enumerating index tuples (its definition)."""

from collections import Counter
from itertools import combinations
from typing import cast

import networkx as nx
import numpy as np
import scipy
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.graphs import NxGraph, gnp, spec, undirected_family
from omai_oracle.helpers import FloatArray, rng, uniform
from omai_oracle.kernels import gp_case
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
from sklearn.metrics.pairwise import additive_chi2_kernel, chi2_kernel, rbf_kernel

REFERENCE: dict[str, str] = {
    "scikit-learn": sklearn.__version__,
    "scipy": scipy.__version__,
    "networkx": nx.__version__,
}

STRINGS = ["abracadabra", "cadabra", "banana", "bandana", "abba", "ab", "kernel"]


def spectrum(a: list[str], b: list[str], k: int) -> FloatArray:
    """Inner products of character k-gram counts (CountVectorizer, case kept)."""
    v = CountVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False).fit(a + b)
    fa = np.asarray(cast(csr_matrix, v.transform(a)).toarray(), dtype=np.float64)
    fb = np.asarray(cast(csr_matrix, v.transform(b)).toarray(), dtype=np.float64)
    return fa @ fb.T


def ssk(s: str, t: str, n: int, lam: float) -> float:
    """The subsequence kernel by its definition: every pair of index n-tuples spelling the same subsequence."""
    total = 0.0
    for i in combinations(range(len(s)), n):
        u = "".join(s[p] for p in i)
        for jj in combinations(range(len(t)), n):
            if "".join(t[q] for q in jj) == u:
                total += lam ** (i[-1] - i[0] + 1 + jj[-1] - jj[0] + 1)
    return total


def normalised(k: FloatArray, da: FloatArray, db: FloatArray) -> FloatArray:
    """K_ij / sqrt(K_ii K_jj), and 0 where a self-similarity is 0 (a string shorter than the substrings)."""
    d = np.sqrt(np.outer(da, db))
    return np.divide(k, d, out=np.zeros_like(k), where=d > 0)


def wl_features(g: NxGraph, h: int, label: str | None) -> Counter[str]:
    """Round-tagged colour counts: round 0 the initial labels (one constant label for an unlabelled graph), then
    NetworkX's subtree hashes. For an unlabelled graph NetworkX's first hash is the degree, which is round 1 of
    refinement from a constant label, so its h hashes are rounds 1..h either way."""
    counts: Counter[str] = Counter()
    for v in g.nodes():
        initial = str(g.nodes[v][label]) if label is not None else ""
        counts[f"0:{initial}"] += 1
    if h > 0:
        hashes = nx.weisfeiler_lehman_subgraph_hashes(g, node_attr=label, iterations=h)
        for v in g.nodes():
            for r, c in enumerate(hashes[v]):
                counts[f"{r + 1}:{c}"] += 1
    return counts


def wl_kernel(gs: list[NxGraph], h: int, label: str | None) -> FloatArray:
    fs = [wl_features(g, h, label) for g in gs]
    return np.array([[sum(c * f2[k] for k, c in f1.items()) for f2 in fs] for f1 in fs], dtype=np.float64)


def cases() -> list[Case]:
    g = rng(41)
    x = uniform(g, (8, 3))
    y = uniform(g, (5, 3))
    hx = uniform(g, (7, 4), 0.0, 2.0)
    hy = uniform(g, (5, 4), 0.0, 2.0)
    hx[1, 2] = 0.0
    hy[3, 2] = 0.0
    out = [
        gp_case("white", WhiteKernel(0.3), x, y, kernel="white"),
        gp_case("constant", ConstantKernel(1.7), x, y, kernel="constant"),
        case("chi2", "chi2", [hx, hy], chi2_kernel(hx, hy, gamma=0.6), rtol=1e-12, gamma=0.6),
        case("chi2-distance", "chi2Distance", [hx, hy], -additive_chi2_kernel(hx, hy), rtol=1e-12),
    ]
    a = hx[:, None, :]
    b = hy[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        additive = np.where(a + b > 0, 2 * a * b / (a + b), 0.0).sum(2)
    out.append(case("additive-chi2", "additiveChi2", [hx, hy], additive, rtol=1e-12))
    c = 0.4
    skewed = np.prod(2 * np.sqrt(a + c) * np.sqrt(b + c) / (a + b + 2 * c), axis=2)
    out.append(case("skewed-chi2", "skewedChi2", [hx, hy], skewed, rtol=1e-12, skewedness=c))
    out.append(case("histogram-intersection", "intersection", [hx, hy], np.minimum(a, b).sum(2), rtol=1e-12))
    for k in [1, 2, 3]:
        kk = spectrum(STRINGS, STRINGS, k)
        d = np.diag(kk)
        out.append(
            case(f"spectrum-{k}", "spectrum", [], {"raw": kk, "normalised": normalised(kk, d, d)}, k=k, strings=STRINGS)
        )
    short = ["cat", "cart", "chat", "act", "tact"]
    for n, lam in [(1, 0.5), (2, 0.5), (2, 1.0), (3, 0.7)]:
        kk = np.array([[ssk(s, t, n, lam) for t in short] for s in short], dtype=np.float64)
        d = np.diag(kk)
        out.append(
            case(
                f"subsequence-{n}-{lam}",
                "subsequence",
                [],
                {"raw": kk, "normalised": normalised(kk, d, d)},
                rtol=1e-12,
                n=n,
                lam=lam,
                strings=short,
            )
        )
    sets_a = [uniform(g, (m, 2)) for m in (3, 1, 4)]
    sets_b = [uniform(g, (m, 2)) for m in (2, 5)]
    gamma = 0.8
    mean_map = np.array([[rbf_kernel(s, t, gamma=gamma).mean() for t in sets_b] for s in sets_a], dtype=np.float64)
    out.append(case("set-rbf", "set", [*sets_a, *sets_b], mean_map, rtol=1e-12, gamma=gamma, left=len(sets_a)))
    labels_a = [[1, 2, 3], [2, 4], [], [5, 1, 2, 4]]
    labels_b = [[1, 2], [3, 4, 5], []]
    universe = 6

    def indicator(sets: list[list[int]]) -> FloatArray:
        out_: FloatArray = np.zeros((len(sets), universe), dtype=np.float64)
        for i, s in enumerate(sets):
            out_[i, s] = 1.0
        return out_

    jac = 1 - cdist(indicator(labels_a) > 0, indicator(labels_b) > 0, "jaccard")
    out.append(case("jaccard", "jaccard", [], jac, rtol=1e-12, a=labels_a, b=labels_b))
    for label, gr in undirected_family()[:5]:
        lap = np.asarray(nx.laplacian_matrix(gr, nodelist=sorted(gr.nodes())).toarray(), dtype=np.float64)
        norm = np.asarray(nx.normalized_laplacian_matrix(gr, nodelist=sorted(gr.nodes())).toarray(), dtype=np.float64)
        n = lap.shape[0]
        out.append(
            case(
                f"{label}/diffusion",
                "diffusion",
                [],
                {"combinatorial": expm(-0.6 * lap), "normalised": expm(-1.3 * norm)},
                rtol=1e-10,
                atol=1e-12,
                **spec(gr),
            )
        )
        out.append(
            case(
                f"{label}/regularised-laplacian",
                "regularisedLaplacian",
                [],
                np.linalg.inv(np.eye(n) + 0.7 * lap),
                rtol=1e-10,
                atol=1e-12,
                **spec(gr),
            )
        )
    # Karate is left out: its two-digit degrees could make NetworkX's unseparated label concatenation ambiguous.
    graphs = [gr for lab, gr in undirected_family() if lab != "karate"] + [gnp(7, 0.4, 50 + s) for s in range(3)]
    for h in [0, 1, 3]:
        out.append(
            case(
                f"wl-subtree-{h}",
                "wl",
                [],
                wl_kernel(graphs, h, None),
                iterations=h,
                graphs=[dict(spec(gr)) for gr in graphs],
            )
        )
    labelled = [gnp(6, 0.5, 70 + s) for s in range(4)]
    for s, gr in enumerate(labelled):
        nx.set_node_attributes(gr, {v: "ab"[(v + s) % 2] for v in gr.nodes()}, "colour")
    out.append(
        case(
            "wl-subtree-labelled",
            "wl",
            [],
            wl_kernel(labelled, 2, "colour"),
            iterations=2,
            graphs=[dict(spec(gr)) for gr in labelled],
            labels=[["ab"[(v + s) % 2] for v in sorted(gr.nodes())] for s, gr in enumerate(labelled)],
        )
    )
    return out
