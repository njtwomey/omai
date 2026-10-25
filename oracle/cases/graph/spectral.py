"""Graph signal processing against numpy, scipy and NetworkX: the graph Fourier transform (numpy eigh, with
mdml-js's sign convention), the heat kernel (scipy.linalg.expm), Chebyshev filters
(numpy.polynomial.chebyshev.chebval on the spectrum), spectral graph wavelets (the mexican-hat kernel on the
spectrum; no reference library ships them), the Dirichlet energy, the Fiedler vector (NetworkX fiedler_vector, up to
sign) and the spectral embedding (numpy eigh, compared as a subspace)."""

import networkx as nx
import numpy as np
import scipy
from mdml_oracle.fixture import Case, case
from mdml_oracle.graphs import GraphSpec, NxGraph, gnp, karate, relabelled, spec
from mdml_oracle.helpers import FloatArray, rng, sign_fixed
from numpy.polynomial import chebyshev
from scipy.linalg import expm

REFERENCE: dict[str, str] = {"networkx": nx.__version__, "scipy": scipy.__version__, "numpy": np.__version__}


class SignalSpec(GraphSpec):
    """A graph with a choice of Laplacian and a signal, as fixture parameters."""

    normalised: bool
    signal: list[list[float]]


def graphs() -> list[tuple[str, NxGraph]]:
    """Connected graphs whose Laplacians have distinct eigenvalues (random weights break the symmetries)."""
    out: list[tuple[str, NxGraph]] = []
    for seed in range(4):
        g = gnp(10, 0.45, 60 + seed, weights=(1, 9))
        if nx.is_connected(g):
            out.append((f"weighted-10-{seed}", g))
    return out


def laplacian(g: NxGraph, normalised: bool) -> FloatArray:
    m = nx.normalized_laplacian_matrix(g) if normalised else nx.laplacian_matrix(g)
    return np.asarray(m.toarray(), dtype=np.float64)


def cases() -> list[Case]:
    out: list[Case] = []
    for label, g in graphs():
        n = g.number_of_nodes()
        x = rng(n + len(label)).normal(size=(n, 2))
        for normalised in [False, True]:
            lap = laplacian(g, normalised)
            w, v = np.linalg.eigh(lap)
            u = sign_fixed(v)
            tag = "sym" if normalised else "comb"
            params: SignalSpec = {**spec(g), "normalised": normalised, "signal": x.tolist()}
            out.append(case(f"{label}/{tag}/frequencies", "frequencies", [], w, rtol=1e-10, atol=1e-10, **params))
            out.append(case(f"{label}/{tag}/gft", "gft", [], u.T @ x, rtol=1e-9, atol=1e-10, **params))
            out.append(case(f"{label}/{tag}/heat", "heat", [], expm(-0.7 * lap), rtol=1e-10, atol=1e-12, **params))
            out.append(
                case(
                    f"{label}/{tag}/diffusion", "diffusion", [], expm(-0.7 * lap) @ x, rtol=1e-10, atol=1e-12, **params
                )
            )
            theta = [0.5, -0.3, 0.2, 0.1]
            lmax = 2.0 if normalised else float(2 * np.max(np.diag(lap)))
            spectrum = chebyshev.chebval(2 * w / lmax - 1, theta)
            out.append(
                case(
                    f"{label}/{tag}/chebyshev",
                    "chebyshev",
                    [],
                    v @ np.diag(spectrum) @ v.T @ x,
                    rtol=1e-9,
                    atol=1e-10,
                    **params,
                    theta=theta,
                    lambdaMax=lmax,
                )
            )
            scales = [0.5, 2.0]
            wavelets = np.stack([v @ np.diag(s * w * np.exp(-s * w)) @ v.T @ x for s in scales])
            out.append(
                case(
                    f"{label}/{tag}/wavelets", "wavelets", [], wavelets, rtol=1e-9, atol=1e-10, **params, scales=scales
                )
            )
            out.append(
                case(
                    f"{label}/{tag}/dirichlet", "dirichlet", [], np.array(np.trace(x.T @ lap @ x)), rtol=1e-12, **params
                )
            )
            fiedler = np.asarray(nx.fiedler_vector(g, normalized=normalised, tol=1e-12, seed=0), dtype=np.float64)
            out.append(case(f"{label}/{tag}/fiedler", "fiedler", [], fiedler, rtol=1e-6, atol=1e-7, **params))
            if normalised:
                out.append(case(f"{label}/embedding", "embedding", [], u[:, :3], rtol=1e-9, atol=1e-10, **params))
    g = karate()
    out.append(
        case(
            "karate/fiedler",
            "fiedler",
            [],
            np.asarray(nx.fiedler_vector(g, tol=1e-12, seed=0)),
            rtol=1e-6,
            atol=1e-7,
            **spec(g),
            normalised=False,
        )
    )
    w = np.linalg.eigvalsh(laplacian(g, True))
    out.append(case("karate/iterative", "iterative", [], w[:3], rtol=1e-6, atol=1e-7, **spec(g)))
    ring = relabelled(nx.connected_caveman_graph(3, 4))
    out.append(
        case(
            "caveman/iterative",
            "iterative",
            [],
            np.linalg.eigvalsh(laplacian(ring, True))[:3],
            rtol=1e-6,
            atol=1e-7,
            **spec(ring),
        )
    )
    return out
