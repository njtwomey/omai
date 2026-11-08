"""t-SNE against scikit-learn's own routines: the dense and the nearest-neighbour joint probabilities, the exact and
Barnes-Hut (opening angle 0) objective with its gradient at a given embedding, and the two-stage gradient descent (early
exaggeration, then momentum 0.8 with the update and gains reset) from a fixed start for a fixed number of iterations."""

import importlib
from importlib.metadata import version
from typing import Any

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

T: Any = importlib.import_module("sklearn.manifold._t_sne")


def data() -> FloatArray:
    g = rng(31)
    return np.concatenate([g.normal(0, 1, (16, 4)), g.normal(3, 1, (16, 4)), g.normal(-3, 0.5, (8, 4))])


def cases() -> list[Case]:
    x = data()
    n = x.shape[0]
    perplexity = 7.0
    out: list[Case] = []

    condensed = T._joint_probabilities(pairwise_distances(x, squared=True), perplexity, 0)  # pyright: ignore[reportUnknownMemberType]
    p_full: FloatArray = squareform(condensed)
    out.append(case("joint", "joint", [x], p_full, rtol=1e-5, atol=1e-9, perplexity=perplexity))

    k = min(n - 1, int(3.0 * perplexity + 1))
    knn = NearestNeighbors(n_neighbors=k).fit(x)
    distances = knn.kneighbors_graph(mode="distance")
    distances.data **= 2  # pyright: ignore[reportUnknownMemberType, reportOptionalMemberAccess]
    sparse = T._joint_probabilities_nn(distances, perplexity, 0)  # pyright: ignore[reportUnknownMemberType]
    p_sparse: FloatArray = np.asarray(sparse.toarray(), dtype=np.float64)
    out.append(case("jointNn", "jointNn", [x], p_sparse, rtol=1e-5, atol=1e-9, perplexity=perplexity))

    g = rng(32)
    y = 0.5 * g.standard_normal((n, 2))
    kl, grad = T._kl_divergence(y.ravel(), condensed, 1, n, 2)  # pyright: ignore[reportUnknownMemberType]
    out.append(
        case(
            "kl",
            "kl",
            [p_full, y],
            np.concatenate([[kl], np.asarray(grad)]),
            rtol=1e-9,
            atol=1e-12,
        )
    )
    p_bh = csr_matrix(p_sparse)
    p_bh.sort_indices()
    kl_bh, grad_bh = T._kl_divergence_bh(  # pyright: ignore[reportUnknownMemberType]
        y.ravel(), p_bh, 1, n, 2, angle=0.0, skip_num_points=0, verbose=False, compute_error=True, num_threads=1
    )
    out.append(
        case(
            "klBh",
            "klBh",
            [p_sparse, y],
            np.concatenate([[kl_bh], np.asarray(grad_bh)]),
            rtol=1e-4,
            atol=1e-6,
        )
    )

    # The two-stage descent, as TSNE._tsne runs it.
    learning_rate, exaggeration, exaggeration_iter, max_iter = 10.0, 4.0, 10, 40
    common: dict[str, Any] = {
        "n_iter_check": 1,
        "min_grad_norm": 0.0,
        "learning_rate": learning_rate,
        "verbose": 0,
        "kwargs": {"skip_num_points": 0},
        "args": [condensed * exaggeration, 1, n, 2],
        "n_iter_without_progress": 10**6,
    }
    params, _, i = T._gradient_descent(  # pyright: ignore[reportUnknownMemberType]
        T._kl_divergence,
        y.ravel(),
        it=0,
        max_iter=exaggeration_iter,
        momentum=0.5,
        **common,
    )
    common["args"] = [condensed, 1, n, 2]
    params, error, _ = T._gradient_descent(  # pyright: ignore[reportUnknownMemberType]
        T._kl_divergence,
        params,
        it=i + 1,
        max_iter=max_iter,
        momentum=0.8,
        **common,
    )
    options = {"learningRate": learning_rate, "exaggeration": exaggeration, "exaggerationIter": exaggeration_iter}
    out.append(
        case(
            "descent/embedding",
            "descent",
            [p_full, y],
            np.asarray(params).reshape(n, 2),
            rtol=1e-8,
            atol=1e-10,
            maxIter=max_iter,
            **options,
        )
    )
    out.append(
        case(
            "descent/error",
            "descentError",
            [p_full, y],
            np.float64(error),
            rtol=1e-8,
            atol=1e-12,
            maxIter=max_iter,
            **options,
        )
    )
    return out
