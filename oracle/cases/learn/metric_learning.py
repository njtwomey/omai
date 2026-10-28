"""Neighbourhood components analysis against sklearn.neighbors.NeighborhoodComponentsAnalysis: the objective, its
gradient and the soft-neighbour probabilities at given maps (through the estimator's own loss function), and the
objective reached by a fit from the identity, PCA and LDA starts (scored with the same function)."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from scipy.special import softmax
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NeighborhoodComponentsAnalysis

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def blobs(g: np.random.Generator, sizes: list[int], d: int) -> tuple[FloatArray, FloatArray]:
    xs: list[FloatArray] = []
    ys: list[FloatArray] = []
    for c, n in enumerate(sizes):
        xs.append(g.normal(size=(n, d)) * np.linspace(1.5, 0.5, d) + 1.2 * c * np.linspace(1, 0.3, d))
        ys.append(np.full(n, float(c)))
    return np.vstack(xs), np.concatenate(ys)


def objective(a: FloatArray, x: FloatArray, y: FloatArray) -> tuple[float, FloatArray, FloatArray]:
    """sklearn's loss and gradient at a (sign flipped back: the objective is maximised)."""
    nca = NeighborhoodComponentsAnalysis()
    nca.n_iter_ = 1
    nca.verbose = 0
    mask = y[:, None] == y[None, :]
    loss, grad = nca._loss_grad_lbfgs(a.ravel(), x, mask, -1.0)  # pyright: ignore[reportPrivateUsage]
    embedded = x @ a.T
    p = pairwise_distances(embedded, squared=True)
    np.fill_diagonal(p, np.inf)
    p = softmax(-p, axis=1)
    return -float(loss), -np.asarray(grad).reshape(a.shape), (p * mask).sum(axis=1)


def cases() -> list[Case]:
    g = rng(5150)
    out: list[Case] = []
    x, y = blobs(g, [14, 12, 10], 3)
    for name, dp in [("2x3", 2), ("3x3", 3), ("1x3", 1)]:
        a = g.normal(size=(dp, 3)) * 0.7
        value, grad, prob = objective(a, x, y)
        out.append(
            case(
                f"objective/{name}",
                "objective",
                [x, y, a],
                {"value": np.array(value), "gradient": grad, "probabilities": prob},
                rtol=1e-9,
                atol=1e-10,
            )
        )
    for init in ["identity", "pca", "lda"]:
        nca = NeighborhoodComponentsAnalysis(n_components=2, init=init, max_iter=500, tol=1e-10, random_state=0).fit(
            x, y
        )  # pyright: ignore[reportArgumentType]
        value, _, _ = objective(nca.components_, x, y)
        out.append(
            case(
                f"fit/{init}", "fit", [x, y], {"value": np.array(value)}, rtol=1e-4, atol=1e-6, init=init, components=2
            )
        )
    return out
