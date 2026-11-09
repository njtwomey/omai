"""Truncated SVD against sklearn.decomposition.TruncatedSVD: the components, singular values, explained variance and its
ratio, and the scores of held-out rows, for the exact (ARPACK) solver on tall and wide data, and for the randomised solver on
a matrix of exact low rank (where every solver finds the same subspace, so the random sketch does not matter). Signs are
fixed so that the largest-magnitude loading of each component is positive."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.decomposition import TruncatedSVD

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def fix_signs(components: FloatArray) -> FloatArray:
    """Make the largest-magnitude loading of every component positive."""
    out = np.array(components)
    for c in range(out.shape[0]):
        if out[c, np.argmax(np.abs(out[c]))] < 0:
            out[c] *= -1
    return out


def cases() -> list[Case]:
    g = rng(12)
    out: list[Case] = []
    problems: list[tuple[str, FloatArray, FloatArray, int, str]] = []
    tall = g.gamma(2.0, size=(50, 7))
    problems.append(("arpack/tall", tall, g.gamma(2.0, size=(6, 7)), 3, "arpack"))
    wide = g.normal(size=(12, 30)) + 1.0
    problems.append(("arpack/wide", wide, g.normal(size=(5, 30)) + 1.0, 4, "arpack"))
    low = g.normal(size=(60, 3)) @ g.normal(size=(3, 15)) + 2.0
    problems.append(("arpack/low-rank", low, g.normal(size=(4, 3)) @ g.normal(size=(3, 15)), 2, "arpack"))
    exact = g.gamma(2.0, size=(40, 3)) @ g.gamma(2.0, size=(3, 12))
    problems.append(
        ("randomized/exact-rank", exact, g.gamma(2.0, size=(5, 3)) @ g.gamma(2.0, size=(3, 12)), 3, "randomized")
    )
    for name, x, held, k, algorithm in problems:
        model = TruncatedSVD(n_components=k, algorithm=algorithm, n_iter=7, random_state=0).fit(x)  # pyright: ignore[reportArgumentType]
        components = fix_signs(model.components_)
        flips = np.sign(np.sum(components * model.components_, axis=1))
        out.append(
            case(
                name,
                "svd",
                [x, held],
                {
                    "components": components,
                    "singularValues": model.singular_values_,
                    "explainedVariance": model.explained_variance_,
                    "explainedVarianceRatio": model.explained_variance_ratio_,
                    "transform": model.transform(held) * flips,
                    "inverse": (model.transform(held) * flips) @ components,
                },
                rtol=1e-7,
                atol=1e-9,
                solver="full" if algorithm == "arpack" else "randomized",
                nComponents=k,
            )
        )
    return out
