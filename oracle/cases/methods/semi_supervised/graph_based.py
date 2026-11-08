"""Label propagation and label spreading against scikit-learn: the label distributions after fitting, with rbf and
k-nearest-neighbour kernels."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def data() -> tuple[FloatArray, FloatArray]:
    g = rng(17)
    a = g.normal(size=(20, 2)) * 0.5
    b = g.normal(size=(20, 2)) * 0.5 + np.array([2.0, 1.0])
    x = np.concatenate([a, b])
    y = np.full(40, -1.0)
    y[[0, 1, 20, 21]] = [0.0, 0.0, 1.0, 1.0]
    return x, y


def cases() -> list[Case]:
    x, y = data()
    out: list[Case] = []
    for kernel in ["rbf", "knn"]:
        lp = LabelPropagation(kernel=kernel, gamma=2, n_neighbors=5, max_iter=1000, tol=1e-3).fit(x, y)
        out.append(
            case(
                f"propagation/{kernel}",
                "graph",
                [x, y],
                {"distributions": np.asarray(lp.label_distributions_)},
                rtol=1e-9,
                atol=1e-12,
                variant="propagation",
                kernel=kernel,
                gamma=2.0,
                n_neighbors=5,
                alpha=0.2,
                max_iter=1000,
            )
        )
        ls = LabelSpreading(kernel=kernel, gamma=2, n_neighbors=5, alpha=0.3, max_iter=30, tol=1e-3).fit(x, y)
        out.append(
            case(
                f"spreading/{kernel}",
                "graph",
                [x, y],
                {"distributions": np.asarray(ls.label_distributions_)},
                rtol=1e-9,
                atol=1e-12,
                variant="spreading",
                kernel=kernel,
                gamma=2.0,
                n_neighbors=5,
                alpha=0.3,
                max_iter=30,
            )
        )
    return out
