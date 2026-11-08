"""Kernel partial least squares with a linear kernel against scikit-learn's PLSRegression (scale=False), whose NIPALS on
the data matrices gives the same predictions as NIPALS on the Gram matrix."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng, uniform
from sklearn.cross_decomposition import PLSRegression

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(515)
    x = uniform(g, (40, 4), -1.0, 1.0)
    xt = uniform(g, (9, 4), -1.0, 1.0)
    y1 = x[:, 0] - 2.0 * x[:, 1] + 0.05 * g.normal(size=40)
    y2 = np.column_stack([y1, x[:, 2] + x[:, 0] + 0.05 * g.normal(size=40)])
    out: list[Case] = []
    for name, y in [("single", y1), ("multi", y2)]:
        for k in (1, 2, 3):
            model = PLSRegression(n_components=k, scale=False, tol=1e-13, max_iter=20000).fit(x, y)
            out.append(
                case(
                    f"{name}/{k}",
                    "kernelPls",
                    [x, y, xt],
                    {"predict": model.predict(xt)},
                    # sklearn's power method for several targets stops at its own tolerance: the iterative tier.
                    rtol=1e-6,
                    atol=5e-7,
                    nComponents=k,
                )
            )
    return out
