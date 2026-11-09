"""GraphicalLassoCV against scikit-learn on a fixed contiguous k-fold split (tolerances tightened).

Each case fits `GraphicalLassoCV` with `cv=KFold(k)` (no shuffling, so the splits are the engine's `kFold(k)`), once
on an explicit penalty grid (one refinement) and once on a grid by count with refinements around the best penalty.
The outputs are the selected penalty, the penalties of the whole path (descending, refinements merged), the held-out
scores of every penalty on every fold (with the unpenalised row last) and the refitted precision.
"""

from importlib.metadata import version
from typing import cast

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.covariance import GraphicalLassoCV
from sklearn.model_selection import KFold

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

TOL = {"rtol": 1e-6, "atol": 1e-8}


def sample(seed: int, p: int, n: int) -> np.ndarray:
    """Gaussian rows whose precision is a sparse chain."""
    g = rng(seed)
    prec = 2.0 * np.eye(p)
    for i in range(p - 1):
        prec[i, i + 1] = prec[i + 1, i] = 0.6 - 0.05 * i
    return g.multivariate_normal(np.zeros(p), np.linalg.inv(prec), n)


def cases() -> list[Case]:
    out: list[Case] = []
    settings: list[tuple[str, int, int, int, int | list[float], int]] = [
        ("grid", 23, 4, 60, [0.4, 0.2, 0.1, 0.05, 0.02], 3),
        ("refined", 29, 5, 80, 4, 4),
    ]
    for name, seed, p, n, alphas, folds in settings:
        x = sample(seed, p, n)
        model = GraphicalLassoCV(
            alphas=cast(int, alphas), n_refinements=3, cv=KFold(folds), tol=1e-10, enet_tol=1e-12, max_iter=1000
        ).fit(x)
        res = model.cv_results_
        path = np.asarray(res["alphas"], dtype=np.float64)
        scores = np.column_stack([np.asarray(res[f"split{k}_test_score"]) for k in range(folds)])
        for fn, output in [
            ("alpha", np.asarray([model.alpha_])),
            ("alphas", path[:-1]),
            ("scores", scores),
            ("precision", np.asarray(model.precision_)),
        ]:
            out.append(case(f"{name}/{fn}", fn, [x], output, **TOL, folds=folds, alphas=alphas, nRefinements=3))
    return out
