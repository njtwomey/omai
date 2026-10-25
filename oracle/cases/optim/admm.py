"""ADMM on the lasso (cached-Cholesky and proximal forms) against sklearn's coordinate descent."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.optim_helpers import sparse_problem
from sklearn.linear_model import Lasso

REFERENCE: dict[str, str] = {"sklearn": sklearn.__version__}


def cases() -> list[Case]:
    """min |A x - b|^2 / 2 + lam |x|_1 with sklearn's alpha = lam / m and no intercept."""
    a, b = sparse_problem(44, 40, 12, 4, 0.1)
    m = a.shape[0]
    out: list[Case] = []
    for i, lam in enumerate([0.3, 4.0]):
        model = Lasso(alpha=lam / m, fit_intercept=False, tol=1e-14, max_iter=1_000_000).fit(a, b)
        out.append(case(f"lasso/{i}", "lasso", [a, b], np.asarray(model.coef_), rtol=1e-6, atol=1e-7, lam=lam))
    return out
