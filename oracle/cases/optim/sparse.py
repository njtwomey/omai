"""Sparse approximation: OMP and the coordinate-descent lasso against sklearn; basis pursuit by scipy linprog."""

import numpy as np
import scipy
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.optim_helpers import sparse_problem
from scipy.optimize import linprog
from sklearn.linear_model import ElasticNet, Lasso, OrthogonalMatchingPursuit

REFERENCE: dict[str, str] = {"sklearn": sklearn.__version__, "scipy": scipy.__version__}


def omp_cases() -> list[Case]:
    d, y = sparse_problem(61, 30, 12, 3, 0.05)
    models: list[tuple[str, OrthogonalMatchingPursuit, dict[str, object]]] = [
        ("nnz", OrthogonalMatchingPursuit(n_nonzero_coefs=3, fit_intercept=False), {"nNonzeroCoefs": 3}),
        ("tol", OrthogonalMatchingPursuit(tol=0.5, fit_intercept=False), {"tol": 0.5}),
        ("default", OrthogonalMatchingPursuit(fit_intercept=False), {}),
    ]
    out: list[Case] = []
    for name, model, params in models:
        coef = np.asarray(model.fit(d, y).coef_, dtype=np.float64)
        out.append(case(f"omp/{name}", "omp", [d, y], coef, rtol=1e-10, atol=1e-12, **params))
    return out


def lasso_cases() -> list[Case]:
    d, y = sparse_problem(63, 25, 10, 3, 0.1)
    m = d.shape[0]
    out: list[Case] = []
    for i, lam in enumerate([0.5, 2.0]):
        model = Lasso(alpha=lam / m, fit_intercept=False, tol=1e-12, max_iter=100_000).fit(d, y)
        out.append(case(f"bpdn/{i}", "bpdn", [d, y], np.asarray(model.coef_), rtol=1e-8, atol=1e-9, lam=lam))
    net = ElasticNet(alpha=1.0 / m, l1_ratio=0.7, fit_intercept=False, tol=1e-12, max_iter=100_000).fit(d, y)
    out.append(
        case("elasticNetCd/0", "elasticNetCd", [d, y], np.asarray(net.coef_), rtol=1e-8, atol=1e-9, lam=0.7, l2=0.3)
    )
    return out


def basis_pursuit_cases() -> list[Case]:
    """min |x|_1 subject to D x = y as the LP min 1.(u + v), D (u - v) = y, u, v >= 0."""
    d, y = sparse_problem(64, 8, 20, 2, 0.0)
    k = d.shape[1]
    res = linprog(np.ones(2 * k), A_eq=np.hstack([d, -d]), b_eq=y, bounds=(0, None), method="highs")
    sol = np.asarray(res.x, dtype=np.float64)
    x = sol[:k] - sol[k:]
    return [case("basisPursuit/0", "basisPursuit", [d, y], x, rtol=1e-6, atol=1e-7)]


def cases() -> list[Case]:
    return omp_cases() + lasso_cases() + basis_pursuit_cases()
