"""Neighbourhood selection against scikit-learn's Lasso (one regression per node on standardised columns) and CLIME
against scipy's linprog (HiGHS) on the same linear programmes."""

from importlib.metadata import version

import numpy as np
import numpy.typing as npt
import scipy
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from scipy.optimize import linprog
from sklearn.linear_model import Lasso

REFERENCE: dict[str, str] = {
    "scikit-learn": version("scikit-learn"),
    "scipy": scipy.__version__,
    "numpy": np.__version__,
}

TOL = {"rtol": 1e-6, "atol": 1e-8}


def sample(seed: int, n: int) -> npt.NDArray[np.float64]:
    """Gaussian rows with a sparse chain precision."""
    g = rng(seed)
    prec = np.array(
        [[2.0, 0.7, 0, 0, 0], [0.7, 2.0, 0.5, 0, 0], [0, 0.5, 2.0, 0.6, 0], [0, 0, 0.6, 2.0, 0.4], [0, 0, 0, 0.4, 2.0]]
    )
    return g.multivariate_normal(np.zeros(5), np.linalg.inv(prec), n)


def node_lassos(x: npt.NDArray[np.float64], penalty: float) -> npt.NDArray[np.float64]:
    """Row a: the lasso coefficients of column a on the others (zero diagonal), columns standardised (ddof 0)."""
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    p = x.shape[1]
    coefs = np.zeros((p, p))
    for a in range(p):
        others = [j for j in range(p) if j != a]
        model = Lasso(alpha=penalty, fit_intercept=False, tol=1e-14, max_iter=1_000_000).fit(z[:, others], z[:, a])
        coefs[a, others] = model.coef_
    return coefs


def clime_columns(s: npt.NDArray[np.float64], lam: float) -> npt.NDArray[np.float64]:
    """Column j (row j of the output) of min |b|_1 s.t. |S b - e_j|_inf <= lam, as an LP in (u, v), b = u - v."""
    p = s.shape[0]
    a_ub = np.block([[s, -s], [-s, s]])
    out = np.zeros((p, p))
    for j in range(p):
        e = np.zeros(p)
        e[j] = 1.0
        res = linprog(
            np.ones(2 * p), A_ub=a_ub, b_ub=np.concatenate([lam + e, lam - e]), bounds=(0, None), method="highs"
        )
        assert res.x is not None
        out[j] = res.x[:p] - res.x[p:]
    return out


def cases() -> list[Case]:
    out: list[Case] = []
    x = sample(37, 120)
    for penalty in [0.05, 0.15]:
        out.append(
            case(f"lasso{penalty}/coefficients", "coefficients", [x], node_lassos(x, penalty), **TOL, penalty=penalty)
        )
    s = np.cov(x, rowvar=False, bias=True)
    for lam in [0.05, 0.2]:
        out.append(case(f"clime{lam}/columns", "climeColumns", [s], clime_columns(s, lam), **TOL, lambda_=lam))
    return out
