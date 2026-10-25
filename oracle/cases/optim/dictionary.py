"""Sparse coding over a dictionary against sklearn.decomposition.sparse_encode."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.decomposition import sparse_encode

REFERENCE: dict[str, str] = {"sklearn": sklearn.__version__}


def encode_cases() -> list[Case]:
    g = rng(62)
    dictionary = g.normal(size=(10, 8))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    x = g.normal(size=(5, 8))
    out: list[Case] = []
    for algorithm, params in [
        ("omp", {"n_nonzero_coefs": 2}),
        ("lasso_cd", {"alpha": 0.3}),
        ("threshold", {"alpha": 0.5}),
    ]:
        code = sparse_encode(x, dictionary, algorithm=algorithm, **params)  # pyright: ignore[reportArgumentType]
        ts = {"algorithm": {"omp": "omp", "lasso_cd": "lassoCd", "threshold": "threshold"}[algorithm]}
        ts |= (
            {"nNonzeroCoefs": params["n_nonzero_coefs"]} if "n_nonzero_coefs" in params else {"alpha": params["alpha"]}
        )
        # sparse_encode's lasso stops at sklearn's fixed duality-gap tol (1e-4 |x|^2), so its codes carry that error.
        tol = 1e-4 if algorithm == "lasso_cd" else 1e-9
        out.append(
            case(
                f"sparseEncode/{algorithm}", "sparseEncode", [x, dictionary], np.asarray(code), rtol=tol, atol=tol, **ts
            )
        )
    return out


def cases() -> list[Case]:
    return encode_cases()
