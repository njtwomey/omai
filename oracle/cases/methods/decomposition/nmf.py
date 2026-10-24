"""NMF by multiplicative updates from fixed W and H against scikit-learn (Frobenius and Kullback-Leibler)."""

from importlib.metadata import version
from typing import Any

import numpy as np
from sklearn.decomposition import NMF
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

TOL = {"rtol": 1e-8, "atol": 1e-10}


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(9)
    x = g.uniform(0, 1, (12, 3)) @ g.uniform(0, 1, (3, 6)) + 0.01 * g.uniform(0, 1, (12, 6))
    w0 = g.uniform(0.1, 1, (12, 3))
    h0 = g.uniform(0.1, 1, (3, 6))
    for loss in ["frobenius", "kullback-leibler"]:
        options: dict[str, Any] = {
            "n_components": 3,
            "init": "custom",
            "solver": "mu",
            "beta_loss": loss,
            "max_iter": 60,
            "tol": 0.0,
        }
        model = NMF(**options)
        w = model.fit_transform(x, W=w0.copy(), H=h0.copy())
        inputs = [x, w0, h0]
        out.append(case(f"{loss}/h", "h", inputs, np.asarray(model.components_), **TOL, loss=loss))
        out.append(case(f"{loss}/w", "w", inputs, np.asarray(w), **TOL, loss=loss))
        out.append(case(f"{loss}/error", "error", inputs, np.float64(model.reconstruction_err_), **TOL, loss=loss))
    return out
