"""NMF by cyclic coordinate descent (HALS) from fixed W and H against scikit-learn's solver='cd', Frobenius loss, for
a fixed number of sweeps."""

from importlib.metadata import version
from typing import Any

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.decomposition import NMF

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

TOL = {"rtol": 1e-8, "atol": 1e-10}


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(9)
    x = g.uniform(0, 1, (12, 3)) @ g.uniform(0, 1, (3, 6)) + 0.01 * g.uniform(0, 1, (12, 6))
    w0 = g.uniform(0.1, 1, (12, 3))
    h0 = g.uniform(0.1, 1, (3, 6))
    for sweeps in [1, 5, 40]:
        options: dict[str, Any] = {
            "n_components": 3,
            "init": "custom",
            "solver": "cd",
            "beta_loss": "frobenius",
            "max_iter": sweeps,
            "tol": 0.0,
            "shuffle": False,
        }
        model = NMF(**options)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = model.fit_transform(x, W=w0.copy(), H=h0.copy())
        inputs = [x, w0, h0]
        out.append(case(f"{sweeps}/h", "h", inputs, np.asarray(model.components_), **TOL, sweeps=sweeps))
        out.append(case(f"{sweeps}/w", "w", inputs, np.asarray(w), **TOL, sweeps=sweeps))
        out.append(
            case(f"{sweeps}/error", "error", inputs, np.float64(model.reconstruction_err_), **TOL, sweeps=sweeps)
        )
    return out
