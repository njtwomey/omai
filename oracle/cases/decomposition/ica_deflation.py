"""Deflationary FastICA with a fixed initial unmixing matrix against scikit-learn: components and sources."""

from importlib.metadata import version
from typing import Any

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.decomposition import FastICA

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

TOL = {"rtol": 1e-7, "atol": 1e-9}


def data() -> FloatArray:
    t = np.linspace(0, 8, 120)
    s = np.c_[np.sin(2 * t), np.sign(np.sin(3 * t)), (t % 1.3) - 0.65]
    return s @ np.array([[1.0, 0.5, 0.2], [0.4, 1.0, 0.3], [0.2, 0.6, 1.0]]).T


def cases() -> list[Case]:
    out: list[Case] = []
    x = data()
    for fun in ["logcosh", "exp", "cube"]:
        w0 = rng(4).standard_normal((3, 3))
        options: dict[str, Any] = {
            "n_components": 3,
            "algorithm": "deflation",
            "fun": fun,
            "w_init": w0,
            "whiten": "unit-variance",
            "max_iter": 400,
        }
        ica = FastICA(**options).fit(x)
        inputs = [x, w0]
        out.append(case(f"{fun}/components", "components", inputs, np.asarray(ica.components_), **TOL, fun=fun))
        out.append(case(f"{fun}/sources", "sources", inputs, np.asarray(ica.transform(x)), **TOL, fun=fun))
    return out
