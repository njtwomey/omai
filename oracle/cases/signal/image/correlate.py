"""ndimage-style correlation and convolution against scipy.ndimage (every border mode, origins, even kernels)."""

from typing import Any, Literal

import numpy as np
import scipy
import scipy.ndimage as N
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

MODES: list[Literal["reflect", "mirror", "nearest", "wrap", "constant"]] = [
    "reflect",
    "mirror",
    "nearest",
    "wrap",
    "constant",
]
TOL = {"rtol": 1e-12, "atol": 1e-12}


def cases() -> list[Case]:
    g = rng(404)
    out: list[Case] = []
    x = g.normal(size=9)
    for mode in MODES:
        for w in [np.array([0.2, 0.5, 0.3]), np.array([1.0, -2.0, 0.5, 0.25])]:
            k = len(w)
            for origin in sorted({0, -(k // 2), (k - 1) // 2}):
                params: dict[str, Any] = {"mode": mode, "cval": 0.7, "origin": origin}
                c = N.correlate1d(x, w, mode=mode, cval=0.7, origin=origin)
                out.append(case(f"correlate1d/{mode}/{k}/{origin}", "correlate1d", [x, w], c, **TOL, **params))
                v = N.convolve1d(x, w, mode=mode, cval=0.7, origin=origin)
                out.append(case(f"convolve1d/{mode}/{k}/{origin}", "convolve1d", [x, w], v, **TOL, **params))
    img = g.normal(size=(6, 7))
    w2 = g.normal(size=(3, 2))
    for mode in MODES:
        c = N.correlate(img, w2, mode=mode, cval=0.3)
        out.append(case(f"correlateNd/{mode}", "correlateNd", [img, w2], c, **TOL, mode=mode, cval=0.3))
        v = N.convolve(img, w2, mode=mode, cval=0.3)
        out.append(case(f"convolveNd/{mode}", "convolveNd", [img, w2], v, **TOL, mode=mode, cval=0.3))
    shifted: dict[str, Any] = {"origin": [1, -1]}
    c = N.correlate(img, w2, **shifted)
    out.append(case("correlateNd/origin", "correlateNd", [img, w2], c, **TOL, mode="reflect", cval=0.0, origin=[1, -1]))
    m = g.normal(size=(5, 6))
    c = N.correlate1d(m, [1.0, 2.0, 1.0], axis=0)
    out.append(case("correlate1d/axis0", "correlate1d", [m, np.array([1.0, 2.0, 1.0])], c, **TOL, axis=0))
    return out
