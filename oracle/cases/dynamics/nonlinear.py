"""Time-series measures of chaos against nolds: correlation dimension (corr_dim) and Rosenstein's largest Lyapunov
exponent (lyap_r), both with fit='poly' (nolds' default RANSAC fit is randomised).

Series: the logistic map at r = 4 and the x coordinate of the Henon map, generated here and passed as inputs.
"""

import nolds
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"nolds": str(getattr(nolds, "__version__", "0.6"))}


def logistic(n: int) -> FloatArray:
    x = np.empty(n)
    v = 0.3
    for i in range(n):
        v = 4 * v * (1 - v)
        x[i] = v
    return x


def henon(n: int) -> FloatArray:
    x = np.empty(n)
    a, b = 0.1, 0.1
    for _ in range(100):
        a, b = 1 - 1.4 * a * a + b, 0.3 * a
    for i in range(n):
        a, b = 1 - 1.4 * a * a + b, 0.3 * a
        x[i] = a
    return x


def cases() -> list[Case]:
    out: list[Case] = []
    for label, series in {"logistic": logistic(300), "henon": henon(300)}.items():
        for emb in [2, 3]:
            d = nolds.corr_dim(series, emb, fit="poly")
            out.append(
                case(f"corr_dim/{label}/{emb}", "corr_dim", [series], np.array(d), rtol=1e-9, atol=1e-10, emb=emb)
            )
        le = nolds.lyap_r(series, emb_dim=3, lag=1, min_tsep=5, trajectory_len=6, fit="poly")
        out.append(case(f"lyap_r/{label}", "lyap_r", [series], np.array(le), rtol=1e-9, atol=1e-10))
    return out
