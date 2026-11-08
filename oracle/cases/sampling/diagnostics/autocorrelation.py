"""Autocovariance and autocorrelation by FFT against ArviZ's array functions."""

import importlib.metadata
from typing import Any

import numpy as np
from arviz_stats.base.array import array_stats
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"arviz-stats": importlib.metadata.version("arviz-stats")}

# ArviZ's array functions are loosely annotated; every result is converted to float64 before it is stored.
azs: Any = array_stats

# FFT round-off: relative agreement at the large lags, absolute where the autocovariance is near zero.
RTOL = 1e-10
ATOL = 1e-12


def walk(g: np.random.Generator, shape: tuple[int, ...]) -> FloatArray:
    """Correlated draws: cumulative sums of normals, so the autocorrelation decays slowly."""
    return np.cumsum(g.normal(size=shape), axis=-1)


def cases() -> list[Case]:
    g = rng(31)
    inputs: dict[str, FloatArray] = {
        "1d-n17": walk(g, (17,)),
        "1d-n64": g.normal(size=64),
        "2d-3x21": walk(g, (3, 21)),
        "2d-2x5": g.normal(size=(2, 5)),
    }
    out: list[Case] = []
    for name, x in inputs.items():
        out.append(case(f"autocovariance/{name}", "autocovariance", [x], azs.autocov(x), rtol=RTOL, atol=ATOL))
        out.append(case(f"autocorrelation/{name}", "autocorrelation", [x], azs.autocorr(x), rtol=RTOL, atol=ATOL))
    return out
