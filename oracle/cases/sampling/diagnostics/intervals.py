"""Highest-density and equal-tailed intervals against ArviZ's array functions."""

import importlib.metadata
from typing import Any

import numpy as np
from arviz_stats.base.array import array_stats
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"arviz-stats": importlib.metadata.version("arviz-stats")}

# ArviZ's array functions are loosely annotated; every result is converted to float64 before it is stored.
azs: Any = array_stats


def cases() -> list[Case]:
    g = rng(5)
    out: list[Case] = []
    samples = {
        "normal-200": g.normal(size=200),
        "gamma-151": g.gamma(2.0, size=151),
        "chains-3x40": g.normal(size=(3, 40)),
    }
    for name, x in samples.items():
        for prob in [0.5, 0.89, 0.94, 1.0]:
            flat = x.ravel()
            out.append(case(f"hdi/{name}/{prob}", "hdi", [x], azs.hdi(flat, prob), prob=prob))
            out.append(case(f"eti/{name}/{prob}", "eti", [x], azs.eti(flat, prob), prob=prob))
    # Circular draws concentrated around +-pi, where the interval wraps.
    angles = np.mod(g.vonmises(np.pi - 0.2, 4.0, size=120) + np.pi, 2 * np.pi) - np.pi
    for prob in [0.5, 0.9]:
        out.append(
            case(
                f"hdi/circular/{prob}", "hdi", [angles], azs.hdi(angles, prob, circular=True), prob=prob, circular=True
            )
        )
    # Discrete bimodal draws: the multimodal HDI of ArviZ on integer data (padded with NaN to max_modes).
    bimodal = np.concatenate([g.poisson(2.0, size=60), 12 + g.poisson(1.0, size=40)]).astype(np.int64)
    for prob in [0.5, 0.8, 0.95]:
        modes = azs.hdi(bimodal, prob, method="multimodal", max_modes=4)
        kept = np.asarray([row for row in np.asarray(modes, dtype=np.float64) if not np.isnan(row[0])])
        out.append(case(f"hdiModes/bimodal/{prob}", "hdiModes", [bimodal], kept, prob=prob, maxModes=4))
    return out
