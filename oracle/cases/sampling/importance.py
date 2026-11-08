"""Pareto-smoothed importance weights and the Pareto k-hat diagnostic against ArviZ's array functions."""

import importlib.metadata
from typing import Any

import numpy as np
from arviz_stats.base.array import array_stats
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"arviz-stats": importlib.metadata.version("arviz-stats")}

# ArviZ's array functions are loosely annotated; every result is converted to float64 before it is stored.
azs: Any = array_stats

# The generalised Pareto fit is a closed-form empirical-Bayes average of exponentials and logarithms.
RTOL = 1e-10


def psis(log_ratios: FloatArray, r_eff: float) -> tuple[FloatArray, float]:
    """ArviZ's PSIS log weights and k-hat for log ratios (its psislw takes the negated ratios)."""
    lw, k = azs.psislw(-log_ratios, r_eff=r_eff)
    return np.asarray(lw, dtype=np.float64), float(k)


def cases() -> list[Case]:
    g = rng(23)
    out: list[Case] = []
    ratios: dict[str, FloatArray] = {
        "normal-100": g.normal(size=100) * 0.8,
        "normal-400": g.normal(size=400) * 0.6,
        "heavy-300": g.standard_t(1.5, size=300),
        "skewed-60": g.gamma(1.0, size=60) * 2.0,
    }
    for name, r in ratios.items():
        for r_eff in [1.0, 0.6]:
            lw, k = psis(r, r_eff)
            out.append(case(f"psis/weights/{name}/{r_eff}", "psisWeights", [r], lw, rtol=RTOL, atol=1e-12, rEff=r_eff))
            out.append(case(f"psis/k/{name}/{r_eff}", "psisK", [r], np.array(k), rtol=RTOL, atol=1e-12, rEff=r_eff))
    draws: dict[str, FloatArray] = {
        "normal-2x200": g.normal(size=(2, 200)),
        "t2-1x500": g.standard_t(2.0, size=(1, 500)),
        "exp-4x50": g.exponential(size=(4, 50)),
    }
    for name, x in draws.items():
        for tail in ["both", "right", "left"]:
            k = float(azs.pareto_khat(x, tail=tail, r_eff=1.0))
            out.append(
                case(f"paretoKHat/{name}/{tail}", "paretoKHat", [x], np.array(k), rtol=RTOL, tail=tail, rEff=1.0)
            )
        default = float(azs.pareto_khat(x))
        out.append(case(f"paretoKHat/{name}/default", "paretoKHat", [x], np.array(default), rtol=RTOL))
    return out
