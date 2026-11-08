"""R-hat, effective sample sizes and Monte Carlo standard errors against ArviZ's array functions on fixed chains."""

import importlib.metadata
from typing import Any

import numpy as np
from arviz_stats.base.array import array_stats
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"arviz-stats": importlib.metadata.version("arviz-stats")}

# ArviZ's array functions are loosely annotated; every result is converted to float64 before it is stored.
azs: Any = array_stats

# Rank normalisation goes through the normal quantile and the autocorrelations through an FFT, so agreement is to
# rounding of those (special-function tier).
RTOL = 1e-10
ATOL = 1e-12

RHAT_METHODS: dict[str, str] = {
    "rank": "rank",
    "split": "split",
    "folded": "folded",
    "zScale": "z_scale",
    "identity": "identity",
}
ESS_METHODS: dict[str, str] = {
    "bulk": "bulk",
    "mean": "mean",
    "sd": "sd",
    "median": "median",
    "mad": "mad",
    "folded": "folded",
    "zScale": "z_scale",
    "identity": "identity",
}


def ar1(g: np.random.Generator, chains: int, draws: int, phi: float, offsets: list[float] | None = None) -> FloatArray:
    """Stationary AR(1) chains x_t = phi x_{t-1} + sqrt(1 - phi^2) e_t, optionally shifted per chain."""
    x = np.zeros((chains, draws))
    x[:, 0] = g.normal(size=chains)
    for t in range(1, draws):
        x[:, t] = phi * x[:, t - 1] + np.sqrt(1 - phi * phi) * g.normal(size=chains)
    if offsets is not None:
        x += np.asarray(offsets)[:, None]
    return x


def chain_sets() -> dict[str, FloatArray]:
    g = rng(2024)
    return {
        "ar1-4x50": ar1(g, 4, 50, 0.6),
        "ar1-2x37-odd": ar1(g, 2, 37, 0.9),
        "anticorrelated-3x40": ar1(g, 3, 40, -0.5),
        "offset-2x30": ar1(g, 2, 30, 0.3, [0.0, 1.5]),
        "student-t-4x25": g.standard_t(2.0, size=(4, 25)),
        "ties-3x20": g.poisson(2.0, size=(3, 20)).astype(np.float64),
        "short-2x8": ar1(g, 2, 8, 0.2),
        "one-chain-1x30": ar1(g, 1, 30, 0.5),
    }


def cases() -> list[Case]:
    out: list[Case] = []
    for name, x in chain_sets().items():
        for ours, theirs in RHAT_METHODS.items():
            value = float(azs.rhat(x, method=theirs))
            out.append(case(f"rhat/{ours}/{name}", "rhat", [x], np.array(value), rtol=RTOL, atol=ATOL, method=ours))
        for ours, theirs in ESS_METHODS.items():
            value = float(azs.ess(x, method=theirs))
            out.append(case(f"ess/{ours}/{name}", "ess", [x], np.array(value), rtol=RTOL, atol=ATOL, method=ours))
        rel = float(azs.ess(x, method="bulk", relative=True))
        out.append(case(f"ess/bulk-relative/{name}", "ess", [x], np.array(rel), rtol=RTOL, atol=ATOL, relative=True))
        tail = float(azs.ess(x, method="tail", prob=0.05))
        out.append(case(f"ess/tail/{name}", "ess", [x], np.array(tail), rtol=RTOL, atol=ATOL, method="tail"))
        tail10 = float(azs.ess(x, method="tail", prob=0.1))
        out.append(
            case(f"ess/tail-0.1/{name}", "ess", [x], np.array(tail10), rtol=RTOL, atol=ATOL, method="tail", prob=0.1)
        )
        q = float(azs.ess(x, method="quantile", prob=0.3))
        out.append(
            case(f"ess/quantile-0.3/{name}", "ess", [x], np.array(q), rtol=RTOL, atol=ATOL, method="quantile", prob=0.3)
        )
        loc = float(azs.ess(x, method="local", prob=(0.2, 0.7)))
        out.append(
            case(f"ess/local/{name}", "ess", [x], np.array(loc), rtol=RTOL, atol=ATOL, method="local", prob=[0.2, 0.7])
        )
        for method in ["mean", "sd", "median"]:
            value = float(azs.mcse(x, method=method))
            out.append(case(f"mcse/{method}/{name}", "mcse", [x], np.array(value), rtol=RTOL, atol=ATOL, method=method))
        mq = float(azs.mcse(x, method="quantile", prob=0.8))
        out.append(
            case(
                f"mcse/quantile-0.8/{name}",
                "mcse",
                [x],
                np.array(mq),
                rtol=RTOL,
                atol=ATOL,
                method="quantile",
                prob=0.8,
            )
        )
    # Edges: a constant chain (ESS is the number of draws), a NaN draw (NaN), too few draws (NaN).
    constant = np.full((2, 10), 3.0)
    out.append(case("ess/constant", "ess", [constant], np.array(float(azs.ess(constant))), method="bulk"))
    with_nan = ar1(rng(7), 2, 10, 0.5)
    with_nan[0, 3] = np.nan
    out.append(case("rhat/nan", "rhat", [with_nan], np.array(float(azs.rhat(with_nan))), method="rank"))
    out.append(case("ess/nan", "ess", [with_nan], np.array(float(azs.ess(with_nan))), method="bulk"))
    tiny = ar1(rng(8), 2, 3, 0.5)
    out.append(case("rhat/too-short", "rhat", [tiny], np.array(float(azs.rhat(tiny))), method="rank"))
    return out
