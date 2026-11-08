"""Posterior summaries and E-BFMI against ArviZ (`summary` on a posterior group, `bfmi` on energies)."""

import importlib.metadata
from typing import Any

import arviz_base as azb
import numpy as np
from arviz_stats.base.array import array_stats
from arviz_stats.summary import summary
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {
    "arviz-stats": importlib.metadata.version("arviz-stats"),
    "arviz-base": importlib.metadata.version("arviz-base"),
}

# ArviZ's array functions are loosely annotated; every result is converted to float64 before it is stored.
azs: Any = array_stats

COLUMNS = ["mean", "sd", "lb", "ub", "ess_bulk", "ess_tail", "r_hat", "mcse_mean", "mcse_sd"]


def summary_row(x: FloatArray, kind: str, prob: float) -> FloatArray:
    """ArviZ's summary of one variable as the columns of omai's SummaryRow, in order."""
    tree = azb.from_dict({"posterior": {"x": x}})
    table: Any = summary(tree, round_to="none", ci_kind=kind, ci_prob=prob)
    perc = int(prob * 100)
    renamed = table.rename(columns={f"{kind}{perc}_lb": "lb", f"{kind}{perc}_ub": "ub"})
    return np.asarray(renamed.loc["x", COLUMNS].to_numpy(dtype=np.float64))


def cases() -> list[Case]:
    g = rng(77)
    out: list[Case] = []
    x = np.cumsum(g.normal(size=(4, 60)), axis=1) * 0.1 + g.normal(size=(4, 60))
    # ArviZ 1.x computes the tail ESS of a summary with its default ci_prob 0.89 (q11 and q89), whatever ci_prob is.
    for kind, prob in [("eti", 0.89), ("hdi", 0.94)]:
        row = summary_row(x, kind, prob)
        out.append(case(f"summary/{kind}", "summary", [x], row, rtol=1e-10, interval=kind, prob=prob, tailProb=0.89))
    energy = np.cumsum(g.normal(size=(3, 50)), axis=1)
    out.append(case("bfmi/random-walk", "bfmi", [energy], azs.bfmi(energy)))
    white = g.normal(size=(2, 40))
    out.append(case("bfmi/white", "bfmi", [white], azs.bfmi(white)))
    return out
