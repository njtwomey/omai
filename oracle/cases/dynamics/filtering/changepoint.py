"""Offline changepoint detection against ruptures: PELT and binary segmentation with the l2 and normal costs
(jump = 1 so every index is a candidate)."""

import numpy as np
import ruptures as rpt
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"ruptures": rpt.__version__}


def series(g: np.random.Generator) -> FloatArray:
    return np.concatenate([g.normal(0, 1, 40), g.normal(3, 1, 30), g.normal(1, 0.3, 30), g.normal(1, 2.5, 40)])


def cases() -> list[Case]:
    g = rng(26_5)
    y = series(g)
    out: list[Case] = []
    for cost in ["l2", "normal"]:
        for pen in [5.0, 20.0]:
            bk = rpt.Pelt(model=cost, min_size=2, jump=1).fit(y).predict(pen=pen)
            out.append(
                case(
                    f"pelt/{cost}/{pen:g}",
                    "pelt",
                    [y],
                    np.array(bk, dtype=np.float64),
                    rtol=0,
                    atol=0,
                    cost=cost,
                    pen=pen,
                )
            )
        bs = rpt.Binseg(model=cost, min_size=2, jump=1).fit(y).predict(n_bkps=3)
        out.append(
            case(f"binseg/{cost}/3", "binseg_n", [y], np.array(bs, dtype=np.float64), rtol=0, atol=0, cost=cost, n=3)
        )
        bp = rpt.Binseg(model=cost, min_size=2, jump=1).fit(y).predict(pen=10.0)
        out.append(
            case(
                f"binseg/{cost}/pen",
                "binseg_pen",
                [y],
                np.array(bp, dtype=np.float64),
                rtol=0,
                atol=0,
                cost=cost,
                pen=10.0,
            )
        )
    return out
