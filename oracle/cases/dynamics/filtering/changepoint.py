"""Offline changepoint detection against ruptures: PELT and binary segmentation with the l2 and normal costs
(jump = 1 so every index is a candidate).

PELT's reference is the exact penalised optimum, from ruptures' `Dynp` (exact for each number of breakpoints)
minimised over the count. ruptures' own `Pelt` prunes unsafely when min_size > 1 and can miss it (on the l2 case with
penalty 5 it returns 16 segments costing 248.88; the optimum has 15 costing 248.53)."""

import numpy as np
import ruptures as rpt
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"ruptures": rpt.__version__}


def series(g: np.random.Generator) -> FloatArray:
    return np.concatenate([g.normal(0, 1, 40), g.normal(3, 1, 30), g.normal(1, 0.3, 30), g.normal(1, 2.5, 40)])


def penalised_optimum(y: FloatArray, model: str, pen: float, min_size: int) -> list[int]:
    """The partition minimising the sum of segment costs plus `pen` per segment."""
    algo = rpt.Dynp(model=model, min_size=min_size, jump=1).fit(y)
    n = len(y)
    best: tuple[float, list[int]] = (algo.cost.sum_of_costs([n]) + pen, [n])
    for k in range(1, n // min_size):
        bk: list[int] = algo.predict(n_bkps=k)
        value = algo.cost.sum_of_costs(bk) + pen * (k + 1)
        if value < best[0] - 1e-9:
            best = (value, bk)
    return best[1]


def cases() -> list[Case]:
    g = rng(26_5)
    y = series(g)
    out: list[Case] = []
    for cost in ["l2", "normal"]:
        for pen in [5.0, 20.0]:
            bk = penalised_optimum(y, cost, pen, 2)
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
