"""Exact, entropic and unbalanced optimal transport against POT."""

import numpy as np
import ot
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.optim_helpers import clouds, histogram

REFERENCE: dict[str, str] = {"pot": ot.__version__}


def as_array(x: object) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def exact_cases() -> list[Case]:
    out: list[Case] = []
    for i, (n, m) in enumerate([(5, 7), (8, 8)]):
        g = rng(70 + i)
        x, y = clouds(70 + i, n, m, 2)
        a, b = histogram(g, n), histogram(g, m)
        cost = as_array(ot.dist(x, y))
        plan = as_array(ot.emd(a, b, cost))
        out.append(case(f"emd/{i}", "emd", [a, b, cost], {"plan": plan, "cost": float(np.sum(plan * cost))}))
    # Uniform weights on equal sizes: a permutation (degenerate pivots throughout).
    x, y = clouds(72, 6, 6, 2)
    u = np.full(6, 1 / 6)
    cost = as_array(ot.dist(x, y))
    plan = as_array(ot.emd(u, u, cost))
    out.append(case("emd/permutation", "emd", [u, u, cost], {"plan": plan, "cost": float(np.sum(plan * cost))}))
    return [c.model_copy(update={"rtol": 1e-10, "atol": 1e-12}) for c in out]


def entropic_cases() -> list[Case]:
    out: list[Case] = []
    g = rng(73)
    x, y = clouds(73, 6, 5, 2)
    a, b = histogram(g, 6), histogram(g, 5)
    cost = as_array(ot.dist(x, y))
    cost /= cost.max()
    for reg in [0.05, 0.5]:
        plan = as_array(ot.sinkhorn(a, b, cost, reg, method="sinkhorn_log", stopThr=1e-14, numItermax=200000))
        out.append(case(f"sinkhorn/{reg}", "sinkhorn", [a, b, cost], plan, rtol=1e-8, atol=1e-11, reg=reg))
    plan = as_array(ot.sinkhorn(a, b, cost, 0.5, method="sinkhorn", stopThr=1e-14, numItermax=200000))
    out.append(
        case("sinkhorn/standard", "sinkhorn", [a, b, cost], plan, rtol=1e-8, atol=1e-11, reg=0.5, method="standard")
    )
    # Unbalanced: different masses, KL relaxation.
    a2 = a * 2.0
    for reg, reg_m in [(0.1, 1.0), (0.5, 0.3)]:
        plan = as_array(
            ot.unbalanced.sinkhorn_unbalanced(
                a2, b, cost, reg, reg_m, method="sinkhorn", stopThr=1e-14, numItermax=200000
            )
        )
        out.append(
            case(
                f"unbalanced/{reg}-{reg_m}",
                "unbalanced",
                [a2, b, cost],
                plan,
                rtol=1e-7,
                atol=1e-11,
                reg=reg,
                regM=reg_m,
            )
        )
    return out


def cases() -> list[Case]:
    return exact_cases() + entropic_cases()
