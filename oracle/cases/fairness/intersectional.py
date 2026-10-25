"""Generalised entropy and Theil indices against AIF360's ClassificationMetric, and rich-subgroup auditing against
GerryFair's Auditor (AIF360)."""

import aif360
import numpy as np
from aif360.algorithms.inprocessing.gerryfair.auditor import Auditor
from aif360.metrics import ClassificationMetric
from whitebox_oracle.fairness import binary_dataset, with_labels
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"aif360": aif360.__version__}


def audit(x: FloatArray, y: FloatArray, p: FloatArray, rate: str) -> tuple[FloatArray, float]:
    """GerryFair's audit of predictions: the subgroup's membership and its weighted disparity."""
    data = binary_dataset({"a": x[:, 0], "b": x[:, 1], "c": x[:, 2]}, y, ["a", "b"])
    members, disparity = Auditor(data, rate).audit(p)
    return np.asarray(members, dtype=np.float64), float(disparity)


def cases() -> list[Case]:
    g = rng(91)
    n = 120
    s = (g.uniform(size=n) < 0.4).astype(np.float64)
    y = (g.uniform(size=n) < 0.45 + 0.1 * s).astype(np.float64)
    p = np.where(g.uniform(size=n) < 0.7, y, 1 - y)
    p = np.where((s == 1) & (g.uniform(size=n) < 0.25), 1.0, p)
    data = binary_dataset({"s": s}, y, ["s"])
    metric = ClassificationMetric(data, with_labels(data, p), [{"s": 1.0}], [{"s": 0.0}])
    out: list[Case] = []
    for alpha in [0.0, 0.5, 1.0, 2.0, 3.0]:
        if alpha == 0.0:
            continue  # AIF360 divides by alpha (alpha - 1) without the limits; alpha 1 is the Theil index below.
        if alpha == 1.0:
            total = float(metric.theil_index())
            between = float(metric.between_group_theil_index())
        else:
            total = float(metric.generalized_entropy_index(alpha=alpha))  # pyright: ignore[reportArgumentType]
            between = float(metric.between_group_generalized_entropy_index(alpha=alpha))  # pyright: ignore[reportArgumentType]
        out += [
            case(f"generalisedEntropy/alpha={alpha}", "entropyTotal", [y, p, s], np.asarray(total), alpha=alpha),
            case(f"betweenGroupEntropy/alpha={alpha}", "entropyBetween", [y, p, s], np.asarray(between), alpha=alpha),
        ]
    # Rich subgroups over two protected features, with the unfairness in their interaction.
    x = np.column_stack([g.normal(size=n), g.normal(size=n), g.normal(size=n)])
    flagged = (x[:, 0] + 0.5 * x[:, 1] > 0.6).astype(np.float64)
    yy = (g.uniform(size=n) < 0.5).astype(np.float64)
    pp = np.where(g.uniform(size=n) < 0.2 + 0.5 * flagged, 1.0, 0.0)
    for rate in ["FP", "FN"]:
        members, disparity = audit(x, yy, pp, rate)
        name = "falsePositive" if rate == "FP" else "falseNegative"
        out += [
            case(f"richSubgroup/{rate}/members", "auditMembers", [x[:, :2], yy, pp], members, rate=name),
            case(
                f"richSubgroup/{rate}/disparity", "auditDisparity", [x[:, :2], yy, pp], np.asarray(disparity), rate=name
            ),
        ]
    return out
