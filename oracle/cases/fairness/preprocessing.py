"""Reweighing against AIF360's Reweighing.

No golden for disparate-impact repair: AIF360's DisparateImpactRemover runs BlackBoxAuditing's combinatorial repairer,
which bins each column, moves values to other values present in the data and repairs partially by moving a share of
rows between bins; `repairFeatures` is Feldman et al.'s geometric repair (each value towards the median of the groups'
quantiles at its rank, by the repair level), so the two agree in spirit but not value by value.
"""

import aif360
import numpy as np
from aif360.algorithms.preprocessing import Reweighing
from mdml_oracle.fairness import binary_dataset
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"aif360": aif360.__version__}


def cases() -> list[Case]:
    g = rng(53)
    n = 80
    s = (g.uniform(size=n) < 0.4).astype(np.float64)
    y = (g.uniform(size=n) < 0.6 - 0.3 * s).astype(np.float64)
    data = binary_dataset({"s": s}, y, ["s"])
    weights = Reweighing([{"s": 1.0}], [{"s": 0.0}]).fit_transform(data).instance_weights
    out = [case("reweighing", "reweighing", [y, s], np.asarray(weights, dtype=np.float64))]
    return out
