"""Ranks and tie corrections against scipy.stats.rankdata and scipy.stats.tiecorrect."""

from typing import Literal

import numpy as np
import scipy
import scipy.stats as st
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(77)
    x = np.round(g.normal(size=30), 1)
    out: list[Case] = []
    methods: list[Literal["average", "min", "max", "dense", "ordinal"]] = ["average", "min", "max", "dense", "ordinal"]
    for method in methods:
        out.append(
            case(f"rankdata/{method}", "rankdata", [x], np.asarray(st.rankdata(x, method=method), float), method=method)
        )
    # Tie groups are the same in min ranks, which scipy-stubs types as integers.
    corrected = st.tiecorrect(st.rankdata(x, method="min").astype(np.int64))
    out.append(case("tieCorrection", "tieCorrection", [x], np.asarray(corrected)))
    return out
