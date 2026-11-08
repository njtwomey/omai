"""The pan matrix profile against STUMPY: breadth-first window order and stimp's displayed PAN (partial runs)."""

from importlib.metadata import version

import numpy as np
import stumpy
from omai_oracle.fixture import Case, case
from omai_oracle.timeseries_helpers import as_float, series
from stumpy import core

REFERENCE: dict[str, str] = {"stumpy": version("stumpy"), "numpy": np.__version__}

TOL = {"rtol": 1e-8, "atol": 1e-8}
# Exact repeats (a query taken from the series, copied dimensions, planted motifs) have distances sqrt(2m(1 - rho))
# with rho within rounding of 1, where 1e-14 in rho is 1e-7 in the distance: those cases get atol 1e-6.
SELF = {"rtol": 1e-7, "atol": 1e-6}


def cases() -> list[Case]:
    out: list[Case] = []
    for n in [1, 4, 9, 13]:
        out.append(case(f"bfsIndices/{n}", "bfsIndices", [], as_float(core._bfs_indices(n)), n=n))  # pyright: ignore[reportPrivateUsage]
    pan_t = series(60, 633)
    st = stumpy.stimp(pan_t, 3, 10, percentage=1.0)
    for _ in range(5):
        st.update()
    out.append(case("stimp/partial", "stimp", [pan_t], as_float(st.PAN_), **TOL, minM=3, maxM=10, steps=5))
    raw = st.pan(normalize=True, contrast=False, binary=False, clip=True)
    out.append(
        case(
            "stimp/raw", "stimp", [pan_t], as_float(raw), **TOL, minM=3, maxM=10, steps=5, contrast=False, binary=False
        )
    )
    contrast = st.pan(binary=False)
    out.append(
        case("stimp/contrast", "stimp", [pan_t], as_float(contrast), **TOL, minM=3, maxM=10, steps=5, binary=False)
    )
    return out
