"""Multidimensional matrix profiles against STUMPY: mstump (with include and discords),
multi_distance_profile, subspace and mdl."""

from importlib.metadata import version
from typing import Any

import numpy as np
import stumpy
from omai_oracle.fixture import Case, case
from omai_oracle.timeseries_helpers import as_float, series
from stumpy.mstump import multi_distance_profile

REFERENCE: dict[str, str] = {"stumpy": version("stumpy"), "numpy": np.__version__}

TOL = {"rtol": 1e-8, "atol": 1e-8}
# Exact repeats (a query taken from the series, copied dimensions, planted motifs) have distances sqrt(2m(1 - rho))
# with rho within rounding of 1, where 1e-14 in rho is 1e-7 in the distance: those cases get atol 1e-6.
SELF = {"rtol": 1e-7, "atol": 1e-6}


def cases() -> list[Case]:
    out: list[Case] = []
    tm = np.stack([series(80, 622), series(80, 623), series(80, 624)])
    tm[1, 30:42] = tm[0, 30:42]
    tm[1, 60:72] = tm[0, 30:42] + 0.01
    for name, kw in [("mstump", {}), ("mstump/include", {"include": [2]}), ("mstump/discords", {"discords": True})]:
        P2, I2 = stumpy.mstump(tm, 6, **kw)
        params: dict[str, Any] = {"m": 6}
        if "include" in kw:
            params["include"] = kw["include"]
        if "discords" in kw:
            params["discords"] = True
        out.append(case(name, "mstump", [tm], np.concatenate([as_float(P2), as_float(I2)]), **SELF, **params))
    D = multi_distance_profile(20, tm, 6)
    out.append(case("multiDistanceProfile", "multiDistanceProfile", [tm], as_float(D), **TOL, m=6, idx=20))
    for k in range(3):
        S = stumpy.subspace(tm, 12, 30, 60, k)
        out.append(case(f"subspace/{k}", "subspace", [tm], as_float(S), m=12, subseqIdx=30, nnIdx=60, k=k))
    bits, subs = stumpy.mdl(tm, 12, np.array([30, 30, 30]), np.array([60, 60, 60]))
    out.append(
        case(
            "mdl",
            "mdl",
            [tm],
            np.concatenate([bits, [float(v) for s_ in subs for v in s_]]),
            **TOL,
            m=12,
            subseqIdx=[30, 30, 30],
            nnIdx=[60, 60, 60],
        )
    )
    return out
