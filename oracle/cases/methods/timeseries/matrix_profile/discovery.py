"""Discovery on matrix profiles against STUMPY: match, motifs and time-series chains (atsc, allc)."""

from importlib.metadata import version

import numpy as np
import stumpy
from mdml_oracle.fixture import Case, case
from mdml_oracle.timeseries_helpers import as_float, planted, series

REFERENCE: dict[str, str] = {"stumpy": version("stumpy"), "numpy": np.__version__}

TOL = {"rtol": 1e-8, "atol": 1e-8}
# Exact repeats (a query taken from the series, copied dimensions, planted motifs) have distances sqrt(2m(1 - rho))
# with rho within rounding of 1, where 1e-14 in rho is 1e-7 in the distance: those cases get atol 1e-6.
SELF = {"rtol": 1e-7, "atol": 1e-6}


def cases() -> list[Case]:
    out: list[Case] = []
    m = 12
    t, _, _ = planted(160, 620, m)
    mp = stumpy.stump(t, m)
    P = mp[:, 0].astype(float)
    q = t[32 : 32 + m].copy()
    res = stumpy.match(q, t)
    out.append(case("match", "match", [q, t], as_float(res), **SELF))
    res = stumpy.match(q, t, max_distance=3.0, max_matches=3)
    out.append(case("match/limits", "match", [q, t], as_float(res), **SELF, maxDistance=3.0, maxMatches=3))
    d, i = stumpy.motifs(t, P)
    out.append(case("motifs", "motifs", [t, P], np.concatenate([as_float(d), as_float(i)]), **TOL))
    d, i = stumpy.motifs(t, P, max_motifs=3, max_distance=4.0, cutoff=np.inf, max_matches=4)
    out.append(
        case(
            "motifs/3",
            "motifs",
            [t, P],
            np.concatenate([as_float(d), as_float(i)]),
            **TOL,
            maxMotifs=3,
            maxDistance=4.0,
            cutoff=1e300,
            maxMatches=4,
        )
    )
    s = series(140, 621)
    smp = stumpy.stump(s, 8)
    IL = smp[:, 2].astype(np.int64)
    IR = smp[:, 3].astype(np.int64)
    chains, longest = stumpy.allc(IL, IR)
    flat_chains = sorted([c.tolist() for c in chains])
    enc = [float(len(flat_chains))] + [float(len(c)) for c in flat_chains] + [float(v) for c in flat_chains for v in c]
    out.append(case("allc", "allc", [IL.astype(float), IR.astype(float)], np.array(enc + [float(v) for v in longest])))
    out.append(case("atsc", "atsc", [IL.astype(float), IR.astype(float)], as_float(stumpy.atsc(IL, IR, 10)), j=10))
    return out
