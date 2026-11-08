"""Matrix-profile algorithms against STUMPY: stump (self- and AB-joins, top-k, NaN windows), aamp (non-normalised,
p-norms), stomp (same profile), the reference stamp, prescrump with every row, scrump with every diagonal, and the
incremental stumpi with and without egress. Each output stacks the distances, indices, and left and right indices."""

from importlib.metadata import version

import numpy as np
import numpy.typing as npt
import stumpy
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray
from omai_oracle.timeseries_helpers import series
from stumpy.stamp import stamp as reference_stamp

REFERENCE: dict[str, str] = {"stumpy": version("stumpy"), "numpy": np.__version__}

# Diagonal and row traversals accumulate dot products in different orders: distances agree to 1e-8.
TOL = {"rtol": 1e-8, "atol": 1e-8}


def flat(mp: npt.ArrayLike, k: int = 1) -> FloatArray:
    """P (l * k), I (l * k), left and right indices of a STUMPY profile."""
    a = np.asarray(mp)
    return np.concatenate(
        [
            a[:, :k].astype(float).ravel(),
            a[:, k : 2 * k].astype(float).ravel(),
            a[:, 2 * k].astype(float),
            a[:, 2 * k + 1].astype(float),
        ]
    )


def cases() -> list[Case]:
    out: list[Case] = []
    t = series(120, 610)
    tb = series(90, 611)
    tn = t.copy()
    tn[70] = np.nan
    for m in [5, 12]:
        out.append(case(f"stump/{m}", "stump", [t], flat(stumpy.stump(t, m)), **TOL, m=m))
        out.append(case(f"stomp/{m}", "stomp", [t], flat(stumpy.stump(t, m)), **TOL, m=m))
    out.append(case("stump/k3", "stump", [t], flat(stumpy.stump(t, 8, k=3), 3), **TOL, m=8, k=3))
    out.append(case("stomp/k3", "stomp", [t], flat(stumpy.stump(t, 8, k=3), 3), **TOL, m=8, k=3))
    out.append(case("stump/nan", "stump", [tn], flat(stumpy.stump(tn, 8)), **TOL, m=8))
    out.append(case("stomp/nan", "stomp", [tn], flat(stumpy.stump(tn, 8)), **TOL, m=8))
    out.append(case("stump/ab", "stump", [t, tb], flat(stumpy.stump(t, 8, tb, ignore_trivial=False)), **TOL, m=8))
    out.append(case("stomp/ab", "stomp", [t, tb], flat(stumpy.stump(t, 8, tb, ignore_trivial=False)), **TOL, m=8))
    out.append(
        case(
            "stump/ab/k2", "stump", [t, tb], flat(stumpy.stump(t, 8, tb, ignore_trivial=False, k=2), 2), **TOL, m=8, k=2
        )
    )
    for p in [1.0, 2.0]:
        out.append(case(f"aamp/{p}", "stump", [t], flat(stumpy.aamp(t, 8, p=p)), **TOL, m=8, normalize=False, p=p))
    out.append(
        case(
            "aamp/ab", "stump", [t, tb], flat(stumpy.aamp(t, 8, tb, ignore_trivial=False)), **TOL, m=8, normalize=False
        )
    )
    out.append(case("aamp/nan", "stump", [tn], flat(stumpy.aamp(tn, 8)), **TOL, m=8, normalize=False))
    ref = np.asarray(reference_stamp(t, t, 8, ignore_trivial=True))
    out.append(
        case(
            "stamp",
            "stamp",
            [t],
            np.concatenate([ref[:, 0].astype(float), ref[:, 1].astype(float)]),
            **TOL,
            m=8,
            exclusion=4,
        )
    )
    np.random.seed(640)  # prescrump visits rows in a random order (STUMPY uses the global generator)
    p, i = stumpy.prescrump(t, 8, s=1)
    # Rounded: which row finds a pair (and so the last bits of its distance) depends on the order and thread count.
    out.append(
        case("prescrump/s1", "prescrump", [t], np.concatenate([np.round(p, 12), i.astype(float)]), **TOL, m=8, s=1)
    )
    out.append(case("scrump/full", "scrump", [t], flat(stumpy.stump(t, 8)), **TOL, m=8, percentage=1.0))
    out.append(case("scrump/full/k2", "scrump", [t], flat(stumpy.stump(t, 8, k=2), 2), **TOL, m=8, percentage=1.0, k=2))
    stream = series(16, 612)
    stream[5] = np.nan
    for egress in [True, False]:
        s = stumpy.stumpi(t[:60], 8, egress=egress)
        for v in stream:
            s.update(v)
        res = np.concatenate([s.P_, s.I_.astype(float), s.left_I_.astype(float)])
        out.append(case(f"stumpi/{egress}", "stumpi", [t[:60], stream], res, **TOL, m=8, egress=egress))
    return out
