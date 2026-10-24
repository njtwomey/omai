"""Semantic segmentation against STUMPY: arc counts, fluss and floss, with the engine's idealised arc curve
passed as custom_iac (STUMPY's default fits a beta density to random draws)."""

from importlib.metadata import version

import numpy as np
import stumpy
from stumpy.floss import _nnmark as nnmark  # pyright: ignore[reportPrivateUsage]
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.timeseries_helpers import as_float, ideal_arc_curve

REFERENCE: dict[str, str] = {"stumpy": version("stumpy"), "numpy": np.__version__}

TOL = {"rtol": 1e-8, "atol": 1e-8}
# Exact repeats (a query taken from the series, copied dimensions, planted motifs) have distances sqrt(2m(1 - rho))
# with rho within rounding of 1, where 1e-14 in rho is 1e-7 in the distance: those cases get atol 1e-6.
SELF = {"rtol": 1e-7, "atol": 1e-6}


def cases() -> list[Case]:
    out: list[Case] = []
    g = np.random.default_rng(625)
    seg = np.concatenate([np.sin(np.arange(150) / 3.0), np.sign(np.sin(np.arange(150) / 5.0))]) + 0.05 * g.normal(
        size=300
    )
    segmp = stumpy.stump(seg, 10)
    I_seg = segmp[:, 1].astype(np.int64)
    w = I_seg.shape[0]
    cac, locs = stumpy.fluss(I_seg, 10, 2, custom_iac=ideal_arc_curve(w, True))
    out.append(
        case("fluss", "fluss", [I_seg.astype(float)], np.concatenate([cac, as_float(locs)]), **TOL, L=10, regimes=2)
    )
    out.append(case("arcCounts", "arcCounts", [I_seg.astype(float)], as_float(nnmark(I_seg))))
    old = seg[:200]
    oldmp = stumpy.stump(old, 10)
    fl = stumpy.floss(oldmp, old, 10, 10, custom_iac=ideal_arc_curve(oldmp.shape[0], False))
    for v in seg[200:240]:
        fl.update(v)
    out.append(
        case(
            "floss",
            "floss",
            [old, seg[200:240]],
            np.concatenate([fl.cac_1d_, fl.P_, as_float(fl.I_)]),
            rtol=1e-7,
            atol=1e-7,
            m=10,
            L=10,
        )
    )
    return out
