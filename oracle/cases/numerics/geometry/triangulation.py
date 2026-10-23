"""Delaunay triangulation against scipy.spatial.Delaunay (simplices compared as sorted vertex triples)."""

import numpy as np
import numpy.typing as npt
import scipy
import scipy.spatial as ss
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def canonical(simplices: npt.NDArray[np.int32]) -> FloatArray:
    rows = np.sort(simplices, axis=1)
    return rows[np.lexsort(rows.T[::-1])].astype(np.float64)


def cases() -> list[Case]:
    out: list[Case] = []
    for seed, n in [(70, 10), (71, 40), (72, 150)]:
        pts = rng(seed).uniform(-3, 5, size=(n, 2))
        tri = ss.Delaunay(pts)
        out.append(case(f"delaunay/{n}", "delaunay", [pts], canonical(tri.simplices), rtol=0))
        q = rng(seed + 10).uniform(-4, 6, size=(15, 2))
        found = tri.find_simplex(q) >= 0
        out.append(case(f"inside/{n}", "inside", [pts, q], found.astype(np.float64), rtol=0))
        hull = ss.ConvexHull(pts)
        out.append(case(f"hull/{n}", "hull", [pts], np.sort(hull.vertices).astype(np.float64), rtol=0))
    return out
