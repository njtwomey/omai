"""Distances between vectors and point sets against scipy.spatial (Minkowski, cosine, Mahalanobis, Procrustes,
Hausdorff); the Chamfer distance as PyTorch3D defines it, over scipy's distance matrix.
"""

import numpy as np
import scipy
import scipy.spatial as sp
import scipy.spatial.distance as ssd
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def chamfer(a: FloatArray, b: FloatArray, squared: bool) -> float:
    d = ssd.cdist(a, b)
    d = d**2 if squared else d
    return float(d.min(axis=1).mean() + d.min(axis=0).mean())


def cases() -> list[Case]:
    g = rng(66)
    u, v = g.normal(size=5), g.normal(size=5)
    w = g.uniform(0.2, 2.0, size=5)
    m = g.normal(size=(5, 5))
    vi = np.linalg.inv(m @ m.T + 5 * np.eye(5))
    rows_u, rows_v = g.normal(size=(4, 3)), g.normal(size=(4, 3))
    a, b = g.normal(size=(12, 2)), g.normal(size=(9, 2)) + 0.5
    p1 = g.normal(size=(6, 3))
    p2 = p1 @ np.linalg.qr(g.normal(size=(3, 3)))[0] * 2 + 1 + 0.05 * g.normal(size=(6, 3))
    cos_rows = np.mean([1 - ssd.cosine(x, y) for x, y in zip(rows_u, rows_v, strict=True)])
    return [
        case("minkowskiDistance/2", "minkowskiDistance", [u, v], np.asarray(ssd.minkowski(u, v, 2))),
        case(
            "minkowskiDistance/1.5-weighted",
            "minkowskiDistance",
            [u, v, w],
            np.asarray(ssd.minkowski(u, v, 1.5, w=w)),
            p=1.5,
        ),
        case("minkowskiDistance/inf", "minkowskiDistance", [u, v], np.asarray(ssd.chebyshev(u, v)), p="inf"),
        case("cosineSimilarity", "cosineSimilarity", [u, v], np.asarray(1 - ssd.cosine(u, v))),
        case("cosineSimilarity/rows", "cosineSimilarity", [rows_u, rows_v], np.asarray(cos_rows)),
        case("mahalanobisDistance", "mahalanobisDistance", [u, v, vi], np.asarray(ssd.mahalanobis(u, v, vi))),
        case("procrustesDisparity", "procrustesDisparity", [p1, p2], np.asarray(sp.procrustes(p1, p2)[2]), rtol=1e-10),
        case(
            "hausdorffDistance",
            "hausdorffDistance",
            [a, b],
            np.asarray(max(ssd.directed_hausdorff(a, b)[0], ssd.directed_hausdorff(b, a)[0])),
        ),
        case(
            "hausdorffDistance/directed",
            "hausdorffDistance",
            [a, b],
            np.asarray(ssd.directed_hausdorff(a, b)[0]),
            directed=True,
        ),
        case("chamferDistance", "chamferDistance", [a, b], np.asarray(chamfer(a, b, True))),
        case("chamferDistance/plain", "chamferDistance", [a, b], np.asarray(chamfer(a, b, False)), squared=False),
    ]
