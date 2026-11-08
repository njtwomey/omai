"""Polygons and convex hulls against shapely and scipy.spatial.ConvexHull."""

import numpy as np
import scipy
import scipy.spatial as ss
import shapely
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from shapely.geometry import Point, Polygon

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "shapely": shapely.__version__}

STAR: FloatArray = np.array(
    [[0.0, 0.0], [2.0, -1.0], [4.0, 0.0], [3.0, 2.0], [4.0, 4.0], [2.0, 3.0], [0.0, 4.0], [1.0, 2.0]]
)
CLIP: FloatArray = np.array([[1.0, 1.0], [5.0, 1.5], [4.5, 5.0], [0.5, 3.5]])


def cases() -> list[Case]:
    g = rng(140)
    out: list[Case] = []
    star = Polygon(STAR)
    out.append(case("area", "area", [STAR], np.array(star.area), rtol=1e-13))
    out.append(case("area/clockwise", "area", [STAR[::-1].copy()], np.array(-star.area), rtol=1e-13))
    out.append(case("centroid", "centroid", [STAR], np.array([star.centroid.x, star.centroid.y]), rtol=1e-13))
    out.append(case("perimeter", "perimeter", [STAR], np.array(star.length), rtol=1e-13))
    queries = g.uniform(-1, 5, size=(30, 2))
    inside = np.array([1.0 if star.contains(Point(q)) else -1.0 for q in queries])
    out.append(case("pointInPolygon", "pointInPolygon", [queries, STAR], inside, rtol=0))
    clipped = star.intersection(Polygon(CLIP))
    out.append(case("clip/area", "clipArea", [STAR, CLIP], np.array(clipped.area), rtol=1e-12))
    for n, seed in [(12, 141), (60, 142)]:
        pts = rng(seed).normal(size=(n, 2))
        hull = ss.ConvexHull(pts)
        out.append(case(f"hull2/{n}/vertices", "hull2", [pts], np.sort(hull.vertices).astype(np.float64), rtol=0))
        out.append(case(f"hull2/{n}/area", "hull2Area", [pts], np.array(hull.volume), rtol=1e-12))
        pts3 = rng(seed + 10).normal(size=(n, 3))
        h3 = ss.ConvexHull(pts3)
        out.append(case(f"hull3/{n}/vertices", "hull3", [pts3], np.sort(h3.vertices).astype(np.float64), rtol=0))
        out.append(case(f"hull3/{n}/volume", "hull3Volume", [pts3], np.array(h3.volume), rtol=1e-12))
        out.append(case(f"hull3/{n}/area", "hull3Area", [pts3], np.array(h3.area), rtol=1e-12))
    p = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]])
    q = np.array([[0.0, 0.0], [1.0, -0.5], [1.5, 0.5], [0.5, 1.0]])
    sums = (p[:, None, :] + q[None, :, :]).reshape(-1, 2)
    msum = ss.ConvexHull(sums)
    out.append(case("minkowski/area", "minkowskiArea", [p, q], np.array(msum.volume), rtol=1e-12))
    return out
