"""Vector algebra, lines and planes against numpy, scipy.linalg and sympy.geometry."""

import numpy as np
import scipy
import scipy.linalg as sl
import sympy
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sympy.geometry import Line3D, Plane, Point3D

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__, "sympy": sympy.__version__}


def ints(g: np.random.Generator, shape: tuple[int, ...]) -> FloatArray:
    """Small integer coordinates, so that sympy computes exactly."""
    return np.asarray(g.integers(-5, 6, size=shape), dtype=np.float64)


def point(v: FloatArray) -> Point3D:
    return Point3D(*(sympy.Integer(int(x)) for x in v))


def as_float(x: sympy.Basic) -> float:
    return float(sympy.N(x, 20))


def coords(p: Point3D) -> FloatArray:
    return np.array([as_float(c) for c in p.args], dtype=np.float64)


def cases() -> list[Case]:
    g = rng(1202)
    out: list[Case] = []
    a = g.normal(size=(5, 3))
    b = g.normal(size=(5, 3))
    c = g.normal(size=(5, 3))
    out.append(case("dot", "dotProduct", [a, b], np.einsum("ij,ij->i", a, b), rtol=1e-13, atol=1e-14))
    out.append(case("cross", "crossProduct", [a, b], np.cross(a, b), rtol=1e-13, atol=1e-14))
    a2 = g.normal(size=(4, 2))
    b2 = g.normal(size=(4, 2))
    out.append(case("cross/2d", "crossProduct", [a2, b2], a2[:, 0] * b2[:, 1] - a2[:, 1] * b2[:, 0], rtol=1e-13))
    triple = np.linalg.det(np.stack([a, b, c], axis=1))
    out.append(case("triple", "tripleProduct", [a, b, c], triple, rtol=1e-12, atol=1e-14))
    out.append(
        case("vectorTriple", "vectorTripleProduct", [a, b, c], np.cross(a, np.cross(b, c)), rtol=1e-12, atol=1e-14)
    )
    out.append(case("magnitude", "magnitude", [a], np.linalg.norm(a, axis=-1), rtol=1e-14))
    cos = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    out.append(case("angle", "angleBetween", [a, b], np.arccos(cos), rtol=1e-12))
    proj = (np.einsum("ij,ij->i", a, b) / np.einsum("ij,ij->i", b, b))[:, None] * b
    out.append(case("projection", "projection", [a, b], proj, rtol=1e-12, atol=1e-14))
    out.append(case("rejection", "rejection", [a, b], a - proj, rtol=1e-12, atol=1e-14))

    for i, k in enumerate([2, 3, 4]):
        m = g.normal(size=(k, 5))
        if k == 4:
            m[3] = m[0] - 2 * m[1]
        q = sl.orth(m.T)
        out.append(case(f"span/projector/{i}", "spanProjector", [m], q @ q.T, rtol=1e-10, atol=1e-12))
        out.append(case(f"span/rank/{i}", "spanDimension", [m], np.array(np.linalg.matrix_rank(m))))

    for i in range(4):
        p, d, x = ints(g, (3,)), ints(g, (3,)), ints(g, (3,))
        if not d.any():
            d[0] = 1.0
        line = Line3D(point(p), direction_ratio=[int(v) for v in d])
        out.append(case(f"pointLine/{i}", "pointLineDistance", [x, p, d], np.array(as_float(line.distance(point(x))))))
        n = ints(g, (3,))
        if not n.any():
            n[2] = 1.0
        plane = Plane(point(p), normal_vector=[int(v) for v in n])
        signed = as_float(plane.distance(point(x))) * float(np.sign(n @ (x - p)))
        out.append(case(f"pointPlane/{i}", "pointPlaneDistance", [x, p, n], np.array(signed), rtol=1e-12, atol=1e-14))
        hit = list(Plane(point(x), normal_vector=[int(v) for v in n]).intersection(line) or [])
        if len(hit) == 1 and isinstance(hit[0], Point3D):
            out.append(case(f"linePlane/{i}", "linePlaneIntersection", [p, d, x, n], coords(hit[0]), rtol=1e-12))
        p2, d2 = ints(g, (3,)), ints(g, (3,))
        if not d2.any():
            d2[1] = 1.0
        other = Line3D(point(p2), direction_ratio=[int(v) for v in d2])
        # The skew-line distance: |(p2 - p1) . (d1 x d2)| / |d1 x d2|, as sympy's Line3D.distance gives.
        out.append(case(f"lineLine/{i}", "lineLineDistance", [p, d, p2, d2], np.array(as_float(line.distance(other)))))
        other_plane = Plane(point(p2), normal_vector=[int(v) for v in d2])
        meet = list(plane.intersection(other_plane) or [])
        if len(meet) == 1 and isinstance(meet[0], Line3D):
            # A line is a point and a direction up to scale: compare the point nearest the origin and the projector.
            ln = meet[0]
            foot = ln.projection(Point3D(0, 0, 0))
            assert isinstance(foot, Point3D)
            dr = np.array([as_float(v) for v in ln.direction_ratio], dtype=np.float64)
            given = [p, n, p2, d2]
            out.append(case(f"planePlane/point/{i}", "planePlanePoint", given, coords(foot), rtol=1e-12, atol=1e-14))
            proj = np.outer(dr, dr) / (dr @ dr)
            out.append(case(f"planePlane/dir/{i}", "planePlaneProjector", given, proj, rtol=1e-12, atol=1e-14))
    return out
