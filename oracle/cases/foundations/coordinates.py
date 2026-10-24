"""Coordinate systems, barycentric coordinates, change of basis and Jacobian determinants against numpy, scipy.spatial
and sympy."""

import numpy as np
import scipy
import sympy
from scipy.spatial import Delaunay
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng, uniform

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__, "sympy": sympy.__version__}


def jacobian_det(kind: str, point: FloatArray) -> float:
    """The determinant of sympy's symbolic Jacobian of the map to Cartesian coordinates, evaluated at a point."""
    a, b, c = sympy.symbols("a b c", real=True)
    maps: dict[str, tuple[list[sympy.Expr], list[sympy.Symbol]]] = {
        "polar": ([a * sympy.cos(b), a * sympy.sin(b)], [a, b]),
        "cylindrical": ([a * sympy.cos(b), a * sympy.sin(b), c], [a, b, c]),
        "spherical": ([a * sympy.sin(b) * sympy.cos(c), a * sympy.sin(b) * sympy.sin(c), a * sympy.cos(b)], [a, b, c]),
    }
    exprs, symbols = maps[kind]
    d = sympy.simplify(sympy.Matrix(exprs).jacobian(symbols).det())
    return float(d.subs(dict(zip(symbols, [float(x) for x in point], strict=False))))


def cases() -> list[Case]:
    g = rng(1204)
    out: list[Case] = []
    p2 = uniform(g, (6, 2))
    out.append(case("polar", "cartesianToPolar", [p2], np.stack([np.hypot(*p2.T), np.arctan2(p2[:, 1], p2[:, 0])], -1)))
    rt = np.stack([uniform(g, (6,), 0.1, 3), uniform(g, (6,), -3, 3)], -1)
    out.append(
        case(
            "fromPolar",
            "polarToCartesian",
            [rt],
            np.stack([rt[:, 0] * np.cos(rt[:, 1]), rt[:, 0] * np.sin(rt[:, 1])], -1),
        )
    )
    p3 = uniform(g, (6, 3))
    rho = np.hypot(p3[:, 0], p3[:, 1])
    phi = np.arctan2(p3[:, 1], p3[:, 0])
    out.append(case("cylindrical", "cartesianToCylindrical", [p3], np.stack([rho, phi, p3[:, 2]], -1)))
    r = np.linalg.norm(p3, axis=-1)
    out.append(
        case("spherical", "cartesianToSpherical", [p3], np.stack([r, np.arccos(p3[:, 2] / r), phi], -1), rtol=1e-12)
    )
    sph = np.stack([uniform(g, (6,), 0.1, 3), uniform(g, (6,), 0, 3), uniform(g, (6,), -3, 3)], -1)
    want = np.stack(
        [
            sph[:, 0] * np.sin(sph[:, 1]) * np.cos(sph[:, 2]),
            sph[:, 0] * np.sin(sph[:, 1]) * np.sin(sph[:, 2]),
            sph[:, 0] * np.cos(sph[:, 1]),
        ],
        -1,
    )
    out.append(case("fromSpherical", "sphericalToCartesian", [sph], want, atol=1e-15))

    for dim in [2, 3]:
        pts = g.normal(size=(dim + 1, dim))
        tri = Delaunay(pts)
        q = g.normal(size=(5, dim)) * 0.5 + pts.mean(0)
        t = tri.transform[0]
        lam = np.einsum("ij,nj->ni", t[:dim], q - t[dim])
        # scipy's transform belongs to its own vertex order: put the weights back in the order of `pts`.
        order = tri.simplices[0]
        weights = np.zeros((5, dim + 1))
        weights[:, order] = np.c_[lam, 1 - lam.sum(1)]
        out.append(case(f"barycentric/{dim}d", "barycentric", [q, pts], weights, rtol=1e-10, atol=1e-12))

    basis = g.normal(size=(3, 3)) + 2 * np.eye(3)
    other = g.normal(size=(3, 3)) + 2 * np.eye(3)
    v = g.normal(size=3)
    a = g.normal(size=(3, 3))
    out.append(case("coordinates", "coordinatesInBasis", [v, basis], np.linalg.solve(basis, v), rtol=1e-10))
    out.append(case("changeOfBasis", "changeOfBasis", [basis, other], np.linalg.solve(other, basis), rtol=1e-10))
    out.append(
        case("matrixInBasis", "matrixInBasis", [a, basis], np.linalg.solve(basis, a @ basis), rtol=1e-10, atol=1e-12)
    )

    for kind, pt in [("polar", [1.5, 0.7]), ("cylindrical", [2.0, -1.0, 0.3]), ("spherical", [1.2, 0.9, 2.1])]:
        point = np.array(pt)
        out.append(
            case(f"jacobian/{kind}", "jacobianDeterminant", [point], np.array(jacobian_det(kind, point)), kind=kind)
        )
    return out
