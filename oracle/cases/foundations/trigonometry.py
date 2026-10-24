"""Angles, inverse functions and triangle solving against numpy and sympy.geometry.Triangle."""

import numpy as np
import sympy
from sympy.geometry import Point, Triangle
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng, uniform

REFERENCE: dict[str, str] = {"numpy": np.__version__, "sympy": sympy.__version__}


def parts(t: Triangle) -> FloatArray:
    """[a, b, c, A, B, C, area] with A at the first vertex, B at the second, C at the third and a opposite A."""
    v1, v2, v3 = t.vertices
    assert isinstance(v1, Point) and isinstance(v2, Point) and isinstance(v3, Point)
    sides = [v2.distance(v3), v1.distance(v3), v1.distance(v2)]
    angles = [t.angles[v1], t.angles[v2], t.angles[v3]]
    return np.array([float(sympy.N(x, 20)) for x in [*sides, *angles, abs(t.area)]], dtype=np.float64)


def cases() -> list[Case]:
    g = rng(1201)
    out: list[Case] = []
    deg = np.array([-720.0, -90.0, 0.0, 30.0, 45.0, 180.0, 359.0, 1e3])
    out.append(case("radians", "radians", [deg], np.radians(deg), rtol=1e-15))
    rad = uniform(g, (12,), -10.0, 10.0)
    out.append(case("degrees", "degrees", [rad], np.degrees(rad), rtol=1e-15))
    out.append(case("wrap/signed", "wrapSigned", [rad], np.mod(rad + np.pi, 2 * np.pi) - np.pi, rtol=1e-12, atol=1e-14))
    out.append(case("wrap/positive", "wrapPositive", [rad], np.mod(rad, 2 * np.pi), rtol=1e-12, atol=1e-14))
    y = uniform(g, (10,))
    x = uniform(g, (10,))
    out.append(case("directionAngle", "directionAngle", [y, x], np.mod(np.arctan2(y, x), 2 * np.pi), rtol=1e-14))
    out.append(case("unitCircle", "unitCirclePoint", [rad], np.stack([np.cos(rad), np.sin(rad)], -1), rtol=1e-14))

    for i, sides in enumerate([(3, 4, 5), (7, 8, 9), (2, 3, 4), (5, 5, 8), (1, 1, 1)]):
        # sss=(|v1 v2|, |v2 v3|, |v3 v1|) = (c, a, b)
        t = Triangle(sss=sides)
        assert isinstance(t, Triangle)
        out.append(case(f"sss/{i}", "sss", [np.array(sides, dtype=np.float64)], parts(t), rtol=1e-12))
    for i, (s1, angle, s2) in enumerate([(1, 45, 2), (3, 120, 5), (4, 10, 4), (6, 90, 2)]):
        # sas=(|v1 v3|, angle at v1, |v1 v2|) = (b, A, c)
        t = Triangle(sas=(s1, angle, s2))
        assert isinstance(t, Triangle)
        given = np.array([s1, np.radians(angle), s2], dtype=np.float64)
        out.append(case(f"sas/{i}", "sas", [given], parts(t), rtol=1e-12))
    for i, (a1, side, a2) in enumerate([(30, 1, 30), (40, 3, 75), (100, 2, 20), (60, 5, 60)]):
        # asa=(angle at v1, |v1 v2|, angle at v2) = (A, c, B)
        t = Triangle(asa=(a1, side, a2))
        assert isinstance(t, Triangle)
        given = np.array([np.radians(a1), side, np.radians(a2)], dtype=np.float64)
        out.append(case(f"asa/{i}", "asa", [given], parts(t), rtol=1e-12))
    return out
