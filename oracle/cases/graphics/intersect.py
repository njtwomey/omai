"""Ray-primitive intersections by closed forms, computed independently of the engine's algorithms: sphere hits as the
smallest admissible root of the quadratic (numpy.roots), plane hits from the plane equation, triangle hits by solving
`o + t d = a + u (b - a) + v (c - a)` with numpy.linalg.solve, and box hits by intersecting the six face planes and
keeping the nearest point that lies on its face. Misses are NaN."""

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__}

T_MIN = 1e-9


def unit(v: FloatArray) -> FloatArray:
    return v / np.linalg.norm(v)


def sphere_hit(o: FloatArray, d: FloatArray, c: FloatArray, r: float) -> list[float]:
    oc = o - c
    roots = np.roots([float(d @ d), 2 * float(oc @ d), float(oc @ oc) - r * r])
    real = sorted(float(x.real) for x in roots if abs(x.imag) < 1e-12 and x.real >= T_MIN)
    if not real:
        return [np.nan] * 4
    t = real[0]
    n = (o + t * d - c) / r
    return [t, *(float(x) for x in n)]


def plane_hit(o: FloatArray, d: FloatArray, p: FloatArray, n: FloatArray) -> list[float]:
    denom = float(n @ d)
    t = float((p - o) @ n) / denom if abs(denom) > 1e-15 else np.nan
    return [t] if t >= T_MIN else [np.nan]


def triangle_hit(o: FloatArray, d: FloatArray, a: FloatArray, b: FloatArray, c: FloatArray) -> list[float]:
    m = np.stack([d, a - b, a - c], axis=1)
    t, u, v = np.linalg.solve(m, a - o)
    if t >= T_MIN and u >= 0 and v >= 0 and u + v <= 1:
        return [float(t), float(u), float(v)]
    return [np.nan] * 3


def box_hit(o: FloatArray, d: FloatArray, lo: FloatArray, hi: FloatArray) -> list[float]:
    best: list[float] = [np.inf, np.nan, np.nan, np.nan]
    for axis in range(3):
        if d[axis] == 0:
            continue
        for bound, sign in ((lo[axis], -1.0), (hi[axis], 1.0)):
            t = float((bound - o[axis]) / d[axis])
            if t < T_MIN or t >= best[0]:
                continue
            q = o + t * d
            others = [k for k in range(3) if k != axis]
            if all(lo[k] - 1e-12 <= q[k] <= hi[k] + 1e-12 for k in others):
                n = np.zeros(3)
                n[axis] = sign
                best = [t, *(float(x) for x in n)]
    return best if np.isfinite(best[0]) else [np.nan] * 4


def cases() -> list[Case]:
    g = rng(4242)
    out: list[Case] = []
    n = 24
    origins = g.uniform(-4, 4, size=(n, 3))
    targets = g.uniform(-1, 1, size=(n, 3))
    dirs = np.array([unit(t - o) for o, t in zip(origins, targets, strict=True)])
    # Some rays aim elsewhere, so misses are covered too.
    dirs[::4] = np.array([unit(v) for v in g.normal(size=(len(dirs[::4]), 3))])
    centre, radius = g.uniform(-0.5, 0.5, size=3), 1.2
    out.append(
        case(
            "sphere",
            "sphere",
            [origins, dirs, centre],
            np.array([sphere_hit(o, d, centre, radius) for o, d in zip(origins, dirs, strict=True)]),
            rtol=1e-12,
            atol=1e-12,
            radius=radius,
        )
    )
    point, normal = g.uniform(-0.5, 0.5, size=3), unit(g.normal(size=3))
    out.append(
        case(
            "plane",
            "plane",
            [origins, dirs, point, normal],
            np.array([plane_hit(o, d, point, normal) for o, d in zip(origins, dirs, strict=True)]),
            rtol=1e-12,
            atol=1e-12,
        )
    )
    a, b, c = g.uniform(-1.5, 1.5, size=(3, 3))
    # Aim at points of the triangle's plane around it, so about half the rays hit.
    w = g.uniform(-0.3, 1.0, size=(n, 2))
    aims = a + w[:, :1] * (b - a) + w[:, 1:] * (c - a)
    tri_dirs = np.array([unit(t - o) for o, t in zip(origins, aims, strict=True)])
    out.append(
        case(
            "triangle",
            "triangle",
            [origins, tri_dirs, a, b, c],
            np.array([triangle_hit(o, d, a, b, c) for o, d in zip(origins, tri_dirs, strict=True)]),
            rtol=1e-10,
            atol=1e-12,
        )
    )
    lo, hi = -g.uniform(0.3, 1, size=3), g.uniform(0.3, 1, size=3)
    out.append(
        case(
            "box",
            "box",
            [origins, dirs, lo, hi],
            np.array([box_hit(o, d, lo, hi) for o, d in zip(origins, dirs, strict=True)]),
            rtol=1e-12,
            atol=1e-12,
        )
    )
    return out
