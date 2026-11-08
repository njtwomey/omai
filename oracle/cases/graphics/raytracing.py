"""A small reference image for Whitted ray tracing, rendered by a straightforward independent tracer: camera rays by
unprojecting the near and far planes through the inverse view-projection matrix, brute-force nearest hits (quadratic
roots for spheres, the plane equation for the floor), Lambertian and normalised Blinn-Phong shading under a point light
and a directional light with shadow rays, an ambient term, and mirror reflection weighted by the material's mirror
fraction, to depth 3. The engine must agree pixel for pixel."""

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"numpy": np.__version__}

WIDTH, HEIGHT = 16, 12
EYE = np.array([0.0, 2.0, 6.0])
TARGET = np.array([0.5, 0.8, 0.0])
FOV, NEAR, FAR = 50.0, 0.1, 100.0
BACKGROUND = np.array([0.2, 0.3, 0.5])
AMBIENT = np.array([0.05, 0.05, 0.05])
MAX_DEPTH = 3

# Objects: (kind, geometry, albedo, specular, shininess, model, mirror).
type Obj = tuple[str, tuple[FloatArray, float], FloatArray, FloatArray, float, str, float]
OBJECTS: list[Obj] = [
    ("plane", (np.array([0.0, 1.0, 0.0]), 0.0), np.array([0.6, 0.6, 0.6]), np.zeros(3), 1.0, "lambert", 0.0),
    (
        "sphere",
        (np.array([0.0, 1.0, 0.0]), 1.0),
        np.array([0.7, 0.2, 0.2]),
        np.array([0.5, 0.5, 0.5]),
        20.0,
        "blinn-phong",
        0.3,
    ),
    ("sphere", (np.array([2.2, 0.7, -0.5]), 0.7), np.array([0.2, 0.3, 0.8]), np.zeros(3), 1.0, "lambert", 0.0),
]
POINT_LIGHT = (np.array([3.0, 5.0, 4.0]), np.array([60.0, 60.0, 60.0]))
SUN = (np.array([-1.0, -2.0, -1.0]) / np.sqrt(6.0), np.array([0.5, 0.5, 0.5]))


def unit(v: FloatArray) -> FloatArray:
    return v / np.linalg.norm(v)


def view_projection() -> FloatArray:
    f = unit(TARGET - EYE)
    s = unit(np.cross(f, [0.0, 1.0, 0.0]))
    u = np.cross(s, f)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = s, u, -f
    view[:3, 3] = -view[:3, :3] @ EYE
    t = 1 / np.tan(np.radians(FOV) / 2)
    proj = np.zeros((4, 4))
    proj[0, 0], proj[1, 1] = t / (WIDTH / HEIGHT), t
    proj[2, 2], proj[2, 3], proj[3, 2] = (FAR + NEAR) / (NEAR - FAR), 2 * FAR * NEAR / (NEAR - FAR), -1
    return proj @ view


def hit(o: FloatArray, d: FloatArray, t_max: float = np.inf) -> tuple[float, int]:
    best, index = t_max, -1
    for i, (kind, (g, r), *_) in enumerate(OBJECTS):
        if kind == "plane":
            denom = float(g @ d)
            t = (r - float(g @ o)) / denom if abs(denom) > 1e-15 else np.inf
        else:
            oc = o - g
            b, c = float(oc @ d), float(oc @ oc) - r * r
            disc = b * b - c
            if disc < 0:
                continue
            roots = [x for x in sorted([-b - np.sqrt(disc), -b + np.sqrt(disc)]) if x >= 1e-9]
            t = roots[0] if roots else np.inf
        if 1e-9 <= t < best:
            best, index = t, i
    return best, index


def lift(p: FloatArray, n: FloatArray) -> FloatArray:
    return p + n * 1e-7 * max(1.0, float(np.max(np.abs(p))))


def trace(o: FloatArray, d: FloatArray, depth: int, weight: float) -> FloatArray:
    t, i = hit(o, d)
    if i < 0:
        return weight * BACKGROUND
    kind, (g, _), albedo, spec, shininess, model, mirror = OBJECTS[i]
    p = o + t * d
    n = g if kind == "plane" else unit(p - g)
    if float(n @ d) > 0:
        n = -n
    v = -d
    local = albedo * AMBIENT
    samples = [(unit(POINT_LIGHT[0] - p), np.linalg.norm(POINT_LIGHT[0] - p))]
    for (ld, dist), irradiance in zip(
        [*samples, (-SUN[0], np.inf)],
        [POINT_LIGHT[1] / float(np.sum((POINT_LIGHT[0] - p) ** 2)), SUN[1]],
        strict=True,
    ):
        if float(n @ ld) <= 0:
            continue
        if hit(lift(p, n), ld, float(dist) * (1 - 1e-9))[1] >= 0:
            continue
        f = albedo / np.pi
        if model == "blinn-phong":
            h = unit(ld + v)
            f = f + spec * (shininess + 8) / (8 * np.pi) * max(0.0, float(n @ h)) ** shininess
        local = local + f * irradiance * float(n @ ld)
    colour = weight * (1 - mirror) * local
    if mirror > 0 and depth < MAX_DEPTH and weight * mirror > 1e-4:
        r = d - 2 * float(d @ n) * n
        colour = colour + trace(lift(p, n), r, depth + 1, weight * mirror)
    return colour


def cases() -> list[Case]:
    inv = np.linalg.inv(view_projection())
    image = np.zeros((HEIGHT, WIDTH, 3))
    for y in range(HEIGHT):
        for x in range(WIDTH):
            sx, sy = 2 * (x + 0.5) / WIDTH - 1, 1 - 2 * (y + 0.5) / HEIGHT
            a = inv @ [sx, sy, -1.0, 1.0]
            b = inv @ [sx, sy, 1.0, 1.0]
            d = unit(b[:3] / b[3] - a[:3] / a[3])
            image[y, x] = trace(EYE, d, 0, 1.0)
    return [case("whitted", "whitted", [], image, rtol=1e-9, atol=1e-10)]
