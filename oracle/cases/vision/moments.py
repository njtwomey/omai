"""Moments against OpenCV: the 24 moments of polygons (Green's theorem) and of images (float64 and binary), the Hu
invariants, and matchShapes by its three methods."""

from importlib.metadata import version

import cv2
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.signal_helpers import smooth_image

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}

KEYS = [
    "m00", "m10", "m01", "m20", "m11", "m02", "m30", "m21", "m12", "m03",
    "mu20", "mu11", "mu02", "mu30", "mu21", "mu12", "mu03",
    "nu20", "nu11", "nu02", "nu30", "nu21", "nu12", "nu03",
]  # fmt: skip

TOL = {"rtol": 1e-10, "atol": 1e-9}


def flat(m: dict[str, float]) -> FloatArray:
    return np.asarray([m[k] for k in KEYS], dtype=np.float64)


def polygons(g: np.random.Generator) -> list[np.ndarray]:
    """A rectangle, a clockwise triangle, a random star-shaped polygon and a degenerate segment."""
    angles = np.sort(g.uniform(0, 2 * np.pi, 9))
    radii = g.uniform(4, 10, 9)
    star = np.stack([20 + radii * np.cos(angles), 15 + radii * np.sin(angles)], 1)
    return [
        np.array([[1, 1], [9, 1], [9, 6], [1, 6]], np.int32),
        np.array([[0, 0], [3, 8], [10, 2]], np.int32),
        np.rint(star).astype(np.int32),
        np.array([[0, 0], [5, 5]], np.int32),
    ]


def cases() -> list[Case]:
    g = rng(4350)
    out: list[Case] = []
    for k, p in enumerate(polygons(g)):
        m = cv2.moments(p)
        out.append(case(f"moments/{k}", "moments", [p.astype(np.float64)], flat(m), **TOL))
        out.append(case(f"huMoments/{k}", "huMoments", [p.astype(np.float64)], cv2.HuMoments(m).ravel(), **TOL))
    img = smooth_image(g, 13, 17)
    out.append(case("imageMoments", "imageMoments", [img], flat(cv2.moments(img)), **TOL, binary=False))
    mask = (img > 0.5).astype(np.float64)
    out.append(case("imageMoments/binary", "imageMoments", [img], flat(cv2.moments(img, True)), **TOL, binary=True))
    out.append(case("imageMoments/mask", "imageMoments", [mask], flat(cv2.moments(mask)), **TOL, binary=False))
    shapes = polygons(g)[:3]
    scaled = (shapes[2] * 2 + 3).astype(np.int32)
    pairs = [(shapes[0], shapes[1]), (shapes[1], shapes[2]), (shapes[2], scaled), (shapes[0], shapes[2])]
    for k, (a, b) in enumerate(pairs):
        for method in [1, 2, 3]:
            out.append(
                case(
                    f"matchShapes/{k}/{method}",
                    "matchShapes",
                    [a.astype(np.float64), b.astype(np.float64)],
                    np.atleast_1d(cv2.matchShapes(a, b, method, 0)),
                    rtol=1e-9,
                    atol=1e-10,
                    method=method,
                )
            )
    return out
