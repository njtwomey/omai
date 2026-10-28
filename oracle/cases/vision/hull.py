"""Convex hulls and convexity defects against OpenCV: hull indices (both orientations) of random integer and float
point sets, collinear and duplicated points, and the defects of contours of blobs (depths in 1/256 pixel)."""

from importlib.metadata import version
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
import scipy.ndimage as N
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}

EXACT = {"rtol": 1e-12, "atol": 1e-12}


def cases() -> list[Case]:
    g = rng(4360)
    out: list[Case] = []
    sets = {
        "ints": g.integers(0, 30, (25, 2)).astype(np.int32),
        "floats": g.uniform(0, 30, (20, 2)).astype(np.float32),
        "collinear": np.array([[0, 0], [2, 2], [4, 4], [1, 1], [3, 3]], np.int32),
        "duplicates": np.array([[0, 0], [5, 0], [5, 0], [5, 5], [0, 5], [0, 0], [2, 2]], np.int32),
        "square": np.array([[0, 0], [4, 0], [4, 4], [0, 4], [2, 2], [2, 0], [4, 2]], np.int32),
    }
    for name, pts in sets.items():
        for clockwise in [False, True]:
            idx = cv2.convexHull(pts, clockwise=clockwise, returnPoints=False).ravel()
            out.append(
                case(
                    f"convexHull/{name}/{clockwise}",
                    "convexHull",
                    [pts.astype(np.float64)],
                    idx.astype(np.float64),
                    **EXACT,
                    clockwise=clockwise,
                )
            )
    blob = (N.gaussian_filter(g.normal(size=(40, 40)), 3.0) > 0.0).astype(np.uint8)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for k, c in enumerate(sorted(contours, key=lambda c: -len(c))[:3]):
        hull = cv2.convexHull(c, returnPoints=False)
        # OpenCV returns None when there are no defects (its stubs do not say so).
        defects = cast(object, cv2.convexityDefects(c, hull))
        enc = (
            np.zeros((0, 4)) if defects is None else np.asarray(cast(npt.ArrayLike, defects), np.float64).reshape(-1, 4)
        )
        enc[:, 3] /= 256
        out.append(
            case(
                f"convexityDefects/{k}",
                "convexityDefects",
                [c.reshape(-1, 2).astype(np.float64)],
                enc.ravel(),
                **EXACT,
            )
        )
    return out
