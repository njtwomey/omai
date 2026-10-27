"""Fitted shapes against OpenCV: minAreaRect (single precision, so agreement to about 1e-5), boxPoints,
minEnclosingCircle (OpenCV inflates the radius by 1e-4 and works in single precision), fitEllipse and the least-squares
fitLine."""

from importlib.metadata import version

import cv2
import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}

SINGLE = {"rtol": 1e-5, "atol": 1e-4}


def cases() -> list[Case]:
    g = rng(4370)
    out: list[Case] = []
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ellipses: list[np.ndarray] = []
    for k, ang in enumerate([0.0, 0.4, 1.3, 2.2]):
        a, b = 9.0 - k, 3.0 + 0.5 * k
        x, y = a * np.cos(t), b * np.sin(t)
        pts = np.stack([x * np.cos(ang) - y * np.sin(ang) + 20, x * np.sin(ang) + y * np.cos(ang) + 15], 1)
        ellipses.append((pts + 0.05 * g.normal(size=pts.shape)).astype(np.float32))
    clouds = {
        "ints": g.integers(0, 40, (30, 2)).astype(np.int32),
        "floats": g.uniform(0, 40, (25, 2)).astype(np.float32),
        "rect": np.array([[2, 3], [10, 3], [10, 8], [2, 8]], np.int32),
        "tilted": np.array([[0, 5], [5, 0], [12, 7], [7, 12]], np.int32),
        "segment": np.array([[1, 1], [7, 4]], np.int32),
        "ellipse": ellipses[1],
    }
    for name, pts in clouds.items():
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        out.append(
            case(
                f"minAreaRect/{name}",
                "minAreaRect",
                [pts.astype(np.float64)],
                np.array([cx, cy, w, h, angle]),
                **SINGLE,
            )
        )
        box = cv2.boxPoints(((cx, cy), (w, h), angle))
        out.append(
            case(
                f"boxPoints/{name}",
                "boxPoints",
                [np.array([cx, cy, w, h, angle])],
                box.astype(np.float64).ravel(),
                **SINGLE,
            )
        )
        (ex, ey), r = cv2.minEnclosingCircle(pts)
        out.append(
            case(
                f"minEnclosingCircle/{name}",
                "minEnclosingCircle",
                [pts.astype(np.float64)],
                np.array([ex, ey, r]),
                rtol=1e-6,
                atol=2e-5,
            )
        )
    for k, pts in enumerate(ellipses):
        (cx, cy), (w, h), angle = cv2.fitEllipse(pts)
        out.append(
            case(f"fitEllipse/{k}", "fitEllipse", [pts.astype(np.float64)], np.array([cx, cy, w, h, angle]), **SINGLE)
        )
    for name in ["ints", "floats"]:
        pts = clouds[name]
        line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        out.append(case(f"fitLine/{name}", "fitLine", [pts.astype(np.float64)], line.astype(np.float64), **SINGLE))
    xs = np.arange(20, dtype=np.float64)
    noisy = np.stack([xs, 0.5 * xs + 3 + 0.1 * g.normal(size=20)], 1).astype(np.float32)
    line = cv2.fitLine(noisy, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    out.append(case("fitLine/noisy", "fitLine", [noisy.astype(np.float64)], line.astype(np.float64), **SINGLE))
    return out
