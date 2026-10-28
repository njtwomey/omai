"""Line segments against OpenCV's LineSegmentDetector with LSD_REFINE_NONE at scale 1 on uint8 images (seen by the
engine as levels / 255): drawn lines and polygons, blurred and lightly noisy, at two angle tolerances."""

from importlib.metadata import version

import cv2
import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}


def scene(seed: int) -> FloatArray:
    g = rng(seed)
    img = np.full((48, 64), 40, np.uint8)
    cv2.rectangle(img, (8, 6), (30, 26), 200, -1)
    cv2.line(img, (36, 40), (60, 8), 230, 2)
    pts = np.array([[10, 44], [28, 32], [34, 46]], np.int32)
    cv2.fillPoly(img, [pts], 120)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    noisy = np.clip(img.astype(np.float64) + g.normal(0, 2, img.shape), 0, 255)
    return np.round(noisy).astype(np.uint8).astype(np.float64)


def cases() -> list[Case]:
    out: list[Case] = []
    for seed in [4520, 4521]:
        img = scene(seed)
        for tol in [22.5, 30.0]:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE, 1.0, 0.6, 2.0, tol)
            lines, widths, _, _ = lsd.detect(img.astype(np.uint8))
            res = np.hstack([np.asarray(lines).reshape(-1, 4), np.asarray(widths).reshape(-1, 1)])
            out.append(
                case(
                    f"lineSegmentDetector/{seed}/{tol}",
                    "lineSegmentDetector",
                    [img / 255],
                    res.astype(np.float64),
                    rtol=1e-5,
                    atol=1e-4,
                    angleTolerance=tol,
                )
            )
    return out
