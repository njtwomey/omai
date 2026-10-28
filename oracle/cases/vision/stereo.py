"""3-D reprojection of disparity against OpenCV's reprojectImageTo3D (float32 disparity and points)."""

from importlib.metadata import version

import cv2
import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}


def cases() -> list[Case]:
    g = rng(4500)
    disp = g.uniform(1, 16, (6, 8)).astype(np.float32)
    Q = np.array([[1, 0, 0, -4.0], [0, 1, 0, -3.0], [0, 0, 0, 120.0], [0, 0, 1 / 0.1, 0]])
    pts = cv2.reprojectImageTo3D(disp, Q)
    return [
        case(
            "reprojectImageTo3D",
            "reprojectImageTo3D",
            [disp.astype(np.float64), Q],
            pts.astype(np.float64),
            rtol=1e-6,
            atol=1e-6,
        )
    ]
