"""Window trackers against OpenCV: meanShift and CamShift on float32 probability images (blurred blobs, elongated and
rotated), from several starting windows."""

from importlib.metadata import version

import cv2
import numpy as np
import scipy.ndimage as N
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}

CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1.0)


def cases() -> list[Case]:
    g = rng(4460)
    out: list[Case] = []
    yy, xx = np.mgrid[:40, :48]
    a = np.deg2rad(30)
    u = (xx - 30) * np.cos(a) + (yy - 22) * np.sin(a)
    v = -(xx - 30) * np.sin(a) + (yy - 22) * np.cos(a)
    blob = np.exp(-(u**2) / (2 * 6**2) - v**2 / (2 * 2.5**2))
    prob = (blob + 0.02 * N.gaussian_filter(g.uniform(size=blob.shape), 2)).astype(np.float32)
    for k, window in enumerate([(12, 8, 10, 10), (22, 14, 8, 12), (30, 20, 6, 6)]):
        iters, win = cv2.meanShift(prob, window, CRITERIA)
        out.append(
            case(
                f"meanShift/{k}",
                "meanShift",
                [prob.astype(np.float64)],
                np.asarray([*win, iters], np.float64),
                rtol=1e-12,
                atol=1e-12,
                window=list(window),
            )
        )
        (cx, cy), (w, h), angle = cv2.CamShift(prob, window, CRITERIA)[0]
        _, nwin = cv2.CamShift(prob, window, CRITERIA)
        out.append(
            case(
                f"camShift/{k}",
                "camShift",
                [prob.astype(np.float64)],
                np.asarray([cx, cy, w, h, angle, *nwin], np.float64),
                rtol=1e-6,
                atol=1e-5,
                window=list(window),
            )
        )
    return out
