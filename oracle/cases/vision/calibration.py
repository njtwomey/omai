"""Calibration against OpenCV: calibrateCamera on noisy views of a planar board (both refine the intrinsics, the five
distortion coefficients and the poses to the same least-squares optimum; agreement to about 1e-5), and stereoRectify's
rectifying rotations."""

from importlib.metadata import version

import cv2
import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}


def cases() -> list[Case]:
    # One thread: OpenCV's parallel accumulation in the calibration solver is otherwise not reproducible.
    cv2.setNumThreads(1)
    g = rng(4430)
    out: list[Case] = []
    k = np.array([[400.0, 0, 162], [0, 395.0, 118], [0, 0, 1]])
    d = np.array([-0.15, 0.04, 0.002, -0.001, 0.0])
    board = np.array([[i % 7, i // 7, 0] for i in range(35)], np.float64) * 0.05
    poses = [
        (np.array([0.2, 0.1, 0.05]), np.array([-0.15, -0.1, 0.9])),
        (np.array([-0.25, 0.2, -0.1]), np.array([-0.2, -0.05, 1.0])),
        (np.array([0.1, -0.3, 0.2]), np.array([-0.1, -0.12, 0.8])),
        (np.array([0.35, 0.25, -0.05]), np.array([-0.18, -0.15, 1.1])),
    ]
    obj, img = [], []
    for rv, tv in poses:
        p, _ = cv2.projectPoints(board, rv, tv, k, d)
        obj.append(board.astype(np.float32))
        img.append((p.reshape(-1, 2) + g.normal(0, 0.1, (len(board), 2))).astype(np.float32))
    for name, flags in [("k3", 0), ("fixk3", cv2.CALIB_FIX_K3)]:
        rms, km, dist, _, _ = cv2.calibrateCamera(
            obj,
            img,
            (320, 240),
            None,
            None,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-14),
        )
        enc = np.concatenate([[km[0, 0], km[1, 1], km[0, 2], km[1, 2]], dist.ravel()[:5], [rms]])
        out.append(
            case(
                f"calibrateCamera/{name}",
                "calibrateCamera",
                [np.stack(obj).astype(np.float64), np.stack(img).astype(np.float64)],
                enc,
                rtol=1e-4,
                atol=1e-5,
                fixK3=flags != 0,
            )
        )
    k2 = np.array([[410.0, 0, 158], [0, 405.0, 121], [0, 0, 1]])
    for name, rv, t in [
        ("horizontal", np.array([0.01, -0.02, 0.005]), np.array([-0.12, 0.002, 0.004])),
        ("vertical", np.array([0.02, 0.01, -0.01]), np.array([0.003, -0.1, 0.002])),
    ]:
        R, _ = cv2.Rodrigues(rv)
        R1, R2, *_ = cv2.stereoRectify(k, d, k2, d, (320, 240), R, t.reshape(3, 1))
        out.append(
            case(
                f"stereoRectify/{name}",
                "stereoRectify",
                [k, d, k2, d, R, t],
                np.concatenate([R1.ravel(), R2.ravel()]),
                rtol=1e-9,
                atol=1e-10,
                size=[320, 240],
            )
        )
    return out
