"""Regions against OpenCV: connectedComponentsWithStats (4- and 8-connected; labels, stats and centroids) and floodFill
on float32 images (floating and fixed ranges, both connectivities, colour images and a barrier mask)."""

from collections.abc import Sequence
from importlib.metadata import version

import cv2
import numpy as np
import scipy.ndimage as N
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.signal_helpers import smooth_image

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}

EXACT = {"rtol": 1e-12, "atol": 1e-12}


def encode_fill(area: int, img: np.ndarray, mask: np.ndarray, rect: Sequence[int]) -> FloatArray:
    filled = (mask[1:-1, 1:-1] > 0).astype(np.float64)
    return np.concatenate([[float(area)], img.astype(np.float64).ravel(), filled.ravel(), np.asarray(rect, float)])


def cases() -> list[Case]:
    g = rng(4380)
    out: list[Case] = []
    binary = (N.gaussian_filter(g.normal(size=(18, 21)), 1.2) > 0.1).astype(np.uint8)
    for conn in [4, 8]:
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=conn)
        enc = np.concatenate(
            [[float(n)], labels.astype(np.float64).ravel(), stats.astype(np.float64).ravel(), centroids.ravel()]
        )
        out.append(
            case(
                f"connectedComponentsWithStats/{conn}",
                "connectedComponentsWithStats",
                [binary.astype(np.float64)],
                enc,
                rtol=1e-12,
                atol=1e-12,
                connectivity=conn,
            )
        )
    img = smooth_image(g, 16, 19).astype(np.float32)
    rgb = np.stack([img, img[::-1], img[:, ::-1]], -1).astype(np.float32)
    barrier = np.zeros((16, 19), np.uint8)
    barrier[:, 9] = 1
    settings = [
        ("floating4", img, (8, 7), 0.03, 0.03, 4, False, None),
        ("floating8", img, (8, 7), 0.02, 0.04, 8, False, None),
        ("fixed4", img, (3, 3), 0.1, 0.1, 4, True, None),
        ("fixed8", img, (15, 12), 0.15, 0.05, 8, True, None),
        ("rgb", rgb, (8, 7), 0.05, 0.05, 4, False, None),
        ("barrier", img, (3, 8), 0.2, 0.2, 8, False, barrier),
    ]
    for name, src, seed, lo, up, conn, fixed, bar in settings:
        mask = np.zeros((18, 21), np.uint8)
        if bar is not None:
            mask[1:-1, 1:-1] = bar
        flags = conn | (cv2.FLOODFILL_FIXED_RANGE if fixed else 0)
        work = src.copy()
        lo_s = (lo, lo, lo) if work.ndim == 3 else lo
        up_s = (up, up, up) if work.ndim == 3 else up
        new_val = (0.75, 0.75, 0.75) if work.ndim == 3 else 0.75
        area, filled, mask_out, rect = cv2.floodFill(work, mask, seed, new_val, lo_s, up_s, flags)
        if bar is not None:
            mask_out[1:-1, 1:-1][bar > 0] = 0
        inputs: list[FloatArray] = [src.astype(np.float64)]
        if bar is not None:
            inputs.append(bar.astype(np.float64))
        out.append(
            case(
                f"floodFill/{name}",
                "floodFill",
                inputs,
                encode_fill(area, filled, mask_out, rect),
                **EXACT,
                seed=list(seed),
                loDiff=lo,
                upDiff=up,
                connectivity=conn,
                fixedRange=fixed,
            )
        )
    return out
