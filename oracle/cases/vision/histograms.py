"""Histograms against OpenCV (calcHist, calcBackProject, equalizeHist, CLAHE, normalize, compareHist) and scikit-image
(match_histograms, rescale_intensity). The 8-bit operations run on uint8 images, seen by the engine as levels / 255."""

from collections.abc import Callable
from importlib.metadata import version
from typing import cast

import cv2
import numpy as np
import skimage.exposure as E
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.signal_helpers import smooth_image

REFERENCE: dict[str, str] = {
    "opencv-python-headless": version("opencv-python-headless"),
    "scikit-image": version("scikit-image"),
}

# scikit-image's inferred signature types the ranges as strings (from their default, 'image').
rescale_intensity = cast(Callable[..., FloatArray], E.rescale_intensity)

EXACT = {"rtol": 1e-12, "atol": 1e-12}

COMPARISONS: dict[str, int] = {
    "correlation": cv2.HISTCMP_CORREL,
    "chi-square": cv2.HISTCMP_CHISQR,
    "chi-square-alt": cv2.HISTCMP_CHISQR_ALT,
    "intersection": cv2.HISTCMP_INTERSECT,
    "bhattacharyya": cv2.HISTCMP_BHATTACHARYYA,
    "kl-divergence": cv2.HISTCMP_KL_DIV,
}

NORMS: dict[str, int] = {
    "minmax": cv2.NORM_MINMAX,
    "l1": cv2.NORM_L1,
    "l2": cv2.NORM_L2,
    "inf": cv2.NORM_INF,
}


def levels(img: FloatArray) -> np.ndarray:
    """A [0, 1] image as uint8 levels."""
    return np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)


def cases() -> list[Case]:
    g = rng(4310)
    out: list[Case] = []
    # Values below 1 and float32-representable: OpenCV's ranges are half-open and it bins in single precision inputs.
    img = g.uniform(0, 0.999, (12, 14, 3)).astype(np.float32).astype(np.float64)
    grey = img[..., 0].copy()
    h1 = cv2.calcHist([grey.astype(np.float32)], [0], None, [16], [0, 1])
    out.append(case("calcHist/1d", "calcHist", [grey], h1.ravel(), **EXACT, bins=16))
    h2 = cv2.calcHist([img.astype(np.float32)], [0, 2], None, [8, 6], [0, 1, 0, 1])
    out.append(case("calcHist/2d", "calcHist", [img], h2, **EXACT, channels=[0, 2], bins=[8, 6]))
    mask = (g.uniform(size=(12, 14)) > 0.4).astype(np.uint8)
    hm = cv2.calcHist([grey.astype(np.float32)], [0], mask, [10], [0, 1])
    out.append(case("calcHist/mask", "calcHist", [grey, mask.astype(np.float64)], hm.ravel(), **EXACT, bins=10))
    norm = (h2 / h2.sum()).astype(np.float32)
    bp = cv2.calcBackProject([img.astype(np.float32)], [0, 2], norm, [0, 1, 0, 1], scale=2)
    out.append(
        case(
            "calcBackProject",
            "calcBackProject",
            [img, norm.astype(np.float64)],
            bp,
            rtol=1e-6,
            atol=1e-7,
            channels=[0, 2],
            scale=2,
        )
    )
    smooth = smooth_image(g, 20, 24)
    u8 = levels(smooth * 0.6 + 0.2)
    out.append(case("equalizeHist", "equalizeHist", [u8 / 255.0], cv2.equalizeHist(u8) / 255.0, **EXACT))
    flat = np.full((4, 5), 77, np.uint8)
    out.append(case("equalizeHist/flat", "equalizeHist", [flat / 255.0], cv2.equalizeHist(flat) / 255.0, **EXACT))
    for name, shape, clip, tiles in [
        ("divisible", (32, 32), 2.0, (4, 4)),
        ("ragged", (30, 37), 40.0, (8, 8)),
        ("unclipped", (24, 20), 0.0, (3, 2)),
        ("wide", (24, 21), 3.0, (3, 3)),
    ]:
        src = levels(smooth_image(g, *shape) * 0.5 + 0.25 * g.uniform(size=shape))
        eq = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles).apply(src)
        out.append(
            case(
                f"clahe/{name}",
                "clahe",
                [src / 255.0],
                eq / 255.0,
                **EXACT,
                clipLimit=clip,
                tileGridSize=list(tiles),
            )
        )
    ref = g.gamma(2.0, 0.2, (9, 11))
    out.append(case("matchHistograms", "matchHistograms", [grey, ref], E.match_histograms(grey, ref), **EXACT))
    refc = g.uniform(0, 1, (8, 9, 3)) ** 2
    out.append(
        case(
            "matchHistograms/rgb",
            "matchHistograms",
            [img, refc],
            E.match_histograms(img, refc, channel_axis=-1),
            **EXACT,
        )
    )
    out.append(
        case(
            "rescaleIntensity",
            "rescaleIntensity",
            [grey],
            rescale_intensity(grey, in_range=(0.2, 0.7), out_range=(0.0, 1.0)),
            **EXACT,
            inRange=[0.2, 0.7],
        )
    )
    p2, p98 = np.percentile(grey, (2, 98))
    stretched = rescale_intensity(grey, in_range=(float(p2), float(p98)), out_range=(0.0, 1.0))
    out.append(case("contrastStretch", "contrastStretch", [grey], stretched, **EXACT, low=2, high=98))
    signed = g.normal(size=(5, 6))
    for name, code in NORMS.items():
        alpha, beta = (0.2, 0.9) if name == "minmax" else (2.0, 0.0)
        res = cv2.normalize(signed, np.zeros_like(signed), alpha, beta, code)
        out.append(
            case(
                f"normalize/{name}",
                "normalize",
                [signed],
                np.asarray(res, dtype=np.float64),
                **EXACT,
                normType=name,
                alpha=alpha,
                beta=beta,
            )
        )
    ha = cv2.calcHist([levels(smooth)], [0], None, [32], [0, 256]).ravel()
    hb = cv2.calcHist([levels(smooth**1.5)], [0], None, [32], [0, 256]).ravel()
    for name, code in COMPARISONS.items():
        out.append(
            case(
                f"compareHist/{name}",
                "compareHist",
                [ha.astype(np.float64), hb.astype(np.float64)],
                np.atleast_1d(cv2.compareHist(ha, hb, code)),
                rtol=1e-10,
                atol=1e-12,
                method=name,
            )
        )
    return out
