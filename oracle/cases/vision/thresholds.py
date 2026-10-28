"""Thresholds against OpenCV (threshold's five rules, the triangle method and adaptiveThreshold on uint8 images, seen
by the engine as levels / 255) and scikit-image (multi-level Otsu, Niblack and Sauvola)."""

from collections.abc import Callable
from importlib.metadata import version
from typing import cast

import cv2
import numpy as np
import skimage.filters as F
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.signal_helpers import blocks_image, smooth_image

REFERENCE: dict[str, str] = {
    "opencv-python-headless": version("opencv-python-headless"),
    "scikit-image": version("scikit-image"),
}

EXACT = {"rtol": 1e-12, "atol": 1e-12}

RULES: dict[str, int] = {
    "binary": cv2.THRESH_BINARY,
    "binary-inv": cv2.THRESH_BINARY_INV,
    "trunc": cv2.THRESH_TRUNC,
    "tozero": cv2.THRESH_TOZERO,
    "tozero-inv": cv2.THRESH_TOZERO_INV,
}

# scikit-image's inferred signatures type the window size as int and k as float from their defaults.
threshold_niblack = cast(Callable[..., FloatArray], F.threshold_niblack)
threshold_sauvola = cast(Callable[..., FloatArray], F.threshold_sauvola)


def levels(img: FloatArray) -> np.ndarray:
    """A [0, 1] image as uint8 levels."""
    return np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)


def cases() -> list[Case]:
    g = rng(4330)
    out: list[Case] = []
    img = g.uniform(0, 1, (7, 9)).astype(np.float32)
    for name, code in RULES.items():
        _, res = cv2.threshold(img, 0.375, 0.75, code)
        out.append(
            case(
                f"threshold/{name}",
                "threshold",
                [img.astype(np.float64)],
                res.astype(np.float64),
                **EXACT,
                thresh=0.375,
                maxValue=0.75,
                type=name,
            )
        )
    noisy = blocks_image(24, 24) * 0.7 + 0.15 + 0.05 * g.normal(size=(24, 24))
    skewed = smooth_image(g, 20, 20) ** 3
    for name, src in [("blocks", noisy), ("skewed", skewed), ("dark", 1 - skewed)]:
        u8 = levels(src)
        t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        out.append(case(f"thresholdTriangle/{name}", "thresholdTriangle", [u8 / 255.0], np.atleast_1d(t), **EXACT))
    page = levels(smooth_image(g, 26, 31) * 0.6 + 0.3 * (g.uniform(size=(26, 31)) > 0.8))
    for method, mcode in [("mean", cv2.ADAPTIVE_THRESH_MEAN_C), ("gaussian", cv2.ADAPTIVE_THRESH_GAUSSIAN_C)]:
        for block, c in [(5, 2.0), (11, -3.0), (9, 0.5)]:
            for name, tcode in [("binary", cv2.THRESH_BINARY), ("binary-inv", cv2.THRESH_BINARY_INV)]:
                res = cv2.adaptiveThreshold(page, 255, mcode, tcode, block, c)
                out.append(
                    case(
                        f"adaptiveThreshold/{method}/{block}/{c}/{name}",
                        "adaptiveThreshold",
                        [page / 255.0],
                        res / 255.0,
                        **EXACT,
                        method=method,
                        blockSize=block,
                        C=c / 255,
                        type=name,
                    )
                )
    modes = np.concatenate([g.normal(0.2, 0.04, 150), g.normal(0.5, 0.05, 150), g.normal(0.85, 0.03, 150)])
    trimodal = np.clip(modes, 0, 1).reshape(18, 25)
    for classes, nbins in [(3, 256), (4, 64), (2, 32)]:
        th = F.threshold_multiotsu(trimodal, classes=classes, nbins=nbins)
        out.append(
            case(
                f"thresholdMultiOtsu/{classes}/{nbins}",
                "thresholdMultiOtsu",
                [trimodal],
                th,
                **EXACT,
                classes=classes,
                nbins=nbins,
            )
        )
    doc = smooth_image(g, 16, 18)
    for w, k in [(5, 0.2), (9, -0.1)]:
        out.append(
            case(
                f"thresholdNiblack/{w}",
                "thresholdNiblack",
                [doc],
                threshold_niblack(doc, window_size=w, k=k),
                rtol=1e-9,
                atol=1e-10,
                windowSize=w,
                k=k,
            )
        )
        out.append(
            case(
                f"thresholdSauvola/{w}",
                "thresholdSauvola",
                [doc],
                threshold_sauvola(doc, window_size=w, k=k),
                rtol=1e-9,
                atol=1e-10,
                windowSize=w,
                k=k,
            )
        )
    return out
