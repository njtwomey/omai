"""FAST-9 corners against OpenCV's FastFeatureDetector on uint8 images (seen by the engine as levels / 255): keypoint
positions and responses, with and without non-maximum suppression."""

from collections.abc import Sequence
from importlib.metadata import version

import cv2
import numpy as np
import scipy.ndimage as N
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}


def encode(kps: Sequence[cv2.KeyPoint]) -> FloatArray:
    return np.asarray([[k.pt[0], k.pt[1], k.response] for k in kps], dtype=np.float64).reshape(-1, 3)


def cases() -> list[Case]:
    g = rng(4440)
    out: list[Case] = []
    i, j = np.mgrid[:40, :44]
    images = {
        "noise": N.gaussian_filter(g.uniform(0, 255, (40, 44)), 1.2),
        "texture": (np.sin(i * i * 0.05 + j * 0.9) * np.cos(j * j * 0.03 - i * 0.4) * 0.5 + 0.5) * 255,
    }
    for name, img in images.items():
        u8 = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        for threshold in [5, 20]:
            for nms in [True, False]:
                kps = cv2.FastFeatureDetector.create(threshold, nms, cv2.FastFeatureDetector_TYPE_9_16).detect(u8)
                out.append(
                    case(
                        f"fast/{name}/{threshold}/{nms}",
                        "fast",
                        [u8 / 255.0],
                        encode(kps),
                        rtol=1e-12,
                        atol=1e-12,
                        threshold=threshold / 255,
                        nonmaxSuppression=nms,
                    )
                )
    return out
