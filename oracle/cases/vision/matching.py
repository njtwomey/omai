"""Descriptor matching against OpenCV's BFMatcher: match (with and without cross-checking) and knnMatch under the L2
and L1 norms on float32 descriptors (distances in single precision) and the Hamming norm on byte descriptors."""

from collections.abc import Sequence
from importlib.metadata import version

import cv2
import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}

NORMS: dict[str, int] = {"l2": cv2.NORM_L2, "l1": cv2.NORM_L1, "hamming": cv2.NORM_HAMMING}


def encode(matches: Sequence[cv2.DMatch]) -> FloatArray:
    return np.asarray([[m.queryIdx, m.trainIdx, m.distance] for m in matches], dtype=np.float64).reshape(-1, 3)


def cases() -> list[Case]:
    g = rng(4450)
    out: list[Case] = []
    q = g.normal(size=(18, 16)).astype(np.float32)
    t = np.concatenate([q[::2] + 0.05 * g.normal(size=(9, 16)), g.normal(size=(14, 16))]).astype(np.float32)
    qb = g.integers(0, 256, (15, 32)).astype(np.uint8)
    tb = np.concatenate(
        [qb[::3] ^ (g.uniform(size=(5, 32)) > 0.9).astype(np.uint8), g.integers(0, 256, (12, 32))]
    ).astype(np.uint8)
    for name, code in NORMS.items():
        a, b = (qb, tb) if name == "hamming" else (q, t)
        tol = {"rtol": 1e-12, "atol": 1e-12} if name == "hamming" else {"rtol": 1e-5, "atol": 1e-5}
        for cross in [False, True]:
            m = cv2.BFMatcher(code, crossCheck=cross).match(a, b)
            out.append(
                case(
                    f"matchDescriptors/{name}/{cross}",
                    "matchDescriptors",
                    [a.astype(np.float64), b.astype(np.float64)],
                    encode(m),
                    **tol,
                    norm=name,
                    crossCheck=cross,
                )
            )
        knn = cv2.BFMatcher(code).knnMatch(a, b, k=3)
        out.append(
            case(
                f"knnMatch/{name}",
                "knnMatch",
                [a.astype(np.float64), b.astype(np.float64)],
                np.concatenate([encode(row) for row in knn]),
                **tol,
                norm=name,
                k=3,
            )
        )
    return out
