"""Haar-like features against scikit-image's haar_like_feature: every type and all types at once, on windows of several
sizes. The windows sit at the origin of each image: scikit-image 0.26 mis-sums rectangles touching the top or left of a
window placed away from the origin (it tests the relative, not the absolute, corner before subtracting), so offsets are
checked in the engine's tests by cropping instead."""

from importlib.metadata import version

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from skimage.feature import haar_like_feature
from skimage.transform import integral_image

REFERENCE: dict[str, str] = {"scikit-image": version("scikit-image")}

TYPES = ["type-2-x", "type-2-y", "type-3-x", "type-3-y", "type-4"]


def cases() -> list[Case]:
    g = rng(4510)
    out: list[Case] = []
    img = g.uniform(size=(10, 12))
    for r, c, width, height in [(0, 0, 6, 6), (2, 3, 5, 7), (4, 1, 8, 4)]:
        ii = integral_image(img[r:, c:])
        for t in TYPES:
            out.append(
                case(
                    f"haarLikeFeature/{t}/{r}{c}{width}{height}",
                    "haarLikeFeature",
                    [ii],
                    np.asarray(haar_like_feature(ii, 0, 0, width, height, t), dtype=np.float64),
                    rtol=1e-12,
                    atol=1e-12,
                    width=width,
                    height=height,
                    types=[t],
                )
            )
        out.append(
            case(
                f"haarLikeFeature/all/{r}{c}{width}{height}",
                "haarLikeFeature",
                [ii],
                np.asarray(haar_like_feature(ii, 0, 0, width, height), dtype=np.float64),
                rtol=1e-12,
                atol=1e-12,
                width=width,
                height=height,
                types=TYPES,
            )
        )
    return out
