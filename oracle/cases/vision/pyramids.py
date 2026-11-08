"""Pyramids against OpenCV on float64 images: pyrDown (two borders), pyrUp (to twice the size and to odd sizes),
buildPyramid, and the Laplacian pyramid as OpenCV's samples compute it (each level minus pyrUp of the next)."""

from importlib.metadata import version

import cv2
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}

TOL = {"rtol": 1e-10, "atol": 1e-12}


def cases() -> list[Case]:
    g = rng(4490)
    out: list[Case] = []
    for shape in [(12, 16), (13, 9), (7, 10, 3)]:
        img = g.uniform(size=shape)
        name = "x".join(map(str, shape))
        out.append(case(f"pyrDown/{name}", "pyrDown", [img], cv2.pyrDown(img), **TOL, border="reflect101"))
        out.append(
            case(
                f"pyrDown/{name}/replicate",
                "pyrDown",
                [img],
                cv2.pyrDown(img, borderType=cv2.BORDER_REPLICATE),
                **TOL,
                border="replicate",
            )
        )
        out.append(case(f"pyrUp/{name}", "pyrUp", [img], cv2.pyrUp(img), **TOL))
        h, w = shape[0], shape[1]
        for dh, dw in [(-1, -1), (1, 0), (0, 1)]:
            size = (2 * h + dh, 2 * w + dw)
            out.append(
                case(
                    f"pyrUp/{name}/{dh}{dw}",
                    "pyrUp",
                    [img],
                    cv2.pyrUp(img, dstsize=(size[1], size[0])),
                    **TOL,
                    size=list(size),
                )
            )
    img = g.uniform(size=(20, 18))
    # cv2.buildPyramid (not bound in Python) is successive pyrDown calls.
    levels = [img]
    for _ in range(3):
        levels.append(np.asarray(cv2.pyrDown(levels[-1]), dtype=np.float64))
    out.append(
        case("buildPyramid", "buildPyramid", [img], np.concatenate([lv.ravel() for lv in levels]), **TOL, maxLevel=3)
    )
    lap = [levels[i] - cv2.pyrUp(levels[i + 1], dstsize=(levels[i].shape[1], levels[i].shape[0])) for i in range(3)]
    lap.append(levels[3])
    out.append(
        case("laplacianPyramid", "laplacianPyramid", [img], np.concatenate([lv.ravel() for lv in lap]), **TOL, levels=3)
    )
    return out
