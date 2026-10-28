"""Smoothing filters against OpenCV on float64 images (double precision throughout): getGaussianKernel, GaussianBlur,
blur, filter2D under several borders, and medianBlur (float32, the sizes OpenCV supports for floats)."""

from importlib.metadata import version

import cv2
import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from mdml_oracle.signal_helpers import smooth_image

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}

TOL = {"rtol": 1e-10, "atol": 1e-12}

BORDERS: dict[str, int] = {
    "reflect101": cv2.BORDER_REFLECT_101,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "constant": cv2.BORDER_CONSTANT,
}


def cases() -> list[Case]:
    g = rng(4320)
    out: list[Case] = []
    for k, sigma in [(1, 0.0), (3, 0.0), (5, 0.0), (7, 0.0), (9, 0.0), (5, 1.3), (4, 0.9)]:
        kern = cv2.getGaussianKernel(k, sigma).ravel()
        out.append(case(f"getGaussianKernel/{k}/{sigma}", "getGaussianKernel", [], kern, **TOL, ksize=k, sigma=sigma))
    img = smooth_image(g, 11, 13)
    rgb = g.uniform(0, 1, (9, 10, 3))
    for name, border in BORDERS.items():
        res = cv2.GaussianBlur(img, (5, 3), 1.2, sigmaY=0.8, borderType=border)
        out.append(
            case(
                f"gaussianBlur/{name}",
                "gaussianBlur",
                [img],
                res,
                **TOL,
                ksize=[5, 3],
                sigmaX=1.2,
                sigmaY=0.8,
                border=name,
            )
        )
    out.append(
        case(
            "gaussianBlur/derived",
            "gaussianBlur",
            [rgb],
            cv2.GaussianBlur(rgb, (0, 0), 0.9),
            **TOL,
            ksize=[0, 0],
            sigmaX=0.9,
            border="reflect101",
        )
    )
    out.append(
        case(
            "gaussianBlur/table",
            "gaussianBlur",
            [img],
            cv2.GaussianBlur(img, (7, 7), 0),
            **TOL,
            ksize=[7, 7],
            sigmaX=0,
            border="reflect101",
        )
    )
    for name, border in BORDERS.items():
        out.append(
            case(
                f"blur/{name}",
                "blur",
                [img],
                cv2.blur(img, (4, 3), borderType=border),
                **TOL,
                ksize=[4, 3],
                border=name,
            )
        )
    out.append(case("blur/rgb", "blur", [rgb], cv2.blur(rgb, (3, 3)), **TOL, ksize=[3, 3], border="reflect101"))
    kernel = g.normal(size=(3, 4))
    for name, border in BORDERS.items():
        out.append(
            case(
                f"filter2D/{name}",
                "filter2D",
                [img, kernel],
                cv2.filter2D(img, -1, kernel, borderType=border),
                **TOL,
                border=name,
            )
        )
    noisy = (img + 0.3 * (g.uniform(size=img.shape) > 0.9)).astype(np.float32)
    for k in [3, 5]:
        out.append(
            case(
                f"medianBlur/{k}",
                "medianBlur",
                [noisy.astype(np.float64)],
                cv2.medianBlur(noisy, k).astype(np.float64),
                **TOL,
                ksize=k,
            )
        )
    return out
