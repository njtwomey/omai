"""Colour conversions against OpenCV's cvtColor on float32 images (single precision: agreement to about 1e-6; OpenCV's
hue divides by the range plus FLT_EPSILON, its float Lab interpolates a lookup table) and against scikit-image's exact
HSV and Lab conversions."""

from importlib.metadata import version

import cv2
import numpy as np
import skimage.color as C
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {
    "opencv-python-headless": version("opencv-python-headless"),
    "scikit-image": version("scikit-image"),
}

SINGLE = {"rtol": 1e-6, "atol": 2e-6}
EXACT = {"rtol": 1e-10, "atol": 1e-12}
SKIMAGE_WHITE = [0.95047, 1.0, 1.08883]


def cv(x: FloatArray, code: int) -> FloatArray:
    """cvtColor on the float32 image, the result as float64."""
    return np.asarray(cv2.cvtColor(x.astype(np.float32), code), dtype=np.float64)


def cases() -> list[Case]:
    g = rng(4300)
    out: list[Case] = []
    # Float32-representable inputs, so both sides see the same numbers.
    rgb = g.uniform(0, 1, (6, 7, 3)).astype(np.float32).astype(np.float64)
    bright = g.uniform(0.25, 1, (5, 6, 3)).astype(np.float32).astype(np.float64)
    grey = rgb[..., 0].copy()
    out.append(case("rgbToGrey/601", "rgbToGrey", [rgb], cv(rgb, cv2.COLOR_RGB2GRAY), **SINGLE, standard="601"))
    out.append(
        case(
            "rgbToGrey/709",
            "rgbToGrey",
            [rgb],
            np.asarray(rgb @ np.array([0.2126, 0.7152, 0.0722])),
            **EXACT,
            standard="709",
        )
    )
    out.append(case("greyToRgb", "greyToRgb", [grey], cv(grey, cv2.COLOR_GRAY2RGB), **SINGLE))
    hsv = C.rgb2hsv(rgb)
    hsv[..., 0] *= 360
    out.append(case("rgbToHsv/skimage", "rgbToHsv", [rgb], hsv, **EXACT))
    out.append(case("rgbToHsv/opencv", "rgbToHsv", [rgb], cv(rgb, cv2.COLOR_RGB2HSV), rtol=1e-5, atol=1e-3))
    back = hsv.copy()
    back[..., 0] /= 360
    out.append(case("hsvToRgb/skimage", "hsvToRgb", [hsv], C.hsv2rgb(back), **EXACT))
    hsv32 = hsv.astype(np.float32).astype(np.float64)
    out.append(case("hsvToRgb/opencv", "hsvToRgb", [hsv32], cv(hsv32, cv2.COLOR_HSV2RGB), **SINGLE))
    out.append(case("rgbToXyz", "rgbToXyz", [rgb], cv(rgb, cv2.COLOR_RGB2XYZ), **SINGLE))
    xyz = cv(rgb, cv2.COLOR_RGB2XYZ).astype(np.float32).astype(np.float64)
    out.append(case("xyzToRgb", "xyzToRgb", [xyz], cv(xyz, cv2.COLOR_XYZ2RGB), **SINGLE))
    # scikit-image computes Lab exactly with the same rounded constants (and its own D65 white); away from the dark
    # toe, where OpenCV's 903.3 Y and scikit-image's 116 (7.787 Y + 16 / 116) - 16 differ, they agree exactly.
    out.append(
        case(
            "rgbToLab/skimage",
            "rgbToLab",
            [bright],
            C.rgb2lab(bright),
            rtol=1e-10,
            atol=1e-9,
            white=SKIMAGE_WHITE,
        )
    )
    # OpenCV's float Lab interpolates a lookup table: agreement to about 0.4 in a and b.
    out.append(case("rgbToLab/opencv", "rgbToLab", [rgb], cv(rgb, cv2.COLOR_RGB2Lab), rtol=0, atol=0.5))
    lab = cv(rgb, cv2.COLOR_RGB2Lab).astype(np.float32).astype(np.float64)
    out.append(case("labToRgb/opencv", "labToRgb", [lab], cv(lab, cv2.COLOR_Lab2RGB), rtol=0, atol=5e-3))
    out.append(case("rgbToYcrcb", "rgbToYcrcb", [rgb], cv(rgb, cv2.COLOR_RGB2YCrCb), **SINGLE))
    ycc = cv(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32).astype(np.float64)
    out.append(case("ycrcbToRgb", "ycrcbToRgb", [ycc], cv(ycc, cv2.COLOR_YCrCb2RGB), **SINGLE))
    return out
