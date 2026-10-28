"""Contours against OpenCV: findContours in every retrieval mode with both chain approximations (encoded as the contour
count, each contour's length and points, then the hierarchy), approxPolyDP, arcLength, contourArea, boundingRect,
isContourConvex and pointPolygonTest, all on integer contours where OpenCV's arithmetic is exact."""

from collections.abc import Sequence
from importlib.metadata import version

import cv2
import numpy as np
import numpy.typing as npt
import scipy.ndimage as N
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}

EXACT = {"rtol": 1e-12, "atol": 1e-12}

MODES: dict[str, int] = {
    "external": cv2.RETR_EXTERNAL,
    "list": cv2.RETR_LIST,
    "ccomp": cv2.RETR_CCOMP,
    "tree": cv2.RETR_TREE,
}
METHODS: dict[str, int] = {"none": cv2.CHAIN_APPROX_NONE, "simple": cv2.CHAIN_APPROX_SIMPLE}


def shapes() -> np.ndarray:
    """Nested shapes: a ring with an island and a hole in the island, a diagonal bar, a pixel and a border blob."""
    img = np.zeros((24, 26), np.uint8)
    img[2:16, 2:16] = 1
    img[4:14, 4:14] = 0
    img[6:12, 6:12] = 1
    img[8:10, 8:10] = 0
    for k in range(8):
        img[16 + k // 2, 17 + k] = 1
        img[17 + k // 2, 17 + k] = 1
    img[20, 3] = 1
    img[0:5, 21:26] = 1
    img[2, 23] = 0
    return img


def blobs(g: np.random.Generator) -> np.ndarray:
    """A random binary image of smooth blobs with holes."""
    noise = N.gaussian_filter(g.normal(size=(30, 32)), 2.0)
    return (noise > 0.05).astype(np.uint8)


def encode(contours: Sequence[npt.ArrayLike], hierarchy: npt.ArrayLike | None) -> FloatArray:
    parts: list[float] = [float(len(contours))]
    for c in contours:
        pts = np.asarray(c).reshape(-1, 2)
        parts.append(float(len(pts)))
        parts.extend(pts.astype(np.float64).ravel().tolist())
    if hierarchy is not None:
        parts.extend(np.asarray(hierarchy, dtype=np.float64).reshape(-1).tolist())
    return np.asarray(parts, dtype=np.float64)


def cases() -> list[Case]:
    g = rng(4340)
    out: list[Case] = []
    images = {
        "shapes": shapes(),
        "blobs": blobs(g),
        "full": np.ones((5, 6), np.uint8),
        "empty": np.zeros((4, 4), np.uint8),
    }
    for iname, img in images.items():
        for mname, mode in MODES.items():
            for aname, method in METHODS.items():
                contours, hierarchy = cv2.findContours(img, mode, method)
                out.append(
                    case(
                        f"findContours/{iname}/{mname}/{aname}",
                        "findContours",
                        [img.astype(np.float64)],
                        encode(contours, hierarchy),
                        **EXACT,
                        mode=mname,
                        method=aname,
                    )
                )
    contours, _ = cv2.findContours(blobs(g), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    big = sorted(contours, key=lambda c: -len(c))[:3]
    for k, c in enumerate(big):
        pts = c.reshape(-1, 2).astype(np.float64)
        for eps in [1.0, 2.5]:
            for closed in [True, False]:
                approx = cv2.approxPolyDP(c, eps, closed).reshape(-1, 2)
                out.append(
                    case(
                        f"approxPolyDP/{k}/{eps}/{closed}",
                        "approxPolyDP",
                        [pts],
                        approx.astype(np.float64),
                        **EXACT,
                        epsilon=eps,
                        closed=closed,
                    )
                )
        for closed in [True, False]:
            out.append(
                case(
                    f"arcLength/{k}/{closed}",
                    "arcLength",
                    [pts],
                    np.atleast_1d(cv2.arcLength(c, closed)),
                    rtol=1e-12,
                    atol=1e-12,
                    closed=closed,
                )
            )
        for oriented in [True, False]:
            out.append(
                case(
                    f"contourArea/{k}/{oriented}",
                    "contourArea",
                    [pts],
                    np.atleast_1d(cv2.contourArea(c, oriented)),
                    **EXACT,
                    oriented=oriented,
                )
            )
        out.append(case(f"boundingRect/{k}", "boundingRect", [pts], np.asarray(cv2.boundingRect(c), float), **EXACT))
        out.append(
            case(
                f"isContourConvex/{k}", "isContourConvex", [pts], np.atleast_1d(float(cv2.isContourConvex(c))), **EXACT
            )
        )
    poly = np.array([[2, 1], [12, 3], [14, 11], [7, 14], [5, 8], [1, 9]], np.int32)
    hull = cv2.convexHull(poly)
    out.append(
        case("isContourConvex/hull", "isContourConvex", [hull.reshape(-1, 2).astype(np.float64)], np.ones(1), **EXACT)
    )
    queries = np.array(
        [[6.0, 6.0], [2.0, 1.0], [7.0, 2.0], [0.0, 0.0], [13.0, 7.0], [4.5, 8.5], [12.0, 3.0], [20.0, 5.0]]
    )
    for measure in [False, True]:
        res = [cv2.pointPolygonTest(poly, (float(q[0]), float(q[1])), measure) for q in queries]
        out.append(
            case(
                f"pointPolygonTest/{measure}",
                "pointPolygonTest",
                [poly.astype(np.float64), queries],
                np.asarray(res, float),
                rtol=1e-12,
                atol=1e-12,
                measureDist=measure,
            )
        )
    return out
