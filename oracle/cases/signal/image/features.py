"""Image features against scikit-image: structure tensors, Harris and Shi-Tomasi corners, peak_local_max, Canny
edges, local binary patterns, HOG, and the line and circle Hough transforms."""

import warnings
from collections.abc import Callable
from importlib.metadata import version
from typing import cast

import numpy as np
import numpy.typing as npt
import skimage.feature as F
import skimage.transform as T
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.signal_helpers import blocks_image, smooth_image

REFERENCE: dict[str, str] = {"scikit-image": version("scikit-image")}

# scikit-image's inferred signatures type sigma as int (from its default).
structure_tensor = cast(Callable[..., list[FloatArray]], F.structure_tensor)
corner_harris = cast(Callable[..., FloatArray], F.corner_harris)

TOL = {"rtol": 1e-10, "atol": 1e-12}


def flat(*parts: npt.ArrayLike) -> FloatArray:
    return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts])


def cases() -> list[Case]:
    g = rng(490)
    out: list[Case] = []
    img = smooth_image(g, 14, 16)
    blocks = blocks_image(20, 20)
    for sigma in [1.0, 1.7]:
        out.append(
            case(
                f"structureTensor/{sigma}",
                "structureTensor",
                [img],
                flat(*structure_tensor(img, sigma=sigma, order="rc")),
                **TOL,
                sigma=sigma,
            )
        )
    out.append(
        case(
            "structureTensor/nearest",
            "structureTensor",
            [img],
            flat(*structure_tensor(img, mode="nearest", order="rc")),
            **TOL,
            mode="nearest",
        )
    )
    out.append(case("cornerHarris", "cornerHarris", [blocks], F.corner_harris(blocks), **TOL))
    out.append(
        case(
            "cornerHarris/eps",
            "cornerHarris",
            [blocks],
            corner_harris(blocks, method="eps", sigma=1.5),
            **TOL,
            method="eps",
            sigma=1.5,
        )
    )
    out.append(case("cornerShiTomasi", "cornerShiTomasi", [blocks], F.corner_shi_tomasi(blocks), **TOL))
    resp = F.corner_harris(blocks)
    for md in [1, 3]:
        pk = F.peak_local_max(resp, min_distance=md, threshold_rel=0.05)
        out.append(case(f"peakLocalMax/{md}", "peakLocalMax", [resp], flat(pk), minDistance=md, thresholdRel=0.05))
    pk = F.peak_local_max(img, min_distance=2, exclude_border=False, num_peaks=3)
    out.append(
        case("peakLocalMax/count", "peakLocalMax", [img], flat(pk), minDistance=2, excludeBorder=False, numPeaks=3)
    )
    for sigma in [1.0, 2.0]:
        out.append(case(f"canny/{sigma}", "canny", [blocks], F.canny(blocks, sigma=sigma).astype(float), sigma=sigma))
    noisy = blocks + 0.05 * g.normal(size=blocks.shape)
    out.append(
        case(
            "canny/quantiles",
            "canny",
            [noisy],
            F.canny(noisy, sigma=1.2, low_threshold=0.6, high_threshold=0.85, use_quantiles=True).astype(float),
            sigma=1.2,
            lowThreshold=0.6,
            highThreshold=0.85,
            useQuantiles=True,
        )
    )
    q = np.round(img * 20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ["default", "ror", "uniform"]:
            out.append(
                case(
                    f"localBinaryPattern/{method}",
                    "localBinaryPattern",
                    [q],
                    F.local_binary_pattern(q, 8, 1.0, method),
                    **TOL,
                    P=8,
                    R=1.0,
                    method=method,
                )
            )
        out.append(
            case(
                "localBinaryPattern/12",
                "localBinaryPattern",
                [q],
                F.local_binary_pattern(q, 12, 2.0, "default"),
                **TOL,
                P=12,
                R=2.0,
                method="default",
            )
        )
    big = smooth_image(g, 24, 32)
    # scikit-image's compiled cell histograms accumulate in single precision: agreement to 1e-7 relative.
    for norm in ["L1", "L1-sqrt", "L2", "L2-Hys"]:
        hv = F.hog(big, orientations=6, pixels_per_cell=(6, 8), cells_per_block=(2, 2), block_norm=norm)
        out.append(
            case(
                f"hog/{norm}",
                "hog",
                [big],
                hv,
                rtol=1e-6,
                atol=1e-8,
                orientations=6,
                pixelsPerCell=[6, 8],
                cellsPerBlock=[2, 2],
                blockNorm=norm,
            )
        )
    edges = F.canny(blocks, sigma=1.0).astype(float)
    acc, theta, dist = T.hough_line(edges)
    out.append(case("houghLine", "houghLine", [edges], flat(acc, theta, dist), **TOL))
    th = np.linspace(-1.2, 1.4, 7)
    acc, theta, dist = T.hough_line(edges, theta=th)
    out.append(case("houghLine/thetas", "houghLine", [edges], flat(acc, theta, dist), **TOL, thetas=th.tolist()))
    for normalize in [True, False]:
        hc = T.hough_circle(edges, [2, 3, 5], normalize=normalize)
        out.append(
            case(f"houghCircle/{normalize}", "houghCircle", [edges], hc, **TOL, radii=[2, 3, 5], normalize=normalize)
        )
    return out
