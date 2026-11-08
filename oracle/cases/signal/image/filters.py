"""Image filters against scipy.ndimage (Gaussian with derivatives, uniform, rank, median, minimum, maximum,
percentile, Laplace, Laplacian of Gaussian, Gaussian gradient magnitude) and scikit-image (Sobel, Scharr, Prewitt,
Gabor kernels and filters); the bilateral filter against its definition evaluated directly in numpy (scikit-image
bins the range weights)."""

from collections.abc import Callable
from importlib.metadata import version
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import scipy
import scipy.ndimage as N
import skimage.filters as F
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.signal_helpers import blocks_image, smooth_image

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "scikit-image": version("scikit-image")}

TOL = {"rtol": 1e-12, "atol": 1e-12}
# scikit-image's inferred signatures take theta, bandwidth and offset as int (from their defaults).
gabor_kernel = cast(Callable[..., npt.NDArray[np.complex128]], F.gabor_kernel)
gabor = cast(Callable[..., tuple[FloatArray, FloatArray]], F.gabor)

Mode = Literal["reflect", "mirror", "nearest", "wrap", "constant"]
MODES: list[Mode] = ["reflect", "mirror", "nearest", "wrap", "constant"]


def bilateral(img: FloatArray, win: int, sigma_color: float, sigma_spatial: float) -> FloatArray:
    """The bilateral filter by its definition, zero padding (as scikit-image's default mode)."""
    half = win // 2
    h, w = img.shape[:2]
    c = 1 if img.ndim == 2 else img.shape[2]
    x = img.reshape(h, w, c)
    pad = np.pad(x, ((half, half), (half, half), (0, 0)))
    out = np.zeros_like(x)
    dy, dx = np.mgrid[-half : half + 1, -half : half + 1]
    spatial = np.exp(-(dy**2 + dx**2) / (2 * sigma_spatial**2))
    for i in range(h):
        for j in range(w):
            win_v = pad[i : i + win, j : j + win]
            d2 = np.sum((win_v - x[i, j]) ** 2, axis=2)
            wt = spatial * np.exp(-d2 / (2 * sigma_color**2))
            out[i, j] = np.tensordot(wt, win_v, axes=([0, 1], [0, 1])) / wt.sum()
    return out.reshape(img.shape)


def cases() -> list[Case]:
    g = rng(430)
    out: list[Case] = []
    img = smooth_image(g, 9, 11)
    for sigma in [0.8, 1.7]:
        for order in [0, 1, 2, 3]:
            r = N.gaussian_filter1d(img, sigma, axis=1, order=order)
            out.append(
                case(
                    f"gaussianFilter1d/{sigma}/{order}",
                    "gaussianFilter1d",
                    [img],
                    r,
                    **TOL,
                    sigma=sigma,
                    order=order,
                    axis=1,
                )
            )
    for mode in MODES:
        r = N.gaussian_filter(img, [1.2, 0.7], mode=mode, cval=0.2)
        out.append(
            case(f"gaussianFilter/{mode}", "gaussianFilter", [img], r, **TOL, sigma=[1.2, 0.7], mode=mode, cval=0.2)
        )
    r = N.gaussian_filter(img, 1.0, order=[1, 2], truncate=2.5)
    out.append(case("gaussianFilter/orders", "gaussianFilter", [img], r, **TOL, sigma=1.0, order=[1, 2], truncate=2.5))
    r = N.gaussian_filter(img, 2.0, radius=2)
    out.append(case("gaussianFilter/radius", "gaussianFilter", [img], r, **TOL, sigma=2.0, radius=2))
    uniform_modes: list[Mode] = ["reflect", "constant"]
    for size in [3, 4]:
        for mode in uniform_modes:
            r = N.uniform_filter(img, size, mode=mode)
            out.append(case(f"uniformFilter/{size}/{mode}", "uniformFilter", [img], r, **TOL, size=size, mode=mode))
    r = N.uniform_filter(img, [3, 2], origin=[1, 0])
    out.append(case("uniformFilter/origin", "uniformFilter", [img], r, **TOL, size=[3, 2], origin=[1, 0]))
    noisy = img + 0.3 * (g.random(img.shape) > 0.9)
    for mode in MODES:
        params: dict[str, Any] = {"mode": mode, "cval": 0.4}
        out.append(
            case(
                f"medianFilter/{mode}",
                "medianFilter",
                [noisy],
                N.median_filter(noisy, 3, mode=mode, cval=0.4),
                **TOL,
                size=3,
                **params,
            )
        )
    out.append(case("medianFilter/rect", "medianFilter", [noisy], N.median_filter(noisy, (2, 4)), **TOL, size=[2, 4]))
    out.append(case("minimumFilter", "minimumFilter", [noisy], N.minimum_filter(noisy, (3, 2)), **TOL, size=[3, 2]))
    out.append(
        case("maximumFilter", "maximumFilter", [noisy], N.maximum_filter(noisy, 3, origin=1), **TOL, size=3, origin=1)
    )
    out.append(case("rankFilter", "rankFilter", [noisy], N.rank_filter(noisy, -2, 3), **TOL, rank=-2, size=3))
    for p in [10.0, 50.0, 90.0, 100.0]:
        out.append(
            case(
                f"percentileFilter/{p}",
                "percentileFilter",
                [noisy],
                N.percentile_filter(noisy, p, 3),
                **TOL,
                percentile=p,
                size=3,
            )
        )
    blocks = blocks_image(12, 12)
    for name, fn in [("sobel", F.sobel), ("scharr", F.scharr), ("prewitt", F.prewitt)]:
        out.append(case(f"{name}", name, [blocks], fn(blocks), **TOL))
        for axis in [0, 1]:
            out.append(case(f"{name}/{axis}", name, [blocks], fn(blocks, axis=axis), **TOL, axis=axis))
    out.append(case("sobel/constant", "sobel", [blocks], F.sobel(blocks, mode="constant"), **TOL, mode="constant"))
    out.append(case("laplace", "laplace", [img], N.laplace(img), **TOL))
    out.append(case("laplace/wrap", "laplace", [img], N.laplace(img, mode="wrap"), **TOL, mode="wrap"))
    for sigma in [1.0, 2.0]:
        out.append(
            case(
                f"gaussianLaplace/{sigma}", "gaussianLaplace", [img], N.gaussian_laplace(img, sigma), **TOL, sigma=sigma
            )
        )
        out.append(
            case(
                f"gaussianGradientMagnitude/{sigma}",
                "gaussianGradientMagnitude",
                [img],
                N.gaussian_gradient_magnitude(img, sigma),
                **TOL,
                sigma=sigma,
            )
        )
    for freq, theta, bw in [(0.2, 0.0, 1.0), (0.15, 0.6, 1.5), (0.3, 2.0, 0.8)]:
        k = gabor_kernel(freq, theta=theta, bandwidth=bw)
        out.append(
            case(
                f"gaborKernel/{freq}/{theta}",
                "gaborKernel",
                [],
                np.stack([k.real, k.imag]),
                **TOL,
                frequency=freq,
                theta=theta,
                bandwidth=bw,
            )
        )
    k = gabor_kernel(0.25, theta=0.3, sigma_x=2.0, sigma_y=1.0, n_stds=2, offset=0.5)
    out.append(
        case(
            "gaborKernel/sigmas",
            "gaborKernel",
            [],
            np.stack([k.real, k.imag]),
            **TOL,
            frequency=0.25,
            theta=0.3,
            sigmaX=2.0,
            sigmaY=1.0,
            nStds=2,
            offset=0.5,
        )
    )
    re, im = gabor(img, 0.3, theta=0.4)
    out.append(case("gabor", "gabor", [img], np.stack([re, im]), rtol=1e-10, atol=1e-12, frequency=0.3, theta=0.4))
    for sc, ss in [(0.1, 1.0), (0.3, 2.0)]:
        win = max(5, 2 * int(np.ceil(3 * ss)) + 1)
        out.append(
            case(
                f"bilateralFilter/{sc}/{ss}",
                "bilateralFilter",
                [blocks],
                bilateral(blocks, win, sc, ss),
                **TOL,
                sigmaColor=sc,
                sigmaSpatial=ss,
            )
        )
    color = np.stack([blocks, 0.5 * blocks.T, blocks[::-1]], axis=2)
    out.append(
        case(
            "bilateralFilter/color",
            "bilateralFilter",
            [color],
            bilateral(color, 7, 0.2, 1.0),
            **TOL,
            sigmaColor=0.2,
            sigmaSpatial=1.0,
        )
    )
    return out
