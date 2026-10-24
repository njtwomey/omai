"""Image quality metrics against scikit-image (MSE, PSNR, SSIM), sewar (MS-SSIM, VIF-P, SAM, ERGAS, RASE) and
torchmetrics (UQI); LPIPS and DISTS from supplied features as their papers define them, written over numpy.
"""

from importlib.metadata import version
from typing import Any

import numpy as np
import sewar.full_ref as S
import skimage.metrics as K
import torch
from torchmetrics.functional.image import universal_image_quality_index
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {
    "scikit-image": version("scikit-image"),
    "sewar": version("sewar"),
    "torchmetrics": version("torchmetrics"),
}


def smooth_image(g: np.random.Generator, h: int, w: int, c: int) -> FloatArray:
    yy, xx = np.mgrid[:h, :w]
    base = np.stack([np.sin(yy / (3 + k)) * np.cos(xx / (4 + k)) for k in range(c)], axis=-1)
    return 0.5 + 0.4 * base + 0.05 * g.normal(size=(h, w, c))


def lpips(fx: list[FloatArray], fy: list[FloatArray], weights: list[FloatArray]) -> float:
    total = 0.0
    for x, y, w in zip(fx, fy, weights, strict=True):
        xn = x / (np.sqrt((x**2).sum(-1, keepdims=True)) + 1e-10)
        yn = y / (np.sqrt((y**2).sum(-1, keepdims=True)) + 1e-10)
        total += float((((xn - yn) ** 2) * w).sum(-1).mean())
    return total


def dists(fx: list[FloatArray], fy: list[FloatArray], alpha: list[FloatArray], beta: list[FloatArray]) -> float:
    score = 0.0
    weight = sum(float(a.sum() + b.sum()) for a, b in zip(alpha, beta, strict=True))
    for x, y, a, b in zip(fx, fy, alpha, beta, strict=True):
        mx, my = x.mean((0, 1)), y.mean((0, 1))
        vx, vy = x.var((0, 1)), y.var((0, 1))
        cxy = ((x - mx) * (y - my)).mean((0, 1))
        s1 = (2 * mx * my + 1e-6) / (mx**2 + my**2 + 1e-6)
        s2 = (2 * cxy + 1e-6) / (vx + vy + 1e-6)
        score += float((a * s1 + b * s2).sum())
    return 1 - score / weight


def cases() -> list[Case]:
    g = rng(69)
    ref = smooth_image(g, 48, 52, 3)
    test = np.clip(ref + 0.08 * g.normal(size=ref.shape), 0, 1)
    gray, gray_test = ref[:, :, 0], test[:, :, 0]
    big = smooth_image(g, 72, 72, 1)[:, :, 0]
    big_test = big + 0.05 * g.normal(size=big.shape)
    uqi = universal_image_quality_index(
        torch.tensor(test.transpose(2, 0, 1)[None]), torch.tensor(ref.transpose(2, 0, 1)[None])
    )
    shapes = [(12, 13, 4), (6, 7, 8)]
    fx = [g.normal(size=s) for s in shapes]
    fy = [x + 0.3 * g.normal(size=x.shape) for x in fx]
    w = [g.uniform(0, 1, size=s[-1]) for s in shapes]
    a = [g.uniform(0, 1, size=s[-1]) for s in shapes]
    b = [g.uniform(0, 1, size=s[-1]) for s in shapes]
    out: list[Case] = [
        case(
            "imageMeanSquaredError", "imageMeanSquaredError", [ref, test], np.asarray(K.mean_squared_error(ref, test))
        ),
        case(
            "peakSignalNoiseRatio",
            "peakSignalNoiseRatio",
            [ref, test],
            np.asarray(K.peak_signal_noise_ratio(ref, test, data_range=1.0)),
            dataRange=1.0,
        ),
        case(
            "structuralSimilarity/gray",
            "structuralSimilarity",
            [gray, gray_test],
            np.asarray(K.structural_similarity(gray, gray_test, data_range=1.0)),
            dataRange=1.0,
        ),
        case(
            "structuralSimilarity/color",
            "structuralSimilarity",
            [ref, test],
            np.asarray(K.structural_similarity(ref, test, data_range=1.0, channel_axis=-1)),
            dataRange=1.0,
        ),
        case(
            "structuralSimilarity/gaussian",
            "structuralSimilarity",
            [gray, gray_test],
            np.asarray(
                K.structural_similarity(
                    gray, gray_test, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False
                )
            ),
            dataRange=1.0,
            gaussianWeights=True,
            useSampleCovariance=False,
        ),
        case(
            "structuralSimilarity/win5",
            "structuralSimilarity",
            [gray, gray_test],
            np.asarray(K.structural_similarity(gray, gray_test, data_range=2.0, win_size=5)),
            dataRange=2.0,
            winSize=5,
        ),
        case(
            "multiScaleSsim",
            "multiScaleSsim",
            [big, big_test],
            np.asarray(float(S.msssim(big, big_test, MAX=1.0).real)),
            dataRange=1.0,
        ),
        case(
            "visualInformationFidelity",
            "visualInformationFidelity",
            [big * 255, big_test * 255],
            np.asarray(S.vifp(big * 255, big_test * 255)),
            rtol=1e-10,
        ),
        case(
            "universalImageQualityIndex", "universalImageQualityIndex", [ref, test], np.asarray(float(uqi)), rtol=1e-10
        ),
        case("spectralAngleMapper", "spectralAngleMapper", [ref, test], np.asarray(S.sam(ref, test))),
        case("ergas", "ergas", [ref, test], np.asarray(S.ergas(ref, test, r=0.25))),
        case(
            "relativeAverageSpectralError", "relativeAverageSpectralError", [ref, test], np.asarray(S.rase(ref, test))
        ),
        case(
            "relativeAverageSpectralError/ws5",
            "relativeAverageSpectralError",
            [ref, test],
            np.asarray(S.rase(ref, test, ws=5)),
            windowSize=5,
        ),
    ]
    feature_params: dict[str, Any] = {
        "fx": [x.tolist() for x in fx],
        "fy": [y.tolist() for y in fy],
        "w": [v.tolist() for v in w],
        "alpha": [v.tolist() for v in a],
        "beta": [v.tolist() for v in b],
    }
    out += [
        case("lpipsFromFeatures", "lpipsFromFeatures", [], np.asarray(lpips(fx, fy, w)), **feature_params),
        case("distsFromFeatures", "distsFromFeatures", [], np.asarray(dists(fx, fy, a, b)), **feature_params),
    ]
    return out
