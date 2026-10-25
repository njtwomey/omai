"""Segmentation against scikit-image: Otsu and Li thresholds, marker-based watershed and Felzenszwalb's
graph-based segmentation (on noisy images, so that no two edge weights tie: scikit-image's unstable sort would
otherwise decide)."""

from importlib.metadata import version

import numpy as np
import scipy.ndimage as N
import skimage.filters as F
import skimage.segmentation as S
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from mdml_oracle.signal_helpers import blocks_image, smooth_image

REFERENCE: dict[str, str] = {"scikit-image": version("scikit-image")}

TOL = {"rtol": 1e-12, "atol": 1e-12}


def cases() -> list[Case]:
    g = rng(500)
    out: list[Case] = []
    img = smooth_image(g, 20, 24)
    noisy = blocks_image(20, 20) + 0.08 * g.normal(size=(20, 20))
    for nbins in [256, 32]:
        out.append(
            case(
                f"thresholdOtsu/{nbins}",
                "thresholdOtsu",
                [noisy],
                np.atleast_1d(F.threshold_otsu(noisy, nbins=nbins)),
                **TOL,
                nbins=nbins,
            )
        )
    out.append(
        case("thresholdLi", "thresholdLi", [noisy], np.atleast_1d(F.threshold_li(noisy)), rtol=1e-10, atol=1e-12)
    )
    out.append(
        case("thresholdLi/smooth", "thresholdLi", [img], np.atleast_1d(F.threshold_li(img)), rtol=1e-10, atol=1e-12)
    )
    grad = F.sobel(img)
    markers = np.zeros(img.shape)
    markers[2, 3] = 1
    markers[15, 20] = 2
    markers[10, 2] = 3
    markers[5, 18] = 4
    for conn in [1, 2]:
        ws = S.watershed(grad, markers.astype(int), connectivity=conn)
        out.append(case(f"watershed/{conn}", "watershed", [grad, markers], ws.astype(float), connectivity=conn))
    mask = N.binary_dilation(np.ones((20, 24)) * 0 + (img > 0.45), iterations=1).astype(float)
    ws = S.watershed(grad, markers.astype(int), mask=mask.astype(bool))
    out.append(case("watershed/mask", "watershed", [grad, markers, mask], ws.astype(float)))
    for scale, min_size in [(1.0, 20), (5.0, 10)]:
        fz = S.felzenszwalb(noisy, scale=scale, sigma=0.8, min_size=min_size, channel_axis=None)
        out.append(
            case(f"felzenszwalb/{scale}", "felzenszwalb", [noisy], fz.astype(float), scale=scale, minSize=min_size)
        )
    rgb = np.stack([noisy, 0.5 * noisy[::-1], img[:20, :20]], axis=2)
    fz = S.felzenszwalb(rgb, scale=2.0, sigma=0.5, min_size=15)
    out.append(case("felzenszwalb/rgb", "felzenszwalb", [rgb], fz.astype(float), scale=2.0, sigma=0.5, minSize=15))
    return out
