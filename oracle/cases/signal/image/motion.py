"""Matching and registration against scikit-image: match_template (with and without padding) and
phase_cross_correlation at pixel precision."""

from importlib.metadata import version

import numpy as np
import skimage.feature as F
import skimage.registration as R
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.signal_helpers import smooth_image

REFERENCE: dict[str, str] = {"scikit-image": version("scikit-image")}


def cases() -> list[Case]:
    g = rng(510)
    out: list[Case] = []
    img = smooth_image(g, 18, 20) + 0.05 * g.normal(size=(18, 20))
    tpl = img[5:10, 7:13].copy()
    for pad in [False, True]:
        r = F.match_template(img, tpl, pad_input=pad)
        out.append(case(f"matchTemplate/{pad}", "matchTemplate", [img, tpl], r, rtol=1e-9, atol=1e-10, padInput=pad))
    tpl4 = img[2:6, 3:7].copy()
    r = F.match_template(img, tpl4, pad_input=True)
    out.append(case("matchTemplate/even", "matchTemplate", [img, tpl4], r, rtol=1e-9, atol=1e-10, padInput=True))
    for dy, dx in [(3, -2), (-5, 4)]:
        moved = np.roll(img, (dy, dx), axis=(0, 1)) + 0.01 * g.normal(size=img.shape)
        shift, error, phase = R.phase_cross_correlation(img, moved)
        out.append(
            case(
                f"phaseCrossCorrelation/{dy}/{dx}",
                "phaseCrossCorrelation",
                [img, moved],
                np.concatenate([shift, [error, phase]]),
                rtol=1e-9,
                atol=1e-10,
            )
        )
    return out
