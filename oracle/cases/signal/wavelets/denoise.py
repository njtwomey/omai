"""Wavelet thresholding against pywt.threshold and denoising against skimage.restoration.denoise_wavelet."""

from importlib.metadata import version
from typing import Literal

import numpy as np
import pywt
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from mdml_oracle.signal_helpers import smooth_image
from skimage.restoration import denoise_wavelet

REFERENCE: dict[str, str] = {"pywt": pywt.__version__, "scikit-image": version("scikit-image")}

TOL = {"rtol": 1e-10, "atol": 1e-12}

type Rule = Literal["soft", "hard", "garrote", "greater", "less"]
RULES: list[Rule] = ["soft", "hard", "garrote", "greater", "less"]


def cases() -> list[Case]:
    g = rng(415)
    out: list[Case] = []
    x = np.concatenate([g.normal(size=12) * 2, [1.0, -1.0, 0.0]])
    for mode in RULES:
        out.append(
            case(f"threshold/{mode}", "threshold", [x], pywt.threshold(x, 1.0, mode=mode), **TOL, value=1.0, mode=mode)
        )
    out.append(
        case(
            "threshold/substitute",
            "threshold",
            [x],
            pywt.threshold(x, 1.0, mode="hard", substitute=-9),
            **TOL,
            value=1.0,
            mode="hard",
            substitute=-9,
        )
    )
    t = np.linspace(0, 1, 128)
    sig = np.sin(8 * t) + (t > 0.5) + 0.2 * g.normal(size=128)
    methods: list[Literal["BayesShrink", "VisuShrink"]] = ["BayesShrink", "VisuShrink"]
    for method in methods:
        for name in ["db1", "db4", "sym6"]:
            d = denoise_wavelet(sig, wavelet=name, method=method, mode="soft")
            out.append(
                case(f"denoise/1d/{method}/{name}", "waveletDenoise", [sig], d, **TOL, method=method, wavelet=name)
            )
    d = denoise_wavelet(sig, sigma=0.3, wavelet="db2", method="VisuShrink", mode="hard", wavelet_levels=2)
    out.append(
        case(
            "denoise/1d/given",
            "waveletDenoise",
            [sig],
            d,
            **TOL,
            method="VisuShrink",
            wavelet="db2",
            mode="hard",
            sigma=0.3,
            levels=2,
        )
    )
    img = smooth_image(g, 20, 24) + 0.1 * g.normal(size=(20, 24))
    for method in methods:
        d = denoise_wavelet(img, wavelet="db2", method=method)
        out.append(case(f"denoise/2d/{method}", "waveletDenoise", [img], d, **TOL, method=method, wavelet="db2"))
    return out
