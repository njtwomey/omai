"""Tone mapping and gamma by their closed forms in numpy: the sRGB transfer function and its inverse
(IEC 61966-2-1), power-law gamma, Reinhard's global operator (plain and extended), Narkowicz's ACES fit, exposure and
Rec. 709 luminance."""

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__}


def cases() -> list[Case]:
    g = rng(2200)
    x = np.concatenate([[0.0, 0.002, 0.0031308, 0.04045, 1.0], g.uniform(0, 1, size=11)])
    hdr = np.concatenate([[0.0, 0.5, 1.0], g.uniform(0, 20, size=9)])
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    linear = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    aces = np.clip(hdr * (2.51 * hdr + 0.03) / (hdr * (2.43 * hdr + 0.59) + 0.14), 0, 1)
    rgb = g.uniform(0, 1, size=(6, 3))
    return [
        case("linearToSrgb", "linearToSrgb", [x], srgb),
        case("srgbToLinear", "srgbToLinear", [x], linear),
        case("gammaEncode", "gammaEncode", [x], x ** (1 / 2.2), gamma=2.2),
        case("gammaDecode", "gammaDecode", [x], x**2.4, gamma=2.4),
        case("reinhard", "reinhard", [hdr], hdr / (1 + hdr)),
        case("reinhardExtended", "reinhardExtended", [hdr], hdr * (1 + hdr / 64) / (1 + hdr), white=8.0),
        case("acesFilm", "acesFilm", [hdr], aces),
        case("exposure", "exposure", [hdr], hdr * 2.0**-1.5, ev=-1.5),
        case("luminance", "luminance", [rgb], rgb @ np.array([0.2126, 0.7152, 0.0722])),
    ]
