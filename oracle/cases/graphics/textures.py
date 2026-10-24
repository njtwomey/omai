"""Texture sampling against scipy.ndimage.map_coordinates: texel centres at ((i + 0.5) / w, (j + 0.5) / h), so a
sample at (u, v) is the interpolation at pixel coordinates (v h - 0.5, u w - 0.5), nearest (order 0) or bilinear
(order 1), with tiling ('grid-wrap'), edge clamping ('nearest') and mirroring ('reflect'); mipmap levels by 2 x 2
block means; and the procedural checkerboard and brick patterns written out directly."""

import math
from typing import Literal

import numpy as np
import scipy
from scipy import ndimage as ndi
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}

type Mode = Literal["grid-wrap", "nearest", "reflect"]

MODES: dict[str, Mode] = {"repeat": "grid-wrap", "clamp": "nearest", "mirror": "reflect"}


def sample(tex: FloatArray, uv: FloatArray, order: Literal[0, 1], wrap: str) -> FloatArray:
    h, w, _ = tex.shape
    coords = np.stack([uv[:, 1] * h - 0.5, uv[:, 0] * w - 0.5])
    return np.stack(
        [ndi.map_coordinates(tex[:, :, k], coords, order=order, mode=MODES[wrap]) for k in range(3)], axis=1
    )


def checker(u: float, v: float, scale: float) -> float:
    return 1.0 if (math.floor(u * scale) + math.floor(v * scale)) % 2 == 0 else 0.1


def brick(u: float, v: float) -> float:
    rows, cols, mortar, offset = 4, 2, 0.08, 0.5
    y = v * rows
    row = math.floor(y)
    x = u * cols + (offset if row % 2 != 0 else 0)
    return 0.8 if (y - row) < mortar or (x - math.floor(x)) * rows / cols < mortar else 0.6


def cases() -> list[Case]:
    g = rng(808)
    out: list[Case] = []
    tex = g.uniform(0, 1, size=(6, 8, 3))
    uv = g.uniform(-0.6, 1.6, size=(20, 2))
    orders: list[tuple[Literal[0, 1], str]] = [(0, "nearest"), (1, "bilinear")]
    for wrap in MODES:
        for order, name in orders:
            out.append(case(f"{name}/{wrap}", name, [tex, uv], sample(tex, uv, order, wrap), wrap=wrap))
    base = g.uniform(0, 1, size=(4, 8, 3))
    levels = [base]
    while levels[-1].shape[0] > 1 or levels[-1].shape[1] > 1:
        cur = levels[-1]
        h, w = max(1, cur.shape[0] // 2), max(1, cur.shape[1] // 2)
        levels.append(cur.reshape(h, cur.shape[0] // h, w, cur.shape[1] // w, 3).mean(axis=(1, 3)))
    out.append(case("mipmaps", "mipmaps", [base], np.concatenate([lv.ravel() for lv in levels])))
    pts = g.uniform(0, 1, size=(30, 2))
    out.append(case("checkerboard", "checkerboard", [pts], np.array([checker(u, v, 8) for u, v in pts])))
    out.append(case("bricks", "bricks", [pts], np.array([brick(u, v) for u, v in pts])))
    return out
