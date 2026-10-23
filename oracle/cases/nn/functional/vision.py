"""Vision ops against torch.nn.functional (float64): interpolate in every mode (nearest, nearest-exact, linear,
bilinear, bicubic, trilinear, area) by size and by scale factor, with and without aligned corners; pixel shuffle and
unshuffle; F.pad in every mode."""

import numpy as np
import torch
import torch.nn.functional as F
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng
from whitebox_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def cases() -> list[Case]:
    g = rng(350)

    def n(*shape: int) -> FloatArray:
        return np.asarray(g.normal(size=shape), dtype=np.float64)

    out: list[Case] = []
    x1, x2, x3 = n(2, 2, 5), n(1, 2, 4, 5), n(1, 1, 3, 4, 2)
    specs: list[tuple[str, FloatArray, str, list[int] | None, float | None, bool | None]] = [
        ("nearest/up-size", x1, "nearest", [9], None, None),
        ("nearest/down-scale", x2, "nearest", None, 0.6, None),
        ("nearest-exact/size", x2, "nearest-exact", [7, 3], None, None),
        ("linear/up", x1, "linear", [8], None, False),
        ("linear/align", x1, "linear", None, 1.7, True),
        ("bilinear/down", x2, "bilinear", [3, 3], None, False),
        ("bilinear/up-align", x2, "bilinear", [7, 9], None, True),
        ("bicubic/up", x2, "bicubic", [6, 8], None, False),
        ("bicubic/scale-align", x2, "bicubic", None, 1.5, True),
        ("trilinear/up", x3, "trilinear", [5, 6, 3], None, False),
        ("area/down", x2, "area", [2, 3], None, None),
    ]
    for name, x, mode, size, scale, align in specs:
        y = F.interpolate(to_torch(x), size=size, scale_factor=scale, mode=mode, align_corners=align)
        out.append(
            case(name, "interpolate", [x], to_numpy(y), size=size, scaleFactor=scale, mode=mode, alignCorners=align)
        )
    xs = n(2, 8, 3, 2)
    out.append(case("pixelShuffle", "pixelShuffle", [xs], to_numpy(F.pixel_shuffle(to_torch(xs), 2)), factor=2))
    xu = n(1, 2, 6, 4)
    out.append(case("pixelUnshuffle", "pixelUnshuffle", [xu], to_numpy(F.pixel_unshuffle(to_torch(xu), 2)), factor=2))
    xp = n(1, 2, 4, 5)
    for mode in ["constant", "reflect", "replicate", "circular"]:
        widths = [2, 1, 1, 3]
        y = F.pad(to_torch(xp), widths, mode=mode, value=0.5 if mode == "constant" else None)
        out.append(case(f"pad/{mode}", "pad", [xp], to_numpy(y), widths=widths, mode=mode, value=0.5))
    return out
