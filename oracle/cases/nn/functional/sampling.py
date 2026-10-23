"""Spatial transformer ops against torch.nn.functional (float64): affine_grid (2-D and 3-D, aligned or not) and
grid_sample (bilinear, nearest, bicubic; zeros, border and reflection padding; aligned or not) at grids reaching
outside the image. ROI Align and ROI pooling have no reference here (torchvision is not an oracle dependency); they
are checked by properties in the tests."""

import numpy as np
import torch
import torch.nn.functional as F
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng
from whitebox_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def cases() -> list[Case]:
    g = rng(351)

    def n(*shape: int) -> FloatArray:
        return np.asarray(g.normal(size=shape), dtype=np.float64)

    out: list[Case] = []
    theta2 = n(2, 2, 3) * 0.5 + np.array([[1.0, 0, 0], [0, 1.0, 0]])
    theta3 = n(1, 3, 4) * 0.3 + np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])
    for align in [False, True]:
        y = F.affine_grid(to_torch(theta2), [2, 1, 3, 4], align_corners=align)
        out.append(
            case(f"affineGrid/2d/{align}", "affineGrid", [theta2], to_numpy(y), size=[2, 1, 3, 4], alignCorners=align)
        )
        y = F.affine_grid(to_torch(theta3), [1, 1, 2, 3, 2], align_corners=align)
        out.append(
            case(
                f"affineGrid/3d/{align}", "affineGrid", [theta3], to_numpy(y), size=[1, 1, 2, 3, 2], alignCorners=align
            )
        )
    x = n(2, 2, 4, 5)
    grid = np.asarray(g.uniform(-1.3, 1.3, size=(2, 3, 4, 2)), dtype=np.float64)
    for mode in ["bilinear", "nearest", "bicubic"]:
        for padding in ["zeros", "border", "reflection"]:
            for align in [False, True]:
                y = F.grid_sample(to_torch(x), to_torch(grid), mode=mode, padding_mode=padding, align_corners=align)
                out.append(
                    case(
                        f"gridSample/{mode}/{padding}/{align}",
                        "gridSample",
                        [x, grid],
                        to_numpy(y),
                        mode=mode,
                        paddingMode=padding,
                        alignCorners=align,
                    )
                )
    return out
