"""Causal and masked convolutions against torch (float64): the causal dilated 1-D convolution as left padding followed
by torch.nn.Conv1d, and the PixelCNN masked 2-D convolution as torch.nn.Conv2d on the weight times the mask (mask A and
B, with and without the colour-channel ordering), the mask built here by raster-order position."""

import numpy as np
import torch
import torch.nn.functional as F
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.torch_helpers import to_numpy, to_torch
from torch import nn

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def pixel_mask(out_c: int, in_c: int, kh: int, kw: int, kind: str, colours: int) -> FloatArray:
    """The mask: a tap is open when it comes before the centre in raster order; the centre is open by colour group."""
    mask = np.zeros((out_c, in_c, kh, kw), dtype=np.float64)
    ch, cw = kh // 2, kw // 2
    for o in range(out_c):
        for i in range(in_c):
            for r in range(kh):
                for c in range(kw):
                    if (r, c) < (ch, cw):
                        mask[o, i, r, c] = 1.0
            go, gi = o // (out_c // colours), i // (in_c // colours)
            if (gi <= go) if kind == "B" else (gi < go):
                mask[o, i, ch, cw] = 1.0
    return mask


def cases() -> list[Case]:
    g = rng(5301)
    torch.manual_seed(5301)

    def n(*shape: int) -> FloatArray:
        return np.asarray(g.normal(size=shape), dtype=np.float64)

    out: list[Case] = []
    for name, cin, cout, k, dilation, groups in [
        ("causal/plain", 2, 3, 3, 1, 1),
        ("causal/dilated", 2, 4, 2, 4, 1),
        ("causal/grouped", 4, 4, 3, 2, 2),
    ]:
        conv = nn.Conv1d(cin, cout, k, dilation=dilation, groups=groups).double()
        x = n(2, cin, 17)
        padded = F.pad(to_torch(x), (dilation * (k - 1), 0))
        y = conv(padded)
        state = {kk: to_numpy(v.double()) for kk, v in conv.state_dict().items()}
        cfg: dict[str, object] = {"in": cin, "out": cout, "kernel": k, "dilation": dilation, "groups": groups}
        out.append(case(name, "causalConv1d", [x], to_numpy(y), state=state, config=cfg))

    for name, cin, cout, kh, kw, kind, colours, dilation in [
        ("masked/A-3x3", 1, 2, 3, 3, "A", 1, 1),
        ("masked/B-5x5", 3, 4, 5, 5, "B", 1, 1),
        ("masked/A-rgb", 3, 6, 3, 3, "A", 3, 1),
        ("masked/B-rgb", 3, 6, 3, 3, "B", 3, 1),
        ("masked/B-dilated", 2, 2, 3, 3, "B", 1, 2),
    ]:
        conv2 = nn.Conv2d(
            cin, cout, (kh, kw), padding=(dilation * (kh - 1) // 2, dilation * (kw - 1) // 2), dilation=dilation
        ).double()
        mask = pixel_mask(cout, cin, kh, kw, kind, colours)
        x2 = n(2, cin, 7, 8)
        y2 = F.conv2d(
            to_torch(x2), conv2.weight * to_torch(mask), conv2.bias, padding=conv2.padding, dilation=conv2.dilation
        )
        state2 = {kk: to_numpy(v.double()) for kk, v in conv2.state_dict().items()}
        cfg2: dict[str, object] = {
            "in": cin,
            "out": cout,
            "kh": kh,
            "kw": kw,
            "type": kind,
            "colours": colours,
            "dilation": dilation,
        }
        out.append(case(name, "maskedConv2d", [x2], to_numpy(y2), state=state2, config=cfg2, mask=mask))
    return out
