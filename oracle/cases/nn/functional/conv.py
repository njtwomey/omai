"""Convolutions against torch.nn.functional (float64): 1-D, 2-D and 3-D, strided, dilated, grouped, depthwise,
padded ('same' and each padding mode), transposed with output padding, unfold and fold."""

import numpy as np
import torch
import torch.nn.functional as F
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}

TORCH_MODE = {"zeros": "constant", "reflect": "reflect", "replicate": "replicate", "circular": "circular"}


def conv_case(
    name: str,
    x: FloatArray,
    w: FloatArray,
    b: FloatArray | None,
    stride: int | list[int] = 1,
    padding: int | list[int] | str = 0,
    dilation: int | list[int] = 1,
    groups: int = 1,
    padding_mode: str = "zeros",
) -> Case:
    d = w.ndim - 2
    fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[d]
    xt = to_torch(x)
    pad: int | list[int] | str = padding
    if padding_mode != "zeros":
        widths = [padding] * d if isinstance(padding, int) else padding
        assert isinstance(widths, list)
        flat = [p for q in reversed(widths) for p in (q, q)]
        xt = F.pad(xt, flat, mode=TORCH_MODE[padding_mode])
        pad = 0
    y = fn(xt, to_torch(w), None if b is None else to_torch(b), stride, pad, dilation, groups)
    inputs = [x, w] if b is None else [x, w, b]
    return case(
        name,
        f"conv{d}d",
        inputs,
        to_numpy(y),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        paddingMode=padding_mode,
    )


def transpose_case(
    name: str,
    x: FloatArray,
    w: FloatArray,
    b: FloatArray | None,
    stride: int | list[int] = 1,
    padding: int | list[int] = 0,
    output_padding: int | list[int] = 0,
    groups: int = 1,
    dilation: int | list[int] = 1,
) -> Case:
    d = w.ndim - 2
    fn = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}[d]
    y = fn(
        to_torch(x), to_torch(w), None if b is None else to_torch(b), stride, padding, output_padding, groups, dilation
    )
    inputs = [x, w] if b is None else [x, w, b]
    return case(
        name,
        f"convTranspose{d}d",
        inputs,
        to_numpy(y),
        stride=stride,
        padding=padding,
        outputPadding=output_padding,
        groups=groups,
        dilation=dilation,
    )


def cases() -> list[Case]:
    g = rng(340)

    def n(*shape: int) -> FloatArray:
        return np.asarray(g.normal(size=shape), dtype=np.float64)

    out: list[Case] = [
        conv_case("conv1d/basic", n(2, 3, 9), n(4, 3, 3), n(4)),
        conv_case("conv1d/stride-pad-dilation", n(2, 2, 11), n(3, 2, 3), n(3), stride=2, padding=2, dilation=2),
        conv_case("conv1d/same-even-kernel", n(1, 2, 7), n(2, 2, 4), None, padding="same"),
        conv_case("conv1d/groups", n(2, 4, 8), n(6, 2, 3), n(6), groups=2),
        conv_case("conv1d/depthwise", n(1, 3, 8), n(6, 1, 3), None, groups=3, padding=1),
        conv_case("conv1d/reflect", n(1, 2, 6), n(2, 2, 3), n(2), padding=2, padding_mode="reflect"),
        conv_case("conv1d/circular", n(1, 2, 6), n(2, 2, 3), None, padding=1, padding_mode="circular"),
        conv_case("conv2d/basic", n(2, 3, 6, 5), n(4, 3, 3, 3), n(4), padding=1),
        conv_case(
            "conv2d/asymmetric", n(1, 2, 7, 6), n(3, 2, 3, 2), n(3), stride=[2, 1], padding=[1, 0], dilation=[1, 2]
        ),
        conv_case("conv2d/groups", n(1, 4, 5, 5), n(4, 1, 3, 3), None, groups=4),
        conv_case("conv2d/replicate", n(1, 1, 4, 4), n(2, 1, 3, 3), n(2), padding=1, padding_mode="replicate"),
        conv_case("conv2d/same", n(1, 2, 5, 6), n(2, 2, 3, 3), None, padding="same", dilation=2),
        conv_case("conv3d/basic", n(1, 2, 4, 4, 3), n(3, 2, 2, 2, 2), n(3), padding=1, stride=2),
        transpose_case("convTranspose1d/basic", n(2, 3, 5), n(3, 2, 3), n(2), stride=2, padding=1, output_padding=1),
        transpose_case("convTranspose1d/groups-dilation", n(1, 4, 4), n(4, 1, 3), None, stride=2, groups=2, dilation=2),
        transpose_case("convTranspose2d/basic", n(1, 3, 3, 4), n(3, 2, 4, 4), n(2), stride=2, padding=1),
        transpose_case(
            "convTranspose2d/output-padding", n(2, 2, 3, 3), n(2, 3, 3, 3), n(3), stride=2, padding=1, output_padding=1
        ),
        transpose_case("convTranspose3d/basic", n(1, 2, 2, 3, 2), n(2, 2, 2, 2, 2), n(2), stride=2),
    ]
    x = n(2, 3, 5, 4)
    cols = F.unfold(to_torch(x), kernel_size=(2, 3), dilation=(1, 1), padding=(1, 0), stride=(2, 1))
    out.append(case("unfold/2d", "unfold", [x], to_numpy(cols), kernelSize=[2, 3], padding=[1, 0], stride=[2, 1]))
    blocks = n(1, 2 * 4, 9)
    folded = F.fold(to_torch(blocks), output_size=(4, 4), kernel_size=2, stride=1)
    out.append(case("fold/2d", "fold", [blocks], to_numpy(folded), outputSize=[4, 4], kernelSize=2, stride=1))
    return out
