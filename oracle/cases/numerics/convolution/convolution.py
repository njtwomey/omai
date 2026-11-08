"""Convolution against numpy.convolve/correlate/pad, scipy.signal and torch.nn.functional."""

from typing import Literal

import numpy as np
import scipy
import scipy.signal as sg
import torch
import torch.nn.functional as tf
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from omai_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__}

MODES: list[Literal["full", "same", "valid"]] = ["full", "same", "valid"]
PADS: list[Literal["constant", "edge", "reflect", "symmetric", "wrap", "maximum", "minimum", "mean"]] = [
    "constant",
    "edge",
    "reflect",
    "symmetric",
    "wrap",
    "maximum",
    "minimum",
    "mean",
]


def cases() -> list[Case]:
    g = rng(100)
    out: list[Case] = []
    for n1, n2 in [(7, 3), (3, 7), (5, 5)]:
        x, v = g.normal(size=n1), g.normal(size=n2)
        for mode in MODES:
            out.append(
                case(
                    f"convolve/{n1}x{n2}/{mode}",
                    "convolve",
                    [x, v],
                    np.convolve(x, v, mode),
                    rtol=1e-12,
                    atol=1e-13,
                    mode=mode,
                )
            )
            out.append(
                case(
                    f"correlate/{n1}x{n2}/{mode}",
                    "correlate",
                    [x, v],
                    np.correlate(x, v, mode),
                    rtol=1e-12,
                    atol=1e-13,
                    mode=mode,
                )
            )
            out.append(
                case(
                    f"fftConvolve/{n1}x{n2}/{mode}",
                    "fftConvolve",
                    [x, v],
                    sg.fftconvolve(x, v, mode),
                    rtol=1e-10,
                    atol=1e-11,
                    mode=mode,
                )
            )
            out.append(
                case(
                    f"oaConvolve/{n1}x{n2}/{mode}",
                    "oaConvolve",
                    [x, v],
                    sg.oaconvolve(x, v, mode),
                    rtol=1e-10,
                    atol=1e-11,
                    mode=mode,
                )
            )
            out.append(
                case(
                    f"lags/{n1}x{n2}/{mode}",
                    "lags",
                    [],
                    sg.correlation_lags(n1, n2, mode).astype(np.float64),
                    rtol=0,
                    n1=n1,
                    n2=n2,
                    mode=mode,
                )
            )
    long, short = g.normal(size=300), g.normal(size=17)
    out.append(
        case(
            "oaConvolve/long",
            "oaConvolve",
            [long, short],
            sg.oaconvolve(long, short),
            rtol=1e-9,
            atol=1e-10,
            mode="full",
        )
    )
    img, ker = g.normal(size=(6, 7)), g.normal(size=(3, 2))
    for mode in MODES:
        out.append(
            case(
                f"fftConvolve/2d/{mode}",
                "fftConvolve",
                [img, ker],
                sg.fftconvolve(img, ker, mode),
                rtol=1e-10,
                atol=1e-11,
                mode=mode,
            )
        )
    for mode in PADS:
        arr = g.normal(size=(3, 4))
        out.append(
            case(
                f"pad/{mode}", "pad", [arr], np.pad(arr, ((2, 1), (3, 5)), mode=mode), rtol=1e-13, atol=1e-14, mode=mode
            )
        )

    x1 = g.normal(size=(2, 4, 11))
    w1 = g.normal(size=(6, 2, 3))
    b1 = g.normal(size=6)
    out.append(
        case(
            "conv1d",
            "conv",
            [x1, w1, b1],
            to_numpy(tf.conv1d(to_torch(x1), to_torch(w1), to_torch(b1), stride=2, padding=1, dilation=2, groups=2)),
            rtol=1e-10,
            atol=1e-11,
            stride=2,
            padding=1,
            dilation=2,
            groups=2,
        )
    )
    out.append(
        case(
            "conv1d/same",
            "convSame",
            [x1, w1, b1],
            to_numpy(tf.conv1d(to_torch(x1), to_torch(w1), to_torch(b1), padding="same", dilation=2, groups=2)),
            rtol=1e-10,
            atol=1e-11,
            dilation=2,
            groups=2,
        )
    )
    wr = g.normal(size=(3, 4, 3))
    reflected = tf.conv1d(tf.pad(to_torch(x1), (2, 2), mode="reflect"), to_torch(wr))
    out.append(case("conv1d/reflect", "convReflect", [x1, wr], to_numpy(reflected), rtol=1e-10, atol=1e-11))
    x2 = g.normal(size=(2, 3, 7, 8))
    w2 = g.normal(size=(4, 3, 3, 2))
    out.append(
        case(
            "conv2d",
            "conv",
            [x2, w2],
            to_numpy(tf.conv2d(to_torch(x2), to_torch(w2), stride=(2, 1), padding=(1, 0), dilation=(1, 2))),
            rtol=1e-10,
            atol=1e-11,
            stride=[2, 1],
            padding=[1, 0],
            dilation=[1, 2],
            groups=1,
        )
    )
    x3 = g.normal(size=(1, 2, 5, 4, 6))
    w3 = g.normal(size=(2, 2, 2, 3, 2))
    out.append(
        case(
            "conv3d",
            "conv",
            [x3, w3],
            to_numpy(tf.conv3d(to_torch(x3), to_torch(w3), padding=1)),
            rtol=1e-10,
            atol=1e-11,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
        )
    )
    xt = g.normal(size=(2, 4, 5))
    wt = g.normal(size=(4, 3, 3))
    bt = g.normal(size=6)
    out.append(
        case(
            "convTranspose1d",
            "convTranspose",
            [xt, wt, bt],
            to_numpy(
                tf.conv_transpose1d(
                    to_torch(xt),
                    to_torch(wt),
                    to_torch(bt),
                    stride=2,
                    padding=1,
                    output_padding=1,
                    groups=2,
                    dilation=1,
                )
            ),
            rtol=1e-10,
            atol=1e-11,
            stride=2,
            padding=1,
            outputPadding=1,
            groups=2,
            dilation=1,
        )
    )
    xt2 = g.normal(size=(1, 3, 4, 3))
    wt2 = g.normal(size=(3, 2, 3, 3))
    out.append(
        case(
            "convTranspose2d",
            "convTranspose",
            [xt2, wt2],
            to_numpy(tf.conv_transpose2d(to_torch(xt2), to_torch(wt2), stride=(2, 1), padding=(1, 1), dilation=(1, 2))),
            rtol=1e-10,
            atol=1e-11,
            stride=[2, 1],
            padding=[1, 1],
            outputPadding=0,
            groups=1,
            dilation=[1, 2],
        )
    )
    xu = g.normal(size=(2, 3, 6, 5))
    out.append(
        case(
            "unfold",
            "unfold",
            [xu],
            to_numpy(tf.unfold(to_torch(xu), (3, 2), dilation=(1, 2), padding=1, stride=(2, 1))),
            rtol=1e-13,
            kernel=[3, 2],
            dilation=[1, 2],
            padding=[1, 1],
            stride=[2, 1],
        )
    )
    cols = to_numpy(tf.unfold(to_torch(xu), (3, 2), dilation=(1, 2), padding=1, stride=(2, 1)))
    out.append(
        case(
            "fold",
            "fold",
            [cols],
            to_numpy(tf.fold(to_torch(cols), (6, 5), (3, 2), dilation=(1, 2), padding=1, stride=(2, 1))),
            rtol=1e-12,
            atol=1e-13,
            kernel=[3, 2],
            dilation=[1, 2],
            padding=[1, 1],
            stride=[2, 1],
        )
    )
    return out
