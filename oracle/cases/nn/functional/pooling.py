"""Pooling against torch.nn.functional (float64): max (padding, dilation, ceil mode, indices), unpooling, average
(padding counted or not, ceil mode, divisor override), adaptive average and max, and Lp pooling."""

import numpy as np
import torch
import torch.nn.functional as F
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng
from whitebox_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def cases() -> list[Case]:
    g = rng(341)

    def n(*shape: int) -> FloatArray:
        return np.asarray(g.normal(size=shape), dtype=np.float64)

    out: list[Case] = []
    max_fns = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}
    max_specs: list[tuple[str, FloatArray, dict[str, object]]] = [
        ("max1d/basic", n(2, 3, 9), {"kernel_size": 3}),
        ("max1d/stride-pad-ceil", n(1, 2, 10), {"kernel_size": 3, "stride": 2, "padding": 1, "ceil_mode": True}),
        ("max1d/dilation", n(1, 2, 11), {"kernel_size": 3, "stride": 1, "dilation": 2}),
        ("max2d/basic", n(2, 2, 6, 6), {"kernel_size": 2}),
        ("max2d/overlap-pad", n(1, 3, 7, 5), {"kernel_size": 3, "stride": 2, "padding": 1}),
        ("max2d/ceil", n(1, 1, 5, 7), {"kernel_size": [2, 3], "stride": [2, 2], "ceil_mode": True}),
        ("max3d/basic", n(1, 2, 4, 4, 4), {"kernel_size": 2, "stride": 2}),
    ]
    for name, x, kw in max_specs:
        d = x.ndim - 2
        y, idx = max_fns[d](to_torch(x), return_indices=True, **kw)  # pyright: ignore[reportArgumentType, reportCallIssue]
        for fn, output in [("maxPool", to_numpy(y)), ("maxPoolIndices", idx.numpy().astype(np.int32))]:
            out.append(
                case(
                    name if fn == "maxPool" else f"{name}/indices",
                    fn,
                    [x],
                    output,
                    kernelSize=kw["kernel_size"],
                    stride=kw.get("stride", kw["kernel_size"]),
                    padding=kw.get("padding", 0),
                    dilation=kw.get("dilation", 1),
                    ceilMode=kw.get("ceil_mode", False),
                    dims=d,
                )
            )
    x = n(1, 2, 6, 6)
    y, idx = F.max_pool2d(to_torch(x), 2, return_indices=True)
    un = F.max_unpool2d(y, idx, (2, 2))
    out.append(
        case("maxUnpool2d", "maxUnpool", [to_numpy(y), idx.numpy().astype(np.int32)], to_numpy(un), kernelSize=[2, 2])
    )

    avg_fns = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}
    avg_specs: list[tuple[str, FloatArray, dict[str, object]]] = [
        ("avg1d/basic", n(2, 3, 8), {"kernel_size": 2}),
        ("avg1d/pad-include", n(1, 2, 7), {"kernel_size": 3, "stride": 2, "padding": 1}),
        ("avg1d/pad-exclude", n(1, 2, 7), {"kernel_size": 3, "stride": 2, "padding": 1, "count_include_pad": False}),
        ("avg2d/ceil", n(1, 2, 5, 5), {"kernel_size": 2, "stride": 2, "ceil_mode": True}),
        ("avg2d/ceil-pad", n(1, 1, 6, 5), {"kernel_size": 3, "stride": 2, "padding": 1, "ceil_mode": True}),
        ("avg2d/divisor", n(1, 2, 4, 4), {"kernel_size": 2, "divisor_override": 3}),
        ("avg3d/basic", n(1, 1, 4, 4, 4), {"kernel_size": 2}),
    ]
    for name, x, kw in avg_specs:
        d = x.ndim - 2
        y = avg_fns[d](to_torch(x), **kw)  # pyright: ignore[reportArgumentType, reportCallIssue]
        out.append(
            case(
                name,
                "avgPool",
                [x],
                to_numpy(y),
                kernelSize=kw["kernel_size"],
                stride=kw.get("stride", kw["kernel_size"]),
                padding=kw.get("padding", 0),
                ceilMode=kw.get("ceil_mode", False),
                countIncludePad=kw.get("count_include_pad", True),
                divisorOverride=kw.get("divisor_override"),
                dims=d,
            )
        )

    adaptive: list[tuple[str, FloatArray, list[int]]] = [
        ("adaptive1d/divides", n(2, 3, 8), [4]),
        ("adaptive1d/uneven", n(1, 2, 10), [3]),
        ("adaptive1d/upsample", n(1, 2, 3), [5]),
        ("adaptive2d/uneven", n(1, 2, 7, 5), [3, 2]),
        ("adaptive3d/uneven", n(1, 1, 5, 4, 3), [2, 3, 2]),
    ]
    avg_adaptive = {1: F.adaptive_avg_pool1d, 2: F.adaptive_avg_pool2d, 3: F.adaptive_avg_pool3d}
    max_adaptive = {1: F.adaptive_max_pool1d, 2: F.adaptive_max_pool2d, 3: F.adaptive_max_pool3d}
    for name, x, size in adaptive:
        d = x.ndim - 2
        a = avg_adaptive[d](to_torch(x), size)  # pyright: ignore[reportArgumentType]
        m = max_adaptive[d](to_torch(x), size)  # pyright: ignore[reportArgumentType]
        out.append(case(f"{name}/avg", "adaptiveAvgPool", [x], to_numpy(a), outputSize=size))
        out.append(case(f"{name}/max", "adaptiveMaxPool", [x], to_numpy(m), outputSize=size))

    pos1 = np.abs(n(1, 2, 8)) + 0.1
    pos2 = np.abs(n(1, 2, 6, 6)) + 0.1
    out.append(
        case("lp1d/p2", "lpPool", [pos1], to_numpy(F.lp_pool1d(to_torch(pos1), 2.0, 2)), normType=2.0, kernelSize=[2])
    )
    out.append(
        case(
            "lp2d/p3-stride",
            "lpPool",
            [pos2],
            to_numpy(F.lp_pool2d(to_torch(pos2), 3.0, 3, stride=2)),
            normType=3.0,
            kernelSize=[3, 3],
            stride=2,
        )
    )
    return out
