"""Normalisation functions against torch.nn.functional (float64): batch normalisation in training (with updated
running statistics) and evaluation, layer, RMS, group, instance and local response normalisation, normalize, and the
weight and spectral normalisation parametrisations."""

import numpy as np
import torch
import torch.nn.functional as F
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.torch_helpers import to_numpy, to_torch
from torch import nn

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def attr(m: nn.Module, name: str) -> torch.Tensor:
    """A tensor attribute of a module (parameters and buffers that parametrisations add by name)."""
    v = getattr(m, name)
    assert isinstance(v, torch.Tensor)
    return v


def cases() -> list[Case]:
    g = rng(342)

    def n(*shape: int) -> FloatArray:
        return np.asarray(g.normal(size=shape), dtype=np.float64)

    out: list[Case] = []
    for name, x in [("2d", n(6, 3)), ("3d", n(4, 3, 5)), ("4d", n(3, 3, 4, 2))]:
        c = x.shape[1]
        rm, rv, w, b = n(c), np.abs(n(c)) + 0.5, n(c), n(c)
        rmt, rvt = to_torch(rm), to_torch(rv)
        y = F.batch_norm(to_torch(x), rmt, rvt, to_torch(w), to_torch(b), training=True, momentum=0.2, eps=1e-4)
        out.append(
            case(f"batchNorm/{name}/train", "batchNormTrain", [x, rm, rv, w, b], to_numpy(y), momentum=0.2, eps=1e-4)
        )
        out.append(
            case(
                f"batchNorm/{name}/runningMean",
                "batchNormRunningMean",
                [x, rm, rv, w, b],
                to_numpy(rmt),
                momentum=0.2,
                eps=1e-4,
            )
        )
        out.append(
            case(
                f"batchNorm/{name}/runningVar",
                "batchNormRunningVar",
                [x, rm, rv, w, b],
                to_numpy(rvt),
                momentum=0.2,
                eps=1e-4,
            )
        )
        ye = F.batch_norm(to_torch(x), to_torch(rm), to_torch(rv), to_torch(w), to_torch(b), training=False, eps=1e-4)
        out.append(case(f"batchNorm/{name}/eval", "batchNormEval", [x, rm, rv, w, b], to_numpy(ye), eps=1e-4))

    x = n(2, 3, 4)
    w, b = n(3, 4), n(3, 4)
    out.append(
        case(
            "layerNorm/2",
            "layerNorm",
            [x, w, b],
            to_numpy(F.layer_norm(to_torch(x), [3, 4], to_torch(w), to_torch(b))),
            shape=[3, 4],
        )
    )
    out.append(case("layerNorm/1-plain", "layerNorm", [x], to_numpy(F.layer_norm(to_torch(x), [4])), shape=[4]))
    out.append(
        case("rmsNorm/weight", "rmsNorm", [x, w], to_numpy(F.rms_norm(to_torch(x), [3, 4], to_torch(w))), shape=[3, 4])
    )
    out.append(
        case("rmsNorm/eps", "rmsNorm", [x], to_numpy(F.rms_norm(to_torch(x), [4], eps=1e-3)), shape=[4], eps=1e-3)
    )

    x = n(2, 6, 3, 2)
    w, b = n(6), n(6)
    for groups in [1, 2, 3, 6]:
        y = F.group_norm(to_torch(x), groups, to_torch(w), to_torch(b))
        out.append(case(f"groupNorm/{groups}", "groupNorm", [x, w, b], to_numpy(y), groups=groups))

    x = n(2, 3, 5)
    w, b = n(3), n(3)
    out.append(
        case(
            "instanceNorm/affine",
            "instanceNorm",
            [x, w, b],
            to_numpy(F.instance_norm(to_torch(x), weight=to_torch(w), bias=to_torch(b))),
        )
    )
    rm, rv = n(3), np.abs(n(3)) + 0.5
    rmt, rvt = to_torch(rm), to_torch(rv)
    y = F.instance_norm(to_torch(x), rmt, rvt, use_input_stats=True, momentum=0.3)
    out.append(case("instanceNorm/running/output", "instanceNormRunning", [x, rm, rv], to_numpy(y), part="output"))
    out.append(case("instanceNorm/running/mean", "instanceNormRunning", [x, rm, rv], to_numpy(rmt), part="runningMean"))
    out.append(case("instanceNorm/running/var", "instanceNormRunning", [x, rm, rv], to_numpy(rvt), part="runningVar"))

    x = n(2, 5, 3)
    out.append(
        case(
            "localResponseNorm/3",
            "localResponseNorm",
            [x],
            to_numpy(F.local_response_norm(to_torch(x), 3, alpha=0.1, beta=0.6, k=2.0)),
            size=3,
            alpha=0.1,
            beta=0.6,
            k=2.0,
        )
    )
    out.append(
        case(
            "localResponseNorm/4",
            "localResponseNorm",
            [x],
            to_numpy(F.local_response_norm(to_torch(x), 4)),
            size=4,
            alpha=1e-4,
            beta=0.75,
            k=1.0,
        )
    )
    out.append(case("normalize/p2", "normalize", [x], to_numpy(F.normalize(to_torch(x), p=2.0, dim=1)), p=2.0, axis=1))
    out.append(
        case("normalize/p1", "normalize", [x], to_numpy(F.normalize(to_torch(x), p=1.0, dim=-1)), p=1.0, axis=-1)
    )

    # Weight normalisation: torch.nn.utils.weight_norm decomposes a Linear and a Conv1d weight.
    torch.manual_seed(1)
    lin = nn.Linear(4, 3).double()
    wn = nn.utils.weight_norm(lin)  # pyright: ignore[reportUnknownMemberType]
    xl = n(5, 4)
    out.append(
        case(
            "weightNorm/linear",
            "weightNormLinear",
            [xl, to_numpy(attr(wn, "weight_g")), to_numpy(attr(wn, "weight_v")), to_numpy(attr(lin, "bias"))],
            to_numpy(wn(to_torch(xl))),
        )
    )
    # Spectral normalisation: one training forward pass updates u and v and normalises by sigma.
    torch.manual_seed(2)
    lin = nn.Linear(4, 3).double()
    sn = nn.utils.spectral_norm(lin)  # pyright: ignore[reportUnknownMemberType]
    u0, v0 = to_numpy(attr(sn, "weight_u")).copy(), to_numpy(attr(sn, "weight_v")).copy()
    worig, bias = to_numpy(attr(sn, "weight_orig")), to_numpy(attr(sn, "bias"))
    y = sn(to_torch(xl))
    inputs = [xl, worig, bias, u0, v0]
    out.append(case("spectralNorm/train/output", "spectralNormLinear", inputs, to_numpy(y), part="output"))
    out.append(case("spectralNorm/train/u", "spectralNormLinear", inputs, to_numpy(attr(sn, "weight_u")), part="u"))
    out.append(case("spectralNorm/train/v", "spectralNormLinear", inputs, to_numpy(attr(sn, "weight_v")), part="v"))
    sn.eval()
    u1, v1 = to_numpy(attr(sn, "weight_u")).copy(), to_numpy(attr(sn, "weight_v")).copy()
    ye = sn(to_torch(xl))
    out.append(case("spectralNorm/eval/output", "spectralNormLinearEval", [xl, worig, bias, u1, v1], to_numpy(ye)))
    return out
