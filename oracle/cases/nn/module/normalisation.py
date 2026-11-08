"""Normalisation modules against torch.nn (float64): two training steps of BatchNorm1d/2d/3d (outputs and running
statistics, with an exponential and a cumulative average), then evaluation; InstanceNorm2d with running statistics;
LayerNorm, RMSNorm and GroupNorm with random affine parameters."""

import numpy as np
import torch
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.torch_helpers import to_numpy, to_torch
from torch import nn

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def state_of(m: nn.Module) -> dict[str, FloatArray]:
    return {k: to_numpy(v.double()) for k, v in m.state_dict().items()}


def steps(
    name: str, kind: str, m: nn.Module, xs: tuple[FloatArray, FloatArray, FloatArray], config: dict[str, object]
) -> list[Case]:
    """Two training forward passes and one evaluation pass, recording each output and the final state."""
    state = state_of(m)
    m.train()
    y1 = m(to_torch(xs[0]))
    y2 = m(to_torch(xs[1]))
    after = state_of(m)
    m.eval()
    y3 = m(to_torch(xs[2]))
    inputs = list(xs)
    parts = [("train1", to_numpy(y1)), ("train2", to_numpy(y2)), ("eval", to_numpy(y3))]
    parts += [(f"state:{k}", v) for k, v in after.items()]
    return [
        case(f"{name}/{part.replace(':', '/')}", kind, inputs, output, state=state, part=part, config=config)
        for part, output in parts
    ]


def randomise_affine(m: nn.Module, g: np.random.Generator) -> None:
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(to_torch(np.asarray(g.normal(size=tuple(p.shape)), dtype=np.float64)))


def cases() -> list[Case]:
    g = rng(343)

    def n(*shape: int) -> FloatArray:
        return np.asarray(g.normal(size=shape), dtype=np.float64)

    def three(*shape: int) -> tuple[FloatArray, FloatArray, FloatArray]:
        return n(*shape), n(*shape), n(2, *shape[1:])

    out: list[Case] = []
    bn1 = nn.BatchNorm1d(3, momentum=0.3).double()
    randomise_affine(bn1, g)
    out += steps("BatchNorm1d/2d-input", "BatchNorm1d", bn1, three(5, 3), {"features": 3, "momentum": 0.3})
    bn1c = nn.BatchNorm1d(2, momentum=None).double()
    out += steps("BatchNorm1d/cumulative", "BatchNorm1d", bn1c, three(3, 2, 4), {"features": 2, "momentum": None})
    bn2 = nn.BatchNorm2d(3, eps=1e-3).double()
    randomise_affine(bn2, g)
    out += steps("BatchNorm2d", "BatchNorm2d", bn2, three(2, 3, 3, 2), {"features": 3, "eps": 1e-3})
    bn3 = nn.BatchNorm3d(2).double()
    out += steps("BatchNorm3d", "BatchNorm3d", bn3, three(2, 2, 2, 2, 2), {"features": 2})
    bnn = nn.BatchNorm1d(3, affine=False, track_running_stats=False).double()
    out += steps(
        "BatchNorm1d/untracked", "BatchNorm1d", bnn, three(4, 3), {"features": 3, "affine": False, "track": False}
    )
    inn = nn.InstanceNorm2d(2, affine=True, track_running_stats=True).double()
    randomise_affine(inn, g)
    config: dict[str, object] = {"features": 2, "affine": True, "track": True}
    out += steps("InstanceNorm2d/tracked", "InstanceNorm2d", inn, three(2, 2, 3, 3), config)

    x = n(2, 3, 4)
    ln = nn.LayerNorm([3, 4]).double()
    randomise_affine(ln, g)
    out.append(
        case("LayerNorm", "LayerNorm", [x], to_numpy(ln(to_torch(x))), state=state_of(ln), config={"shape": [3, 4]})
    )
    rms = nn.RMSNorm(4, eps=1e-6).double()
    randomise_affine(rms, g)
    rms_config: dict[str, object] = {"shape": [4], "eps": 1e-6}
    out.append(case("RMSNorm", "RMSNorm", [x], to_numpy(rms(to_torch(x))), state=state_of(rms), config=rms_config))
    gn = nn.GroupNorm(2, 4).double()
    randomise_affine(gn, g)
    xg = n(2, 4, 3)
    gn_config: dict[str, object] = {"groups": 2, "channels": 4}
    out.append(case("GroupNorm", "GroupNorm", [xg], to_numpy(gn(to_torch(xg))), state=state_of(gn), config=gn_config))
    return out
