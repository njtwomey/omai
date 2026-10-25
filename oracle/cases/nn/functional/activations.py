"""Activations against torch.nn.functional (float64), beyond those in nn/torch_reference.py: ReLU6, hard swish,
CELU, softsign, log-sigmoid, PReLU, the shrinkage functions, threshold, and the GLU variants (SwiGLU, GeGLU, ReGLU)
written as torch users write them."""

import numpy as np
import torch
import torch.nn.functional as F
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from mdml_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def cases() -> list[Case]:
    g = rng(345)
    x = np.concatenate([g.normal(size=16) * 4, [0.0, -3.0, 3.0, 6.0, -40.0, 40.0]])
    xt = to_torch(x)
    out: list[Case] = [
        case("relu6", "relu6", [x], to_numpy(F.relu6(xt))),
        case("hardswish", "hardswish", [x], to_numpy(F.hardswish(xt))),
        case("celu", "celu", [x], to_numpy(F.celu(xt, 0.7)), alpha=0.7),
        case("softsign", "softsign", [x], to_numpy(F.softsign(xt))),
        case("logsigmoid", "logsigmoid", [x], to_numpy(F.logsigmoid(xt))),
        case("tanhshrink", "tanhshrink", [x], to_numpy(F.tanhshrink(xt))),
        case("softshrink", "softshrink", [x], to_numpy(F.softshrink(xt, 0.8)), lambd=0.8),
        case("hardshrink", "hardshrink", [x], to_numpy(F.hardshrink(xt, 0.8)), lambd=0.8),
        case("threshold", "threshold", [x], to_numpy(F.threshold(xt, 0.5, -2.0)), threshold=0.5, value=-2.0),
    ]
    xc = np.asarray(g.normal(size=(2, 3, 4)), dtype=np.float64)
    wc = np.asarray(g.uniform(0.0, 0.5, size=3), dtype=np.float64)
    out.append(case("prelu/channels", "prelu", [xc, wc], to_numpy(F.prelu(to_torch(xc), to_torch(wc)))))
    out.append(case("prelu/shared", "prelu", [xc, wc[:1]], to_numpy(F.prelu(to_torch(xc), to_torch(wc[:1])))))
    xg = np.asarray(g.normal(size=(3, 6)), dtype=np.float64)
    a, b = to_torch(xg).chunk(2, dim=-1)
    out.append(case("swiglu", "swiglu", [xg], to_numpy(F.silu(a) * b)))
    out.append(case("geglu", "geglu", [xg], to_numpy(F.gelu(a) * b)))
    out.append(case("reglu", "reglu", [xg], to_numpy(F.relu(a) * b)))
    return out
