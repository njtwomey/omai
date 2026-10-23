"""Activations and a small network against torch (float64): forward values, the loss and parameter gradients."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def t(x: FloatArray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64)


def value(x: torch.Tensor) -> FloatArray:
    return x.detach().numpy()


def cases() -> list[Case]:
    g = rng(60)
    x = np.concatenate([g.normal(size=20) * 3, [0.0, -25.0, 25.0]])
    xt = t(x)
    out: list[Case] = [
        case("relu", "relu", [x], value(F.relu(xt))),
        case("leakyRelu", "leakyRelu", [x], value(F.leaky_relu(xt, 0.1)), slope=0.1),
        case("elu", "elu", [x], value(F.elu(xt, 0.7)), alpha=0.7),
        case("selu", "selu", [x], value(F.selu(xt))),
        case("gelu", "gelu", [x], value(F.gelu(xt))),
        case("gelu/tanh", "gelu", [x], value(F.gelu(xt, approximate="tanh")), approximate="tanh"),
        case("silu", "silu", [x], value(F.silu(xt))),
        case("mish", "mish", [x], value(F.mish(xt))),
        case("softplus", "softplus", [x], value(F.softplus(xt, beta=2.0, threshold=10.0)), beta=2.0, threshold=10.0),
        case("hardtanh", "hardtanh", [x], value(F.hardtanh(xt))),
        case("hardsigmoid", "hardsigmoid", [x], value(F.hardsigmoid(xt))),
        case(
            "glu",
            "glu",
            [x[:20].reshape(4, 5).repeat(2, axis=1)],
            value(F.glu(t(x[:20].reshape(4, 5).repeat(2, axis=1)))),
        ),
    ]
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 2)).double()
    inputs = g.normal(size=(6, 3))
    targets = g.normal(size=(6, 2))
    prediction = net(t(inputs))
    loss = F.mse_loss(prediction, t(targets))
    loss.backward()  # pyright: ignore[reportUnknownMemberType]  (torch types backward loosely)
    state = {k: value(v) for k, v in net.state_dict().items()}
    grads = {k: value(p.grad) for k, p in net.named_parameters() if p.grad is not None}
    out.append(case("mlp/forward", "mlpForward", [inputs], value(prediction), state=state))
    out.append(case("mlp/loss", "mlpLoss", [inputs, targets], value(loss), state=state))
    for name, grad in grads.items():
        out.append(case(f"mlp/grad/{name}", "mlpGrad", [inputs, targets], grad, state=state, parameter=name))
    return out
