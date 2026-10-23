"""Optimiser trajectories against torch.optim on a fixed smooth objective."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"torch": torch.__version__}

STEPS = 12
P0 = np.array([1.5, -2.0, 0.3, 4.0])
A = np.array([1.0, 3.0, 0.5, 2.0])
C = np.array([0.5, 1.0, -1.0, 2.0])

type Factory = Callable[[list[torch.Tensor]], torch.optim.Optimizer]


def objective(p: torch.Tensor) -> torch.Tensor:
    """0.5 sum a (p - c)^2 + 0.05 sum p^4: the same objective the TypeScript test differentiates."""
    a = torch.tensor(A, dtype=torch.float64)
    c = torch.tensor(C, dtype=torch.float64)
    return 0.5 * (a * (p - c) ** 2).sum() + 0.05 * (p**4).sum()


def trajectory(make: Factory) -> FloatArray:
    p = torch.tensor(P0, dtype=torch.float64, requires_grad=True)
    opt = make([p])
    out: list[FloatArray] = []
    for _ in range(STEPS):
        opt.zero_grad()
        objective(p).backward()  # pyright: ignore[reportUnknownMemberType]  (torch types backward loosely)
        opt.step()
        out.append(p.detach().numpy().copy())
    return np.stack(out)


OPTIMISERS: list[tuple[str, dict[str, Any], Factory]] = [
    ("sgd", {"learningRate": 0.05}, lambda ps: torch.optim.SGD(ps, lr=0.05)),
    ("sgd", {"learningRate": 0.05, "momentum": 0.9}, lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9)),
    (
        "sgd",
        {"learningRate": 0.05, "momentum": 0.9, "nesterov": True},
        lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9, nesterov=True),
    ),
    ("sgd", {"learningRate": 0.05, "weightDecay": 0.1}, lambda ps: torch.optim.SGD(ps, lr=0.05, weight_decay=0.1)),
    ("adam", {"learningRate": 0.1}, lambda ps: torch.optim.Adam(ps, lr=0.1)),
    ("adam", {"learningRate": 0.1, "amsgrad": True}, lambda ps: torch.optim.Adam(ps, lr=0.1, amsgrad=True)),
    (
        "adam",
        {"learningRate": 0.1, "b1": 0.8, "b2": 0.95, "weightDecay": 0.1},
        lambda ps: torch.optim.Adam(ps, lr=0.1, betas=(0.8, 0.95), weight_decay=0.1),
    ),
    ("adamw", {"learningRate": 0.1, "weightDecay": 0.05}, lambda ps: torch.optim.AdamW(ps, lr=0.1, weight_decay=0.05)),
    ("rmsprop", {"learningRate": 0.01}, lambda ps: torch.optim.RMSprop(ps, lr=0.01)),
    (
        "rmsprop",
        {"learningRate": 0.01, "momentum": 0.5},
        lambda ps: torch.optim.RMSprop(ps, lr=0.01, momentum=0.5),
    ),
    ("adagrad", {"learningRate": 0.3}, lambda ps: torch.optim.Adagrad(ps, lr=0.3)),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for i, (name, options, make) in enumerate(OPTIMISERS):
        out.append(
            case(f"{name}/{i}", name, [P0, A, C], trajectory(make), rtol=1e-10, atol=1e-12, steps=STEPS, **options)
        )
    return out
