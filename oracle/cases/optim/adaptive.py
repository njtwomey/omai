"""Adaptive optimiser trajectories (optim/adaptive.ts) against torch.optim and optax on a fixed smooth objective."""

from typing import Any

import optax  # pyright: ignore[reportMissingTypeStubs]
import torch
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.optim_trajectories import (
    A_MATRIX,
    C_MATRIX,
    P0,
    P0_MATRIX,
    STEPS,
    A,
    C,
    TorchFactory,
    optax_trajectory,
    torch_trajectory,
)

REFERENCE: dict[str, str] = {"torch": torch.__version__, "optax": optax.__version__}

TORCH: list[tuple[str, dict[str, Any], TorchFactory]] = [
    ("adam", {"learningRate": 0.1}, lambda ps: torch.optim.Adam(ps, lr=0.1)),
    ("adam", {"learningRate": 0.1, "amsgrad": True}, lambda ps: torch.optim.Adam(ps, lr=0.1, amsgrad=True)),
    (
        "adam",
        {"learningRate": 0.1, "b1": 0.8, "b2": 0.95, "weightDecay": 0.1},
        lambda ps: torch.optim.Adam(ps, lr=0.1, betas=(0.8, 0.95), weight_decay=0.1),
    ),
    ("adamw", {"learningRate": 0.1, "weightDecay": 0.05}, lambda ps: torch.optim.AdamW(ps, lr=0.1, weight_decay=0.05)),
    (
        "amsgrad",
        {"learningRate": 0.1, "b2": 0.9},
        lambda ps: torch.optim.Adam(ps, lr=0.1, betas=(0.9, 0.9), amsgrad=True),
    ),
    ("adamax", {"learningRate": 0.1}, lambda ps: torch.optim.Adamax(ps, lr=0.1)),
    (
        "adamax",
        {"learningRate": 0.1, "b1": 0.7, "b2": 0.9, "weightDecay": 0.1},
        lambda ps: torch.optim.Adamax(ps, lr=0.1, betas=(0.7, 0.9), weight_decay=0.1),
    ),
    ("nadam", {"learningRate": 0.1}, lambda ps: torch.optim.NAdam(ps, lr=0.1)),
    (
        "nadam",
        {"learningRate": 0.1, "weightDecay": 0.1, "momentumDecay": 0.01},
        lambda ps: torch.optim.NAdam(ps, lr=0.1, weight_decay=0.1, momentum_decay=0.01),
    ),
    (
        "nadam",
        {"learningRate": 0.1, "weightDecay": 0.1, "decoupledWeightDecay": True},
        lambda ps: torch.optim.NAdam(ps, lr=0.1, weight_decay=0.1, decoupled_weight_decay=True),
    ),
    ("radam", {"learningRate": 0.1}, lambda ps: torch.optim.RAdam(ps, lr=0.1)),
    (
        "radam",
        {"learningRate": 0.1, "b2": 0.9, "weightDecay": 0.1},
        lambda ps: torch.optim.RAdam(ps, lr=0.1, betas=(0.9, 0.9), weight_decay=0.1),
    ),
    (
        "radam",
        {"learningRate": 0.1, "weightDecay": 0.1, "decoupledWeightDecay": True},
        lambda ps: torch.optim.RAdam(ps, lr=0.1, weight_decay=0.1, decoupled_weight_decay=True),
    ),
    ("adafactor", {"learningRate": 0.1}, lambda ps: torch.optim.Adafactor(ps, lr=0.1)),
    (
        "adafactor",
        {"learningRate": 0.05, "beta2Decay": -0.5, "eps": [1e-30, 1e-2], "d": 1.5, "weightDecay": 0.1},
        lambda ps: torch.optim.Adafactor(ps, lr=0.05, beta2_decay=-0.5, eps=(1e-30, 1e-2), d=1.5, weight_decay=0.1),
    ),
]

OPTAX: list[tuple[str, dict[str, Any], optax.GradientTransformation]] = [
    ("lion", {"learningRate": 0.05}, optax.lion(0.05)),
    (
        "lion",
        {"learningRate": 0.05, "b1": 0.8, "b2": 0.9, "weightDecay": 0.1},
        optax.lion(0.05, 0.8, 0.9, weight_decay=0.1),
    ),
    ("lamb", {"learningRate": 0.05}, optax.lamb(0.05)),
    ("lamb", {"learningRate": 0.05, "weightDecay": 0.1, "eps": 1e-8}, optax.lamb(0.05, eps=1e-8, weight_decay=0.1)),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for i, (name, options, make) in enumerate(TORCH):
        path = torch_trajectory(make)
        out.append(case(f"{name}/{i}", name, [P0, A, C], path, rtol=1e-10, atol=1e-12, steps=STEPS, **options))
        if name in ("adafactor", "adamax", "nadam", "radam"):
            matrix = torch_trajectory(make, P0_MATRIX, A_MATRIX, C_MATRIX)
            out.append(
                case(
                    f"{name}/{i}/matrix",
                    name,
                    [P0_MATRIX, A_MATRIX, C_MATRIX],
                    matrix,
                    rtol=1e-10,
                    atol=1e-12,
                    steps=STEPS,
                    **options,
                )
            )
    for i, (name, options, opt) in enumerate(OPTAX):
        path = optax_trajectory(opt)
        out.append(case(f"{name}/{i}", name, [P0, A, C], path, rtol=1e-10, atol=1e-12, steps=STEPS, **options))
    return out
