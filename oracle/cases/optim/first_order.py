"""First-order optimiser trajectories (optim/first-order.ts) against torch.optim and optax on a smooth objective."""

from typing import Any

import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import torch
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray
from whitebox_oracle.optim_trajectories import (
    A_MATRIX,
    C_MATRIX,
    P0,
    P0_MATRIX,
    STEPS,
    A,
    C,
    TorchFactory,
    objective_torch,
    optax_trajectory,
    torch_trajectory,
)

REFERENCE: dict[str, str] = {"torch": torch.__version__, "optax": optax.__version__}

TORCH: list[tuple[str, dict[str, Any], TorchFactory]] = [
    ("sgd", {"learningRate": 0.05}, lambda ps: torch.optim.SGD(ps, lr=0.05)),
    ("sgd", {"learningRate": 0.05, "momentum": 0.9}, lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9)),
    (
        "sgd",
        {"learningRate": 0.05, "momentum": 0.9, "nesterov": True},
        lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9, nesterov=True),
    ),
    (
        "sgd",
        {"learningRate": 0.05, "momentum": 0.8, "dampening": 0.3},
        lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.8, dampening=0.3),
    ),
    ("sgd", {"learningRate": 0.05, "weightDecay": 0.1}, lambda ps: torch.optim.SGD(ps, lr=0.05, weight_decay=0.1)),
    ("rmsprop", {"learningRate": 0.01}, lambda ps: torch.optim.RMSprop(ps, lr=0.01)),
    ("rmsprop", {"learningRate": 0.01, "momentum": 0.5}, lambda ps: torch.optim.RMSprop(ps, lr=0.01, momentum=0.5)),
    (
        "rmsprop",
        {"learningRate": 0.01, "centered": True},
        lambda ps: torch.optim.RMSprop(ps, lr=0.01, centered=True),
    ),
    (
        "rmsprop",
        {"learningRate": 0.01, "centered": True, "momentum": 0.6, "decay": 0.9, "weightDecay": 0.05},
        lambda ps: torch.optim.RMSprop(ps, lr=0.01, centered=True, momentum=0.6, alpha=0.9, weight_decay=0.05),
    ),
    ("adagrad", {"learningRate": 0.3}, lambda ps: torch.optim.Adagrad(ps, lr=0.3)),
    (
        "adagrad",
        {"learningRate": 0.3, "lrDecay": 0.1, "initialAccumulator": 0.5, "weightDecay": 0.05},
        lambda ps: torch.optim.Adagrad(ps, lr=0.3, lr_decay=0.1, initial_accumulator_value=0.5, weight_decay=0.05),
    ),
    ("adadelta", {}, lambda ps: torch.optim.Adadelta(ps)),
    (
        "adadelta",
        {"learningRate": 0.5, "rho": 0.8, "eps": 1e-4, "weightDecay": 0.1},
        lambda ps: torch.optim.Adadelta(ps, lr=0.5, rho=0.8, eps=1e-4, weight_decay=0.1),
    ),
    ("rprop", {}, lambda ps: torch.optim.Rprop(ps)),
    (
        "rprop",
        {"learningRate": 0.3, "etas": [0.3, 1.5], "stepSizes": [0.01, 0.6]},
        lambda ps: torch.optim.Rprop(ps, lr=0.3, etas=(0.3, 1.5), step_sizes=(0.01, 0.6)),
    ),
]


def asgd_case(name: str, options: dict[str, Any], p0: FloatArray, a: FloatArray, c: FloatArray) -> Case:
    """ASGD's parameter path and its averaged iterate after the last step (torch's scalar state in float64)."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return asgd_run(name, options, p0, a, c)
    finally:
        torch.set_default_dtype(previous)


def asgd_run(name: str, options: dict[str, Any], p0: FloatArray, a: FloatArray, c: FloatArray) -> Case:
    """Run ASGD for the case."""
    p = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.ASGD(
        [p],
        lr=options["learningRate"],
        lambd=options["lambd"],
        alpha=options["alpha"],
        t0=options["t0"],
        weight_decay=options.get("weightDecay", 0.0),
    )
    path: list[FloatArray] = []
    for _ in range(STEPS):
        opt.zero_grad()
        objective_torch(p, a, c).backward()  # pyright: ignore[reportUnknownMemberType]
        opt.step()
        path.append(p.detach().numpy().copy())
    ax = opt.state[p]["ax"].detach().numpy().copy()
    output = {"path": np.stack(path), "ax": ax}
    return case(name, "asgd", [p0, a, c], output, rtol=1e-10, atol=1e-12, steps=STEPS, **options)


def cases() -> list[Case]:
    out: list[Case] = []
    for i, (name, options, make) in enumerate(TORCH):
        out.append(
            case(
                f"{name}/{i}", name, [P0, A, C], torch_trajectory(make), rtol=1e-10, atol=1e-12, steps=STEPS, **options
            )
        )
        if name in ("rmsprop", "adadelta", "rprop"):
            path = torch_trajectory(make, P0_MATRIX, A_MATRIX, C_MATRIX)
            out.append(
                case(
                    f"{name}/{i}/matrix",
                    name,
                    [P0_MATRIX, A_MATRIX, C_MATRIX],
                    path,
                    rtol=1e-10,
                    atol=1e-12,
                    steps=STEPS,
                    **options,
                )
            )
    out.append(asgd_case("asgd/default", {"learningRate": 0.05, "lambd": 1e-4, "alpha": 0.75, "t0": 1e6}, P0, A, C))
    out.append(
        asgd_case(
            "asgd/averaging",
            {"learningRate": 0.05, "lambd": 0.01, "alpha": 0.6, "t0": 4, "weightDecay": 0.05},
            P0,
            A,
            C,
        )
    )
    out.append(
        case(
            "signSgd/0",
            "signSgd",
            [P0, A, C],
            optax_trajectory(optax.sign_sgd(0.05)),
            rtol=1e-12,
            atol=1e-12,
            steps=STEPS,
            learningRate=0.05,
        )
    )
    out.append(
        case(
            "signSgd/signum",
            "signSgd",
            [P0, A, C],
            optax_trajectory(optax.signum(0.05, beta=0.8)),
            rtol=1e-12,
            atol=1e-12,
            steps=STEPS,
            learningRate=0.05,
            momentum=0.8,
        )
    )
    return out
