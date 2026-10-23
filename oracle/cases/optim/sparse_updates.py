"""Sparse-gradient optimisers (optim/sparse-updates.ts) against torch.optim on sparse (COO) gradients.

Gradients of the shared objective are masked at each step, so some entries (or whole rows) are inactive; torch sees
them as sparse tensors, element-wise or row-wise (sparse_dim=1, as embedding gradients are).
"""

from typing import Any

import torch
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.optim_helpers import (
    A_MATRIX,
    C_MATRIX,
    P0,
    P0_MATRIX,
    STEPS,
    A,
    C,
    TorchFactory,
    every_third,
    rows_and_entries,
    torch_trajectory,
)

REFERENCE: dict[str, str] = {"torch": torch.__version__}

TORCH: list[tuple[str, dict[str, Any], TorchFactory]] = [
    ("sparseAdam", {"learningRate": 0.1}, lambda ps: torch.optim.SparseAdam(ps, lr=0.1)),
    (
        "sparseAdam",
        {"learningRate": 0.1, "b1": 0.8, "b2": 0.95, "eps": 1e-6},
        lambda ps: torch.optim.SparseAdam(ps, lr=0.1, betas=(0.8, 0.95), eps=1e-6),
    ),
    ("lazyAdagrad", {"learningRate": 0.3}, lambda ps: torch.optim.Adagrad(ps, lr=0.3)),
    (
        "lazyAdagrad",
        {"learningRate": 0.3, "lrDecay": 0.1, "initialAccumulator": 0.5},
        lambda ps: torch.optim.Adagrad(ps, lr=0.3, lr_decay=0.1, initial_accumulator_value=0.5),
    ),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for i, (name, options, make) in enumerate(TORCH):
        vector = torch_trajectory(make, mask=every_third(P0.shape), sparse=True)
        out.append(
            case(
                f"{name}/{i}", name, [P0, A, C], vector, rtol=1e-10, atol=1e-12, steps=STEPS, mask="element", **options
            )
        )
        matrix = torch_trajectory(make, P0_MATRIX, A_MATRIX, C_MATRIX, mask=every_third(P0_MATRIX.shape), sparse=True)
        out.append(
            case(
                f"{name}/{i}/matrix",
                name,
                [P0_MATRIX, A_MATRIX, C_MATRIX],
                matrix,
                rtol=1e-10,
                atol=1e-12,
                steps=STEPS,
                mask="element",
                **options,
            )
        )
        rows = torch_trajectory(
            make,
            P0_MATRIX,
            A_MATRIX,
            C_MATRIX,
            mask=rows_and_entries((2, 3)),
            sparse=True,
            sparse_dim=1,
        )
        out.append(
            case(
                f"{name}/{i}/rows",
                name,
                [P0_MATRIX, A_MATRIX, C_MATRIX],
                rows,
                rtol=1e-10,
                atol=1e-12,
                steps=STEPS,
                mask="rows",
                rowwise=True,
                **options,
            )
        )
    return out
