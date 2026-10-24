"""The optimiser test objective and trajectory runners for torch.optim and optax (oracle-torch group).

The objective is 0.5 sum a (p - c)^2 + 0.05 sum p^4, elementwise, so it applies to vectors and matrices alike; the
TypeScript tests differentiate the same function. A gradient mask (one per step) makes the gradients sparse for the
sparse-gradient optimisers.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import torch

from whitebox_oracle.helpers import FloatArray

jax.config.update("jax_enable_x64", True)

STEPS = 12
P0 = np.array([1.5, -2.0, 0.3, 4.0])
A = np.array([1.0, 3.0, 0.5, 2.0])
C = np.array([0.5, 1.0, -1.0, 2.0])
P0_MATRIX = np.array([[1.5, -2.0, 0.3], [4.0, -0.7, 2.2]])
A_MATRIX = np.array([[1.0, 3.0, 0.5], [2.0, 1.5, 0.8]])
C_MATRIX = np.array([[0.5, 1.0, -1.0], [2.0, 0.0, 1.0]])

type TorchFactory = Callable[[list[torch.Tensor]], torch.optim.Optimizer]
type Mask = Callable[[int], FloatArray] | None


def objective_torch(p: torch.Tensor, a: FloatArray, c: FloatArray) -> torch.Tensor:
    """The objective in torch."""
    at = torch.tensor(a, dtype=torch.float64)
    ct = torch.tensor(c, dtype=torch.float64)
    return 0.5 * (at * (p - ct) ** 2).sum() + 0.05 * (p**4).sum()


def gradient(p: FloatArray, a: FloatArray, c: FloatArray) -> FloatArray:
    """The objective's gradient a (p - c) + 0.2 p^3, in numpy."""
    return a * (p - c) + 0.2 * p**3


def every_third(shape: tuple[int, ...]) -> Callable[[int], FloatArray]:
    """The mask that drops entry i at step t when (i + t) % 3 == 0 (flat index), making gradients sparse."""

    def mask(t: int) -> FloatArray:
        idx = np.arange(int(np.prod(shape))).reshape(shape)
        return ((idx + t) % 3 != 0).astype(np.float64)

    return mask


def rows_and_entries(shape: tuple[int, int]) -> Callable[[int], FloatArray]:
    """Row r is dropped at step t when (r + t) % 3 == 0, and entry (r, j) of a kept row when (r + j + t) % 4 == 0."""

    def mask(t: int) -> FloatArray:
        r, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        return (((r + t) % 3 != 0) & ((r + j + t) % 4 != 0)).astype(np.float64)

    return mask


def torch_trajectory(
    make: TorchFactory,
    p0: FloatArray = P0,
    a: FloatArray = A,
    c: FloatArray = C,
    steps: int = STEPS,
    mask: Mask = None,
    sparse: bool = False,
    sparse_dim: int | None = None,
) -> FloatArray:
    """The parameters after each of `steps` optimiser steps from p0 (gradients masked, and sparse, if asked).

    torch keeps scalar optimiser state (NAdam's momentum product, ASGD's step size) in the default dtype, so the
    default is float64 while the optimiser runs.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        p = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
        opt = make([p])
        out: list[FloatArray] = []
        for t in range(steps):
            opt.zero_grad()
            objective_torch(p, a, c).backward()  # pyright: ignore[reportUnknownMemberType]
            grad = p.grad
            if grad is not None and mask is not None:
                masked = grad * torch.tensor(mask(t), dtype=torch.float64)
                if not sparse:
                    p.grad = masked
                else:
                    p.grad = masked.to_sparse() if sparse_dim is None else masked.to_sparse(sparse_dim=sparse_dim)
            opt.step()
            out.append(p.detach().numpy().copy())
        return np.stack(out)
    finally:
        torch.set_default_dtype(previous)


def optax_trajectory(
    opt: optax.GradientTransformation,
    p0: FloatArray = P0,
    a: FloatArray = A,
    c: FloatArray = C,
    steps: int = STEPS,
) -> FloatArray:
    """The parameters after each optax update from p0."""
    p = jnp.asarray(p0, dtype=jnp.float64)
    state = opt.init(p)
    out: list[FloatArray] = []
    for _ in range(steps):
        g = jnp.asarray(gradient(np.asarray(p), a, c))
        updates, state = opt.update(g, state, p)
        p = optax.apply_updates(p, updates)
        out.append(np.asarray(p, dtype=np.float64).copy())
    return np.stack(out)
