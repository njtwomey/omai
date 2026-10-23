"""Conversions between numpy arrays and torch tensors for the torch reference cases (oracle-torch group)."""

import numpy as np
import torch

from whitebox_oracle.helpers import FloatArray


def to_torch(x: FloatArray) -> torch.Tensor:
    """A float64 tensor holding a copy of x."""
    return torch.tensor(x, dtype=torch.float64)


def to_numpy(x: torch.Tensor) -> FloatArray:
    """A float64 array holding a detached copy of x."""
    return x.detach().numpy().astype(np.float64)
