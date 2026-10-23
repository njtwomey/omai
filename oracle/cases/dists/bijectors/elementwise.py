"""Elementwise bijectors (dists/bijectors/elementwise.ts) against torch.distributions.transforms."""

import numpy as np
import torch
from torch.distributions import transforms as T
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def cases() -> list[Case]:
    g = rng(40)
    x = g.normal(size=7) * 1.5
    pos = np.exp(g.normal(size=7))
    transforms: list[tuple[str, T.Transform, np.ndarray]] = [
        ("exp", T.ExpTransform(), x),
        ("softplus", T.SoftplusTransform(), x),
        ("sigmoid", T.SigmoidTransform(), x),
        ("tanh", T.TanhTransform(), x),
        ("affine", T.AffineTransform(0.5, -2.0), x),
        ("power", T.PowerTransform(torch.tensor(1.7, dtype=torch.float64)), pos),
    ]
    out: list[Case] = []
    for name, t, at in transforms:
        xt = to_torch(at)
        y = t(xt)
        assert y is not None
        out.append(case(f"{name}/forward", f"{name}.forward", [at], to_numpy(y)))
        out.append(case(f"{name}/inverse", f"{name}.inverse", [to_numpy(y)], at, rtol=1e-10))
        out.append(
            case(
                f"{name}/logDet",
                f"{name}.forwardLogDetJacobian",
                [at],
                to_numpy(t.log_abs_det_jacobian(xt, y)),
                rtol=1e-10,
                atol=1e-14,
            )
        )
    return out
