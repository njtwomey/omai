"""Structured bijectors (dists/bijectors/structured.ts) against torch.distributions.transforms."""

import torch
from torch.distributions import transforms as T
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def cases() -> list[Case]:
    g = rng(41)
    out: list[Case] = []
    # Stick-breaking onto the simplex (batch of 3 vectors of length 4).
    x = g.normal(size=(3, 4))
    t = T.StickBreakingTransform()
    xt = to_torch(x)
    y = t(xt)
    assert y is not None
    out.append(case("stickBreaking/forward", "stickBreaking.forward", [x], to_numpy(y)))
    out.append(case("stickBreaking/inverse", "stickBreaking.inverse", [to_numpy(y)], x, rtol=1e-9))
    out.append(
        case(
            "stickBreaking/logDet",
            "stickBreaking.forwardLogDetJacobian",
            [x],
            to_numpy(t.log_abs_det_jacobian(xt, y)),
            rtol=1e-10,
        )
    )
    # Correlation Cholesky factors from 6 unconstrained values (a 4 x 4 factor), batch of 2.
    z = g.normal(size=(2, 6))
    c = T.CorrCholeskyTransform()
    zt = to_torch(z)
    lc = c(zt)
    assert lc is not None
    out.append(case("correlationCholesky/forward", "correlationCholesky.forward", [z], to_numpy(lc)))
    out.append(case("correlationCholesky/inverse", "correlationCholesky.inverse", [to_numpy(lc)], z, rtol=1e-9))
    out.append(
        case(
            "correlationCholesky/logDet",
            "correlationCholesky.forwardLogDetJacobian",
            [z],
            to_numpy(c.log_abs_det_jacobian(zt, lc)),
            rtol=1e-10,
        )
    )
    return out
