"""Adjoint gradients of an ODE solve against torchdiffeq's odeint_adjoint (a gradient fixture: the reference library
itself differentiates the solve, section 9 rule 4).

Lotka-Volterra with four parameters; the loss is the sum of squares of the states at the output times. Both libraries
solve at tight tolerances, so the gradients agree to the solver tolerance.
"""

from typing import cast

import torch
import torchdiffeq
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"torch": torch.__version__, "torchdiffeq": torchdiffeq.__version__}


class Lotka(torch.nn.Module):
    def __init__(self, params: list[float]) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(params, dtype=torch.float64))

    def forward(self, _t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a, b = y[0], y[1]
        alpha, beta, delta, gamma = self.p[0], self.p[1], self.p[2], self.p[3]
        return torch.stack([alpha * a - beta * a * b, delta * a * b - gamma * b])


def gradients(y0: list[float], params: list[float], times: list[float]) -> tuple[FloatArray, FloatArray, FloatArray]:
    field = Lotka(params)
    y = torch.tensor(y0, dtype=torch.float64, requires_grad=True)
    t = torch.tensor(times, dtype=torch.float64)
    ys = cast(torch.Tensor, torchdiffeq.odeint_adjoint(field, y, t, rtol=1e-11, atol=1e-13, method="dopri5"))
    loss = (ys**2).sum()
    loss.backward()
    assert y.grad is not None and field.p.grad is not None
    return ys.detach().numpy(), y.grad.numpy(), field.p.grad.detach().numpy()


def cases() -> list[Case]:
    out: list[Case] = []
    for label, (y0, params, times) in {
        "lotka": ([2.0, 1.0], [1.5, 1.0, 1.0, 3.0], [0.0, 0.5, 1.5]),
        "lotka-long": ([1.0, 1.5], [1.1, 0.4, 0.1, 0.4], [0.0, 2.0, 5.0, 8.0]),
    }.items():
        ys, gy, gp = gradients(y0, params, times)
        common: dict[str, object] = {"y0": y0, "params": params, "times": times}
        out.append(case(f"{label}/states", "states", [], ys, rtol=1e-8, atol=1e-10, **common))
        for adjoint in ["backsolve", "interpolated", "checkpointed"]:
            out.append(
                case(f"{label}/{adjoint}/y0", "grad_y0", [], gy, rtol=1e-6, atol=1e-8, adjoint=adjoint, **common)
            )
            out.append(
                case(f"{label}/{adjoint}/params", "grad_p", [], gp, rtol=1e-6, atol=1e-8, adjoint=adjoint, **common)
            )
    return out
