"""Gradient attributions against captum on small torch networks with fixed weights: saliency, input times gradient,
integrated gradients (Gauss-Legendre and Riemann rules, with the convergence delta) and DeepLIFT's rescale rule."""

import captum
import numpy as np
import torch
from captum.attr import DeepLift, InputXGradient, IntegratedGradients, Saliency
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng
from whitebox_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"captum": captum.__version__, "torch": torch.__version__}


def network(w1: FloatArray, b1: FloatArray, w2: FloatArray, b2: FloatArray, act: str) -> torch.nn.Sequential:
    """Linear(d, h) -> activation -> Linear(h, k) in float64 with the given weights (rows of w are inputs)."""
    first = torch.nn.Linear(w1.shape[0], w1.shape[1]).double()
    second = torch.nn.Linear(w2.shape[0], w2.shape[1]).double()
    with torch.no_grad():
        first.weight.copy_(to_torch(w1.T))
        first.bias.copy_(to_torch(b1))
        second.weight.copy_(to_torch(w2.T))
        second.bias.copy_(to_torch(b2))
    activation = torch.nn.ReLU() if act == "relu" else torch.nn.Tanh()
    return torch.nn.Sequential(first, activation, second)


def cases() -> list[Case]:
    g = rng(4344)
    d, h, k = 3, 5, 2
    w1, b1 = g.normal(size=(d, h)), g.normal(size=h) * 0.5
    w2, b2 = g.normal(size=(h, k)), g.normal(size=k) * 0.5
    x = g.normal(size=(4, d))
    base = g.normal(size=(4, d)) * 0.3
    out: list[Case] = []
    for act in ["relu", "tanh"]:
        net = network(w1, b1, w2, b2, act)
        xt = to_torch(x).requires_grad_()
        weights = [w1, b1, w2, b2, x, base]
        for target in [0, 1]:
            sal = Saliency(net).attribute(xt, target=target)
            ixg = InputXGradient(net).attribute(xt, target=target)
            out.append(
                case(
                    f"{act}/saliency/{target}",
                    "saliency",
                    weights,
                    {"saliency": to_numpy(sal), "inputXGradient": to_numpy(ixg)},
                    act=act,
                    target=target,
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
        for method, steps in [
            ("gausslegendre", 50),
            ("riemann_trapezoid", 20),
            ("riemann_left", 10),
            ("riemann_middle", 10),
            ("riemann_right", 10),
        ]:
            ig, delta = IntegratedGradients(net).attribute(
                xt, baselines=to_torch(base), target=1, n_steps=steps, method=method, return_convergence_delta=True
            )
            # captum builds the Riemann positions with float32 linspace; the attributions agree to that precision.
            out.append(
                case(
                    f"{act}/ig/{method}",
                    "integratedGradients",
                    weights,
                    {"attributions": to_numpy(ig), "delta": to_numpy(delta)},
                    act=act,
                    method=method,
                    steps=steps,
                    rtol=1e-6,
                    atol=1e-7,
                )
            )
        for label, baseline in [("zero", np.zeros_like(x)), ("random", base)]:
            dl = DeepLift(net).attribute(xt, baselines=to_torch(baseline), target=0)
            out.append(
                case(
                    f"{act}/deeplift/{label}",
                    "deepLift",
                    [w1, b1, w2, b2, x, baseline],
                    {"attributions": to_numpy(dl)},
                    act=act,
                    rtol=1e-10,
                    atol=1e-12,
                )
            )
    return out
