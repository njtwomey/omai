"""Gradient attributions against captum on small torch networks with fixed weights: saliency, input times gradient,
integrated gradients (Gauss-Legendre and Riemann rules, with the convergence delta), DeepLIFT's rescale rule, guided
backpropagation and deconvolution, layer-wise relevance propagation (epsilon and gamma rules; captum's alpha1-beta0
clamps weights rather than contributions, so it is not compared), the layer and neuron methods (layer conductance,
internal influence, layer gradient times activation, neuron conductance) and Grad-CAM on a small convolutional
network."""

import captum
import numpy as np
import torch
from captum.attr import (
    LRP,
    Deconvolution,
    DeepLift,
    GuidedBackprop,
    InputXGradient,
    IntegratedGradients,
    InternalInfluence,
    LayerConductance,
    LayerGradCam,
    LayerGradientXActivation,
    NeuronConductance,
    Saliency,
)
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.torch_helpers import to_numpy, to_torch

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


def single(t: object) -> torch.Tensor:
    """The one tensor a captum attribution returns for a single input."""
    assert isinstance(t, torch.Tensor)
    return t


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
    relu_net = network(w1, b1, w2, b2, "relu")
    weights = [w1, b1, w2, b2, x, base]
    xr = to_torch(x).requires_grad_()
    for name, method in [("guided", GuidedBackprop(relu_net)), ("deconvolution", Deconvolution(relu_net))]:
        attr = method.attribute(xr, target=1)
        out.append(case(f"relu/{name}", "backward", weights, {"attributions": to_numpy(attr)}, rule=name))
    for label, rules in [
        ("epsilon", (EpsilonRule(), EpsilonRule())),
        ("epsilon-0.1", (EpsilonRule(epsilon=0.1), EpsilonRule(epsilon=0.1))),
        ("gamma", (GammaRule(), EpsilonRule())),
    ]:
        lrp_net = network(w1, b1, w2, b2, "relu")
        lrp_net[0].rule = rules[0]  # type: ignore[attr-defined]
        lrp_net[2].rule = rules[1]  # type: ignore[attr-defined]
        attr = LRP(lrp_net).attribute(to_torch(x).requires_grad_(), target=0)
        out.append(case(f"lrp/{label}", "lrp", weights, {"attributions": to_numpy(attr)}, rule=label))
    for act in ["relu", "tanh"]:
        net = network(w1, b1, w2, b2, act)
        xt = to_torch(x).requires_grad_()
        bt = to_torch(base)
        layer = net[1]
        path = {"baselines": bt, "target": 1, "n_steps": 20}
        conductance = LayerConductance(net, layer).attribute(xt, **path)  # type: ignore[arg-type]
        influence = InternalInfluence(net, layer).attribute(xt, **path)  # type: ignore[arg-type]
        neuron = NeuronConductance(net, layer).attribute(xt, neuron_selector=2, **path)  # type: ignore[arg-type]
        outputs = {
            "conductance": to_numpy(single(conductance)),
            "influence": to_numpy(single(influence)),
            "gradientXActivation": to_numpy(single(LayerGradientXActivation(net, layer).attribute(xt, target=1))),
            "neuron": to_numpy(neuron),
        }
        # NeuronConductance scales by float32 step sizes.
        out.append(case(f"{act}/layer", "layer", weights, outputs, act=act, rtol=1e-6, atol=1e-7))
    cw = g.normal(size=(2, 1, 3, 3))
    cb = g.normal(size=2) * 0.3
    lw = g.normal(size=(32, 2)) * 0.5
    lb = g.normal(size=2) * 0.3
    conv = torch.nn.Conv2d(1, 2, 3, padding=1).double()
    lin = torch.nn.Linear(32, 2).double()
    with torch.no_grad():
        conv.weight.copy_(to_torch(cw))
        assert conv.bias is not None
        conv.bias.copy_(to_torch(cb))
        lin.weight.copy_(to_torch(lw.T))
        lin.bias.copy_(to_torch(lb))
    cnn = torch.nn.Sequential(conv, torch.nn.ReLU(), torch.nn.Flatten(), lin)
    images = g.normal(size=(2, 1, 4, 4))
    for relu in [True, False]:
        cam = LayerGradCam(cnn, cnn[1]).attribute(to_torch(images), target=0, relu_attributions=relu)
        out.append(
            case(
                f"gradcam/{'relu' if relu else 'signed'}",
                "gradCam",
                [cw, cb, lw, lb, images.reshape(2, 16)],
                {"maps": to_numpy(single(cam)).reshape(2, 16)},
                relu=relu,
            )
        )
    return out
