"""Perturbation attributions against captum on a small torch network with fixed weights: feature ablation (each
feature, feature masks, scalar and per-row baselines) and occlusion over a series and over an image."""

import captum
import numpy as np
import torch
from captum.attr import FeatureAblation, Occlusion
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng
from whitebox_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"captum": captum.__version__, "torch": torch.__version__}


def network(w1: FloatArray, b1: FloatArray, w2: FloatArray, b2: FloatArray) -> torch.nn.Sequential:
    """Flatten -> Linear(d, h) -> tanh -> Linear(h, k) in float64 with the given weights (rows of w are inputs)."""
    first = torch.nn.Linear(w1.shape[0], w1.shape[1]).double()
    second = torch.nn.Linear(w2.shape[0], w2.shape[1]).double()
    with torch.no_grad():
        first.weight.copy_(to_torch(w1.T))
        first.bias.copy_(to_torch(b1))
        second.weight.copy_(to_torch(w2.T))
        second.bias.copy_(to_torch(b2))
    return torch.nn.Sequential(torch.nn.Flatten(), first, torch.nn.Tanh(), second)


def cases() -> list[Case]:
    g = rng(4346)
    d, h, k = 6, 5, 2
    w1, b1 = g.normal(size=(d, h)), g.normal(size=h) * 0.5
    w2, b2 = g.normal(size=(h, k)), g.normal(size=k) * 0.5
    x = g.normal(size=(3, d))
    base = g.normal(size=(3, d)) * 0.3
    net = network(w1, b1, w2, b2)
    xt = to_torch(x)
    weights = [w1, b1, w2, b2, x, base]
    out: list[Case] = []
    fa = FeatureAblation(net)
    mask = np.array([0, 0, 1, 2, 2, 2])
    for label, kwargs in [
        ("plain", {}),
        ("scalar", {"baselines": 0.5}),
        ("rows", {"baselines": to_torch(base)}),
        ("mask", {"feature_mask": torch.tensor(mask).unsqueeze(0)}),
    ]:
        attr = fa.attribute(xt, target=1, **kwargs)  # type: ignore[arg-type]
        out.append(
            case(f"ablation/{label}", "ablation", weights, {"attributions": to_numpy(attr)}, label=label, mask=mask)
        )
    oc = Occlusion(net)
    for window, stride in [(2, 1), (3, 2)]:
        attr = oc.attribute(xt, sliding_window_shapes=(window,), strides=(stride,), target=0, baselines=0.0)
        out.append(
            case(
                f"occlusion/series/{window}/{stride}",
                "occlusion",
                weights,
                {"attributions": to_numpy(attr)},
                shape=[6],
                window=[window],
                stride=[stride],
                # captum averages over windows with a float32 count tensor.
                rtol=1e-6,
                atol=1e-7,
            )
        )
    img = to_torch(x.reshape(3, 2, 3))
    attr = oc.attribute(img, sliding_window_shapes=(2, 2), strides=(1, 1), target=1, baselines=0.0)
    out.append(
        case(
            "occlusion/image",
            "occlusion",
            weights,
            {"attributions": to_numpy(attr).reshape(3, 6)},
            shape=[2, 3],
            window=[2, 2],
            stride=[1, 1],
            target=1,
            rtol=1e-6,
            atol=1e-7,
        )
    )
    return out
