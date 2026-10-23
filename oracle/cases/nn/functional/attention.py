"""Attention against torch.nn.functional.scaled_dot_product_attention (float64): plain, scaled, boolean and additive
masks, causal (square and rectangular), dropout off, and grouped-query heads."""

import numpy as np
import torch
import torch.nn.functional as F
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng
from whitebox_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def cases() -> list[Case]:
    g = rng(346)

    def n(*shape: int) -> FloatArray:
        return np.asarray(g.normal(size=shape), dtype=np.float64)

    q, k, v = n(2, 3, 4, 5), n(2, 3, 6, 5), n(2, 3, 6, 2)
    qt, kt, vt = to_torch(q), to_torch(k), to_torch(v)
    bool_mask = g.uniform(size=(4, 6)) > 0.3
    bool_mask[:, 0] = True
    float_mask = n(4, 6)
    out: list[Case] = [
        case("sdpa/plain", "sdpa", [q, k, v], to_numpy(F.scaled_dot_product_attention(qt, kt, vt))),
        case(
            "sdpa/scale", "sdpa", [q, k, v], to_numpy(F.scaled_dot_product_attention(qt, kt, vt, scale=0.3)), scale=0.3
        ),
        case(
            "sdpa/bool-mask",
            "sdpa",
            [q, k, v, bool_mask],
            to_numpy(F.scaled_dot_product_attention(qt, kt, vt, attn_mask=torch.tensor(bool_mask))),
        ),
        case(
            "sdpa/float-mask",
            "sdpa",
            [q, k, v, float_mask],
            to_numpy(F.scaled_dot_product_attention(qt, kt, vt, attn_mask=to_torch(float_mask))),
        ),
        case(
            "sdpa/causal-rectangular",
            "sdpa",
            [q, k, v],
            to_numpy(F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)),
            causal=True,
        ),
    ]
    qs = n(1, 2, 5, 4)
    out.append(
        case(
            "sdpa/causal-square",
            "sdpa",
            [qs, qs, qs],
            to_numpy(F.scaled_dot_product_attention(to_torch(qs), to_torch(qs), to_torch(qs), is_causal=True)),
            causal=True,
        )
    )
    qg, kg, vg = n(1, 4, 3, 2), n(1, 2, 5, 2), n(1, 2, 5, 3)
    y = F.scaled_dot_product_attention(to_torch(qg), to_torch(kg), to_torch(vg), enable_gqa=True)
    out.append(case("sdpa/gqa", "sdpa", [qg, kg, vg], to_numpy(y), enableGqa=True))
    return out
