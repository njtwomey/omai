"""The relational graph convolution against a direct torch implementation (float64; torch_geometric is not in the
oracle group): out_i = x_i W_root + b + sum_r mean_{j in N_r(i)} x_j W_r, with per-relation weights or a basis
decomposition W_r = sum_b comp[r, b] V_b, and sum or mean aggregation per relation."""

import numpy as np
import torch
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def rgcn(
    x: torch.Tensor,
    edges: torch.Tensor,
    types: torch.Tensor,
    weight: torch.Tensor,
    comp: torch.Tensor | None,
    root: torch.Tensor,
    bias: torch.Tensor,
    aggregate: str,
) -> torch.Tensor:
    n = x.shape[0]
    relations = comp.shape[0] if comp is not None else weight.shape[0]
    w = torch.einsum("rb,bio->rio", comp, weight) if comp is not None else weight
    out = x @ root + bias
    for r in range(relations):
        mask = types == r
        src, dst = edges[0][mask], edges[1][mask]
        if src.numel() == 0:
            continue
        message = x[src] @ w[r]
        total = torch.zeros(n, w.shape[2], dtype=x.dtype).index_add(0, dst, message)
        if aggregate == "mean":
            counts = torch.zeros(n, dtype=x.dtype).index_add(0, dst, torch.ones(dst.numel(), dtype=x.dtype))
            total = total / counts.clamp(min=1).unsqueeze(1)
        out = out + total
    return out


def cases() -> list[Case]:
    g = rng(5302)

    def n(*shape: int) -> FloatArray:
        return np.asarray(g.normal(size=shape), dtype=np.float64)

    out: list[Case] = []
    edges = np.array([[0, 1, 2, 3, 4, 0, 2, 4, 1, 3], [1, 2, 0, 0, 3, 4, 4, 1, 1, 2]], dtype=np.int32)
    types = np.array([0, 1, 2, 0, 1, 2, 0, 1, 0, 2], dtype=np.int32)
    for name, bases, aggregate in [("plain/mean", 0, "mean"), ("bases/mean", 2, "mean"), ("plain/sum", 0, "sum")]:
        x = n(5, 3)
        weight = n(bases or 3, 3, 4)
        comp = n(3, bases) if bases else None
        root = n(3, 4)
        bias = n(4)
        y = rgcn(
            to_torch(x),
            torch.tensor(edges, dtype=torch.long),
            torch.tensor(types, dtype=torch.long),
            to_torch(weight),
            to_torch(comp) if comp is not None else None,
            to_torch(root),
            to_torch(bias),
            aggregate,
        )
        state: dict[str, FloatArray] = {"weight": weight, "root": root, "bias": bias}
        if comp is not None:
            state["comp"] = comp
        config: dict[str, object] = {"in": 3, "out": 4, "relations": 3, "bases": bases, "aggregate": aggregate}
        out.append(case(f"rgcn/{name}", "rgcn", [x, edges, types], to_numpy(y), state=state, config=config))
    return out
