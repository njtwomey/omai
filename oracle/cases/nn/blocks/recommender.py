"""Recommender blocks against direct torch implementations (float64): the factorisation machine and the field-aware
machine as the explicit sum over pairs of fields, neural collaborative filtering from torch modules, the DLRM dot
interaction by batched matrix products, and the DCN cross network layer by layer."""

import numpy as np
import torch
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.torch_helpers import to_numpy, to_torch
from torch import nn

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def fm_pairs(
    bias: torch.Tensor, linear: torch.Tensor, factors: torch.Tensor, ids: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """w0 + sum_i w_i x_i + sum_{i<j} <v_i, v_j> x_i x_j, over the active features of each row."""
    batch, fields = ids.shape
    out = torch.zeros(batch, dtype=torch.float64) + bias
    for b in range(batch):
        for p in range(fields):
            out[b] = out[b] + linear[ids[b, p], 0] * values[b, p]
            for q in range(p + 1, fields):
                out[b] = out[b] + torch.dot(factors[ids[b, p]], factors[ids[b, q]]) * values[b, p] * values[b, q]
    return out


def ffm_pairs(
    bias: torch.Tensor, linear: torch.Tensor, factors: torch.Tensor, ids: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """The field-aware sum: feature of field p uses its vector for field q, and vice versa."""
    batch, fields = ids.shape
    out = torch.zeros(batch, dtype=torch.float64) + bias
    for b in range(batch):
        for p in range(fields):
            out[b] = out[b] + linear[ids[b, p], 0] * values[b, p]
            for q in range(p + 1, fields):
                v = torch.dot(factors[ids[b, p], q], factors[ids[b, q], p])
                out[b] = out[b] + v * values[b, p] * values[b, q]
    return out


def cases() -> list[Case]:
    g = rng(5303)
    torch.manual_seed(5303)

    def n(*shape: int) -> FloatArray:
        return np.asarray(g.normal(size=shape), dtype=np.float64)

    out: list[Case] = []
    features, fields, k, batch = 12, 4, 3, 5
    ids = np.stack([g.integers(3 * f, 3 * f + 3, size=batch) for f in range(fields)], axis=1).astype(np.int32)
    values = np.asarray(g.uniform(0.5, 1.5, size=(batch, fields)), dtype=np.float64)
    bias = np.asarray(g.normal(), dtype=np.float64)
    linear = n(features, 1)
    factors = n(features, k)
    ffm_factors = n(features, fields, k)
    for name, with_values in [("fm/one-hot", False), ("fm/valued", True)]:
        vals = values if with_values else np.ones_like(values)
        y = fm_pairs(to_torch(bias), to_torch(linear), to_torch(factors), torch.tensor(ids).long(), to_torch(vals))
        state = {"bias": bias, "linear": linear, "factors": factors}
        config: dict[str, object] = {"features": features, "factors": k}
        inputs = [ids, vals] if with_values else [ids]
        out.append(case(name, "fm", inputs, to_numpy(y), state=state, config=config))
    y = ffm_pairs(to_torch(bias), to_torch(linear), to_torch(ffm_factors), torch.tensor(ids).long(), to_torch(values))
    state = {"bias": bias, "linear": linear, "factors": ffm_factors}
    config = {"features": features, "fields": fields, "factors": k}
    out.append(case("ffm/valued", "ffm", [ids, values], to_numpy(y), state=state, config=config))

    # Neural collaborative filtering (NeuMF), from torch modules with the same layout.
    users, items, gd, md = 6, 9, 4, 3
    user_ids = g.integers(0, users, size=7).astype(np.int32)
    item_ids = g.integers(0, items, size=7).astype(np.int32)
    ncf = {
        "userGmf": nn.Embedding(users, gd).double(),
        "itemGmf": nn.Embedding(items, gd).double(),
        "userMlp": nn.Embedding(users, md).double(),
        "itemMlp": nn.Embedding(items, md).double(),
        "tower": nn.Sequential(nn.Linear(2 * md, 5), nn.ReLU(), nn.Linear(5, 3)).double(),
        "output": nn.Linear(gd + 3, 1).double(),
    }
    u, i = torch.tensor(user_ids).long(), torch.tensor(item_ids).long()
    gmf = ncf["userGmf"](u) * ncf["itemGmf"](i)
    deep = torch.relu(ncf["tower"](torch.cat([ncf["userMlp"](u), ncf["itemMlp"](i)], dim=-1)))
    logit = ncf["output"](torch.cat([gmf, deep], dim=-1)).reshape(-1)
    ncf_state: dict[str, FloatArray] = {}
    for part, module in ncf.items():
        for key, value in module.state_dict().items():
            ncf_state[f"{part}.{key}"] = to_numpy(value.double())
    ncf_config: dict[str, object] = {"users": users, "items": items, "gmf": gd, "mlp": md, "hidden": [5, 3]}
    out.append(case("ncf/neumf", "ncf", [user_ids, item_ids], to_numpy(logit), state=ncf_state, config=ncf_config))

    # DLRM dot interaction: the strictly lower triangle of the Gram matrix of n vectors.
    vectors = n(4, 5, 3)
    t = to_torch(vectors)
    gram = torch.bmm(t, t.transpose(1, 2))
    rows, cols = torch.tril_indices(5, 5, offset=-1)
    out.append(case("dlrm/dot-interaction", "dotInteraction", [vectors], to_numpy(gram[:, rows, cols])))

    # The DCN cross network: x_{l+1} = x_0 (w_l . x_l) + b_l + x_l.
    x0 = n(6, 5)
    weights = [n(5) for _ in range(3)]
    biases = [n(5) for _ in range(3)]
    h = to_torch(x0)
    for w, b in zip(weights, biases, strict=True):
        h = to_torch(x0) * (h @ to_torch(w)).unsqueeze(-1) + to_torch(b) + h
    cross_state = {
        f"{layer}.{kind}": value
        for layer in range(3)
        for kind, value in (("weight", weights[layer]), ("bias", biases[layer]))
    }
    out.append(case("dcn/cross-3", "cross", [x0], to_numpy(h), state=cross_state, config={"dim": 5, "layers": 3}))
    return out
