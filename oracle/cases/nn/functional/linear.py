"""Linear maps and embeddings against torch.nn.functional (float64): bilinear, embedding (padding index, maximum
norm) and embedding bags (sum, mean and max; offsets, 2-D input, per-sample weights, padding, last offset)."""

import numpy as np
import torch
import torch.nn.functional as F
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng
from whitebox_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def ints(values: list[int] | list[list[int]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64)


def cases() -> list[Case]:
    g = rng(344)

    def n(*shape: int) -> FloatArray:
        return np.asarray(g.normal(size=shape), dtype=np.float64)

    x1, x2, w, b = n(2, 3, 4), n(2, 3, 5), n(6, 4, 5), n(6)
    out: list[Case] = [
        case(
            "bilinear",
            "bilinear",
            [x1, x2, w, b],
            to_numpy(F.bilinear(to_torch(x1), to_torch(x2), to_torch(w), to_torch(b))),
        ),
    ]
    table = n(7, 3)
    idx = [[1, 4, 0], [6, 1, 2]]
    out.append(
        case(
            "embedding/plain",
            "embedding",
            [np.array(idx, dtype=np.int32), table],
            to_numpy(F.embedding(ints(idx), to_torch(table))),
        )
    )
    big = table * 3
    renormed = F.embedding(ints(idx), to_torch(big).clone(), max_norm=2.0)
    out.append(
        case("embedding/maxNorm", "embedding", [np.array(idx, dtype=np.int32), big], to_numpy(renormed), maxNorm=2.0)
    )
    flat = [1, 2, 4, 5, 4, 3, 2, 6]
    offsets = [0, 2, 2, 5]
    for mode in ["sum", "mean", "max"]:
        y = F.embedding_bag(ints(flat), to_torch(table), ints(offsets), mode=mode)
        out.append(
            case(
                f"embeddingBag/{mode}",
                "embeddingBag",
                [np.array(flat, dtype=np.int32), table],
                to_numpy(y),
                offsets=offsets,
                mode=mode,
            )
        )
        y2 = F.embedding_bag(ints(idx), to_torch(table), mode=mode)
        out.append(
            case(
                f"embeddingBag/{mode}/2d",
                "embeddingBag",
                [np.array(idx, dtype=np.int32), table],
                to_numpy(y2),
                mode=mode,
            )
        )
    psw = n(len(flat))
    y = F.embedding_bag(ints(flat), to_torch(table), ints(offsets), mode="sum", per_sample_weights=to_torch(psw))
    out.append(
        case(
            "embeddingBag/perSampleWeights",
            "embeddingBag",
            [np.array(flat, dtype=np.int32), table, psw],
            to_numpy(y),
            offsets=offsets,
            mode="sum",
        )
    )
    y = F.embedding_bag(ints(flat), to_torch(table), ints(offsets), mode="mean", padding_idx=4)
    out.append(
        case(
            "embeddingBag/padding",
            "embeddingBag",
            [np.array(flat, dtype=np.int32), table],
            to_numpy(y),
            offsets=offsets,
            mode="mean",
            paddingIdx=4,
        )
    )
    last = [0, 3, 8]
    y = F.embedding_bag(ints(flat), to_torch(table), ints(last), mode="sum", include_last_offset=True)
    out.append(
        case(
            "embeddingBag/includeLastOffset",
            "embeddingBag",
            [np.array(flat, dtype=np.int32), table],
            to_numpy(y),
            offsets=last,
            mode="sum",
            includeLastOffset=True,
        )
    )
    return out
