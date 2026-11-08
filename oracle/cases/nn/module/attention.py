"""nn.MultiheadAttention against torch (float64): sequence-first and batch-first, self- and cross-attention, separate
key and value widths, no bias, add_bias_kv, add_zero_attn, key padding and attention masks (boolean and additive),
per-head weights; outputs and the returned weights."""

import numpy as np
import numpy.typing as npt
import torch
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.torch_helpers import to_numpy, to_torch
from torch import nn

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def state_of(m: nn.Module) -> dict[str, FloatArray]:
    return {k: to_numpy(v) for k, v in m.state_dict().items()}


def cases() -> list[Case]:
    g = rng(347)
    torch.manual_seed(347)

    def n(*shape: int) -> FloatArray:
        return np.asarray(g.normal(size=shape), dtype=np.float64)

    out: list[Case] = []

    def record(
        name: str,
        m: nn.MultiheadAttention,
        config: dict[str, object],
        inputs: list[FloatArray],
        masks: dict[str, npt.NDArray[np.generic]],
        average: bool = True,
    ) -> None:
        q = to_torch(inputs[0])
        k = to_torch(inputs[1]) if len(inputs) > 1 else q
        v = to_torch(inputs[2]) if len(inputs) > 2 else k
        kpm = masks.get("keyPaddingMask")
        am = masks.get("attnMask")
        y, w = m(
            q,
            k,
            v,
            key_padding_mask=None if kpm is None else torch.tensor(kpm),
            attn_mask=None
            if am is None
            else (torch.tensor(am) if am.dtype == np.bool_ else to_torch(am.astype(np.float64))),
            average_attn_weights=average,
        )
        assert w is not None
        state = state_of(m)
        all_inputs = [*inputs, *masks.values()]
        mask_names = list(masks.keys())
        for part, value in [("output", to_numpy(y)), ("weights", to_numpy(w))]:
            out.append(
                case(
                    f"{name}/{part}",
                    "MultiheadAttention",
                    all_inputs,
                    value,
                    rtol=1e-11,
                    atol=1e-12,
                    state=state,
                    config=config,
                    masks=mask_names,
                    inputCount=len(inputs),
                    average=average,
                    part=part,
                )
            )

    m = nn.MultiheadAttention(6, 2).double()
    record("self/seq-first", m, {"embedDim": 6, "numHeads": 2}, [n(4, 2, 6)], {})
    m = nn.MultiheadAttention(6, 3, batch_first=True).double()
    pad = np.array([[False, False, True, True, False], [False, False, False, False, True]])
    record(
        "cross/batch-first-padding",
        m,
        {"embedDim": 6, "numHeads": 3, "batchFirst": True},
        [n(2, 3, 6), n(2, 5, 6), n(2, 5, 6)],
        {"keyPaddingMask": pad},
    )
    m = nn.MultiheadAttention(4, 2, kdim=3, vdim=5, bias=False, batch_first=True).double()
    record(
        "kdim-vdim-nobias",
        m,
        {"embedDim": 4, "numHeads": 2, "kdim": 3, "vdim": 5, "bias": False, "batchFirst": True},
        [n(2, 3, 4), n(2, 4, 3), n(2, 4, 5)],
        {},
        average=False,
    )
    m = nn.MultiheadAttention(4, 2, add_bias_kv=True, add_zero_attn=True, batch_first=True).double()
    am = n(3, 3)
    record(
        "bias-kv-zero-attn-float-mask",
        m,
        {"embedDim": 4, "numHeads": 2, "addBiasKv": True, "addZeroAttn": True, "batchFirst": True},
        [n(2, 3, 4)],
        {"attnMask": am},
    )
    m = nn.MultiheadAttention(4, 1, batch_first=True).double()
    bm = np.triu(np.ones((3, 3), dtype=np.bool_), k=1)
    record("bool-causal-mask", m, {"embedDim": 4, "numHeads": 1, "batchFirst": True}, [n(2, 3, 4)], {"attnMask": bm})
    return out
