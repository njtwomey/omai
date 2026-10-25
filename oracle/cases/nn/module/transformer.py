"""Transformer layers against torch.nn (float64, dropout 0 in training mode, which keeps torch off its fused fast
path): post- and pre-norm encoder layers with ReLU and GELU, masks and key padding, decoder layers with memory,
encoder and decoder stacks with a final norm, and the encoder-decoder nn.Transformer."""

import numpy as np
import torch
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.torch_helpers import to_numpy, to_torch
from torch import nn

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def state_of(m: nn.Module) -> dict[str, FloatArray]:
    return {k: to_numpy(v) for k, v in m.state_dict().items()}


def cases() -> list[Case]:
    g = rng(348)
    torch.manual_seed(348)

    def n(*shape: int) -> FloatArray:
        return np.asarray(g.normal(size=shape), dtype=np.float64)

    out: list[Case] = []
    causal = nn.Transformer.generate_square_subsequent_mask(4, dtype=torch.float64)
    pad = np.array([[False, False, False, True], [False, True, False, False]])
    for norm_first in [False, True]:
        for activation in ["relu", "gelu"]:
            layer = nn.TransformerEncoderLayer(
                8, 2, dim_feedforward=12, dropout=0.0, activation=activation, norm_first=norm_first, batch_first=True
            ).double()
            src = n(2, 4, 8)
            config: dict[str, object] = {
                "dModel": 8,
                "nhead": 2,
                "ff": 12,
                "activation": activation,
                "normFirst": norm_first,
            }
            name = f"encoderLayer/{'pre' if norm_first else 'post'}-{activation}"
            out.append(
                case(name, "encoderLayer", [src], to_numpy(layer(to_torch(src))), state=state_of(layer), config=config)
            )
            y = layer(to_torch(src), src_mask=causal, src_key_padding_mask=torch.tensor(pad))
            out.append(
                case(
                    f"{name}/masked",
                    "encoderLayerMasked",
                    [src, to_numpy(causal), pad],
                    to_numpy(y),
                    state=state_of(layer),
                    config=config,
                )
            )
    dec = nn.TransformerDecoderLayer(8, 2, dim_feedforward=12, dropout=0.0, batch_first=True).double()
    tgt, memory = n(2, 4, 8), n(2, 5, 8)
    y = dec(to_torch(tgt), to_torch(memory), tgt_mask=causal)
    dconfig: dict[str, object] = {"dModel": 8, "nhead": 2, "ff": 12, "activation": "relu", "normFirst": False}
    out.append(
        case(
            "decoderLayer/post",
            "decoderLayer",
            [tgt, memory, to_numpy(causal)],
            to_numpy(y),
            state=state_of(dec),
            config=dconfig,
        )
    )
    dec_pre = nn.TransformerDecoderLayer(
        8, 2, dim_feedforward=12, dropout=0.0, batch_first=True, norm_first=True
    ).double()
    y = dec_pre(to_torch(tgt), to_torch(memory), tgt_mask=causal)
    pconfig: dict[str, object] = {**dconfig, "normFirst": True}
    out.append(
        case(
            "decoderLayer/pre",
            "decoderLayer",
            [tgt, memory, to_numpy(causal)],
            to_numpy(y),
            state=state_of(dec_pre),
            config=pconfig,
        )
    )

    model = nn.Transformer(
        d_model=8,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=12,
        dropout=0.0,
        batch_first=True,
    ).double()
    src, tgt = n(2, 5, 8), n(2, 4, 8)
    y = model(to_torch(src), to_torch(tgt), tgt_mask=causal)
    tconfig: dict[str, object] = {"dModel": 8, "nhead": 2, "ff": 12, "layers": 2}
    out.append(
        case(
            "transformer",
            "transformer",
            [src, tgt, to_numpy(causal)],
            to_numpy(y),
            rtol=1e-10,
            atol=1e-11,
            state=state_of(model),
            config=tconfig,
        )
    )
    return out
