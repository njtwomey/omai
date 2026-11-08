"""Recurrent modules against torch.nn (float64): RNN (tanh, ReLU), LSTM (with projection) and GRU; one or two
layers, uni- or bidirectional, sequence- or batch-first, given initial states, packed variable-length sequences;
and the RNNCell, LSTMCell and GRUCell. Outputs and final states."""

import numpy as np
import torch
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.torch_helpers import to_numpy, to_torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def state_of(m: nn.Module) -> dict[str, FloatArray]:
    return {k: to_numpy(v) for k, v in m.state_dict().items()}


def cases() -> list[Case]:
    g = rng(349)
    torch.manual_seed(349)

    def n(*shape: int) -> FloatArray:
        return np.asarray(g.normal(size=shape), dtype=np.float64)

    out: list[Case] = []

    def record(name: str, m: nn.RNNBase, config: dict[str, object], x: FloatArray, with_state: bool) -> None:
        layers = m.num_layers * (2 if m.bidirectional else 1)
        b = x.shape[0] if m.batch_first else x.shape[1]
        h_width = m.proj_size if m.proj_size > 0 else m.hidden_size
        h0, c0 = n(layers, b, h_width), n(layers, b, m.hidden_size)
        inputs: list[FloatArray] = [x]
        if with_state:
            inputs += [h0, c0]
            if isinstance(m, nn.LSTM):
                y, (h, c) = m(to_torch(x), (to_torch(h0), to_torch(c0)))
            else:
                y, h = m(to_torch(x), to_torch(h0))
                c = None
        else:
            if isinstance(m, nn.LSTM):
                y, (h, c) = m(to_torch(x))
            else:
                y, h = m(to_torch(x))
                c = None
        parts = [("output", to_numpy(y)), ("h", to_numpy(h))] + ([] if c is None else [("c", to_numpy(c))])
        for part, value in parts:
            out.append(case(f"{name}/{part}", "recurrent", inputs, value, state=state_of(m), config=config, part=part))

    m = nn.RNN(3, 4).double()
    record("RNN/tanh", m, {"kind": "RNN", "input": 3, "hidden": 4}, n(5, 2, 3), False)
    m = nn.RNN(3, 4, num_layers=2, nonlinearity="relu", bidirectional=True).double()
    cfg: dict[str, object] = {"kind": "RNN", "input": 3, "hidden": 4, "layers": 2, "relu": True, "bidirectional": True}
    record("RNN/relu-2-bi", m, cfg, n(4, 2, 3), True)
    m = nn.LSTM(3, 4, num_layers=2, batch_first=True).double()
    record(
        "LSTM/2-batch-first",
        m,
        {"kind": "LSTM", "input": 3, "hidden": 4, "layers": 2, "batchFirst": True},
        n(2, 5, 3),
        True,
    )
    m = nn.LSTM(3, 5, bidirectional=True, proj_size=2).double()
    cfg = {"kind": "LSTM", "input": 3, "hidden": 5, "bidirectional": True, "proj": 2}
    record("LSTM/bi-proj", m, cfg, n(4, 2, 3), True)
    m = nn.GRU(3, 4, num_layers=2, bidirectional=True, batch_first=True).double()
    cfg = {"kind": "GRU", "input": 3, "hidden": 4, "layers": 2, "bidirectional": True, "batchFirst": True}
    record("GRU/2-bi", m, cfg, n(2, 4, 3), False)

    # Packed variable-length sequences (unsorted lengths).
    lengths = [3, 5, 2]
    for kind in ["LSTM", "GRU"]:
        m = nn.LSTM(2, 3, bidirectional=True).double() if kind == "LSTM" else nn.GRU(2, 3, bidirectional=True).double()
        x = n(5, 3, 2)
        packed = pack_padded_sequence(to_torch(x), torch.tensor(lengths), enforce_sorted=False)
        result = m(packed)
        y_packed = result[0]
        h_n = result[1][0] if kind == "LSTM" else result[1]
        assert isinstance(h_n, torch.Tensor)
        y, _ = pad_packed_sequence(y_packed)
        cfg = {"kind": kind, "input": 2, "hidden": 3, "bidirectional": True, "lengths": lengths}
        out.append(
            case(f"{kind}/packed/output", "packed", [x], to_numpy(y), state=state_of(m), config=cfg, part="output")
        )
        out.append(case(f"{kind}/packed/h", "packed", [x], to_numpy(h_n), state=state_of(m), config=cfg, part="h"))

    x, h = n(3, 4), n(3, 5)
    for kind, cell in [("RNNCell", nn.RNNCell(4, 5)), ("GRUCell", nn.GRUCell(4, 5))]:
        cell = cell.double()
        y = cell(to_torch(x), to_torch(h))
        out.append(case(kind, "cell", [x, h], to_numpy(y), state=state_of(cell), config={"kind": kind}))
    cell = nn.LSTMCell(4, 5).double()
    c = n(3, 5)
    hy, cy = cell(to_torch(x), (to_torch(h), to_torch(c)))
    out.append(
        case("LSTMCell/h", "cell", [x, h, c], to_numpy(hy), state=state_of(cell), config={"kind": "LSTMCell"}, part="h")
    )
    out.append(
        case("LSTMCell/c", "cell", [x, h, c], to_numpy(cy), state=state_of(cell), config={"kind": "LSTMCell"}, part="c")
    )
    return out
