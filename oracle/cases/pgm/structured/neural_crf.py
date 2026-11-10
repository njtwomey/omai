"""The CRF layer of a neural tagger against pytorch-crf (`torchcrf.CRF`) with non-zero start and end transitions.

Each case is a batch of right-padded sequences with random emissions, transitions, start and end transitions and gold
tags (arbitrary at the masked positions, which torchcrf ignores), laid out batch-first or not as the case says. The
output is one flat vector of float64 values in this order: the log-likelihood of each sequence (reduction none), the
`sum`, `mean` and `token_mean` reductions, log Z of each sequence, the score of each decoded path, the torch autograd
gradient of the summed log-likelihood with respect to the emissions, the transitions, the start and the end
transitions, and the posterior marginals (the one-hot tags minus the emissions' gradient: score minus log-likelihood is
log Z, whose gradient is the marginals). The parameters hold the decoded paths of `CRF.decode`."""

from importlib.metadata import version
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from torchcrf import CRF

REFERENCE: dict[str, str] = {"pytorch-crf": version("pytorch-crf")}

type Floats = npt.NDArray[np.float64]

SHAPES: list[tuple[list[int], int, bool]] = [
    ([5, 3, 1], 4, True),
    ([6, 4], 3, False),
    ([1], 2, True),
    ([4, 4, 2, 3], 5, False),
    ([7, 7, 7], 3, True),
]


def path_score(e: Floats, trans: Floats, start: Floats, end: Floats, tags: list[int]) -> float:
    """Score of a path: emissions, transitions, start of the first tag and end of the last."""
    total = float(start[tags[0]] + end[tags[-1]])
    for t, k in enumerate(tags):
        total += float(e[t, k])
        if t > 0:
            total += float(trans[tags[t - 1], k])
    return total


def layout(x: npt.NDArray[np.generic], batch_first: bool) -> npt.NDArray[np.generic]:
    """A batch-first array as the layer is given it: unchanged, or time-major."""
    return x if batch_first else np.swapaxes(x, 0, 1)


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(71)
    for n, (lengths, k, batch_first) in enumerate(SHAPES):
        b = len(lengths)
        t_max = max(lengths)
        emissions = np.round(g.normal(size=(b, t_max, k)), 6)
        trans = np.round(g.normal(size=(k, k)), 6)
        start = np.round(g.normal(size=k), 6)
        end = np.round(g.normal(size=k), 6)
        tags = g.integers(0, k, size=(b, t_max))
        mask = np.array([[1 if t < length else 0 for t in range(t_max)] for length in lengths])
        model = CRF(k, batch_first=batch_first).double()
        with torch.no_grad():
            model.transitions.copy_(torch.tensor(trans))
            model.start_transitions.copy_(torch.tensor(start))
            model.end_transitions.copy_(torch.tensor(end))

        e_in = torch.tensor(layout(emissions, batch_first), requires_grad=True)
        y_in = torch.tensor(layout(tags, batch_first))
        m_in = cast(torch.ByteTensor, torch.tensor(layout(mask, batch_first), dtype=torch.uint8))
        reductions = {r: model(e_in, y_in, mask=m_in, reduction=r) for r in ["none", "sum", "mean", "token_mean"]}
        llh = reductions["none"]
        total = reductions["sum"]
        params_t = [e_in, model.transitions, model.start_transitions, model.end_transitions]
        # A length-1 batch never uses the transitions: their gradient is zero.
        raw = cast(tuple[torch.Tensor | None, ...], torch.autograd.grad(total, params_t, allow_unused=True))
        grads = [torch.zeros_like(p) if gr is None else gr for p, gr in zip(params_t, raw, strict=True)]
        grad_e = np.asarray(grads[0].detach().numpy())
        grad_e_batch_first = grad_e if batch_first else np.swapaxes(grad_e, 0, 1)
        onehot = np.eye(k)[tags] * mask[:, :, None]
        marginals = onehot - grad_e_batch_first
        paths = model.decode(e_in.detach(), mask=m_in)
        llh_np = np.asarray(llh.detach().numpy())
        gold = np.array(
            [path_score(emissions[i], trans, start, end, [int(v) for v in tags[i, : lengths[i]]]) for i in range(b)]
        )
        log_z = gold - llh_np
        path_scores = np.array([path_score(emissions[i], trans, start, end, paths[i]) for i in range(b)])
        pieces = [
            llh_np,
            np.array([float(reductions[r].item()) for r in ["sum", "mean", "token_mean"]]),
            log_z,
            path_scores,
            grad_e.ravel(),
            np.asarray(grads[1].detach().numpy()).ravel(),
            np.asarray(grads[2].detach().numpy()).ravel(),
            np.asarray(grads[3].detach().numpy()).ravel(),
            layout(marginals, batch_first).ravel(),
        ]
        out.append(
            case(
                f"layer/{n}",
                "layer",
                [
                    layout(emissions, batch_first),
                    trans,
                    start,
                    end,
                    layout(tags, batch_first),
                    layout(mask, batch_first),
                ],
                np.concatenate(pieces),
                rtol=1e-10,
                atol=1e-12,
                batch_first=batch_first,
                paths=paths,
            )
        )
    return out
