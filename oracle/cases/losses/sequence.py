"""Connectionist temporal classification against torch.nn.functional.ctc_loss (float64)."""

import numpy as np
import torch
import torch.nn.functional as F
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.torch_helpers import to_numpy, to_torch

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def ctc(
    lp: FloatArray,
    targets: FloatArray,
    input_lengths: list[int],
    target_lengths: list[int],
    blank: int,
    reduction: str,
    zero_infinity: bool = False,
) -> FloatArray:
    return to_numpy(
        F.ctc_loss(
            to_torch(lp),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(input_lengths),
            torch.tensor(target_lengths),
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )
    )


def cases() -> list[Case]:
    g = rng(53)
    t, n, c = 12, 3, 5
    lp = np.asarray(torch.log_softmax(to_torch(g.normal(size=(t, n, c)) * 1.5), dim=-1).numpy(), dtype=np.float64)
    targets = np.array([[1.0, 2, 2, 3], [4.0, 1, 0, 0], [3.0, 3, 2, 1]])
    in_len, tg_len = [12, 10, 12], [4, 2, 4]
    out: list[Case] = []
    for reduction in ["mean", "sum", "none"]:
        out.append(
            case(
                f"batch/{reduction}",
                "ctc",
                [lp, targets],
                ctc(lp, targets, in_len, tg_len, 0, reduction),
                reduction=reduction,
                inputLengths=in_len,
                targetLengths=tg_len,
            )
        )
    # A blank other than zero: the labels shift down so that the last class is the blank.
    shifted = np.where(targets > 0, targets - 1, 0.0)
    out.append(
        case(
            "blank/last",
            "ctc",
            [lp, shifted],
            ctc(lp, shifted, in_len, tg_len, 4, "mean"),
            blank=4,
            inputLengths=in_len,
            targetLengths=tg_len,
        )
    )
    # Too few frames for a repeated-label target: infinite loss, zeroed by zero_infinity.
    short = np.ascontiguousarray(lp[:3])
    rep = np.array([[2.0, 2, 2], [1.0, 2, 0], [3.0, 1, 0]])
    out.append(
        case(
            "infeasible/zero-infinity",
            "ctc",
            [short, rep],
            ctc(short, rep, [3, 3, 3], [3, 2, 2], 0, "none", True),
            reduction="none",
            zeroInfinity=True,
            inputLengths=[3, 3, 3],
            targetLengths=[3, 2, 2],
        )
    )
    out.append(
        case(
            "infeasible/infinite",
            "ctc",
            [short, rep],
            ctc(short, rep, [3, 3, 3], [3, 2, 2], 0, "none"),
            reduction="none",
            inputLengths=[3, 3, 3],
            targetLengths=[3, 2, 2],
        )
    )
    single = np.ascontiguousarray(lp[:, 0, :])
    out.append(
        case(
            "single",
            "ctc",
            [single, targets[0]],
            ctc(single[:, None, :], targets[:1], [12], [4], 0, "sum"),
            reduction="sum",
        )
    )
    return out
