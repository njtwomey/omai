"""The running average against OpenCV's accumulateWeighted (float64 accumulators)."""

from importlib.metadata import version

import cv2
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}


def cases() -> list[Case]:
    g = rng(4470)
    out: list[Case] = []
    for name, shape in [("grey", (6, 7)), ("rgb", (5, 4, 3))]:
        x = g.uniform(size=shape)
        acc = g.uniform(size=shape)
        for alpha in [0.05, 0.5]:
            res = acc.copy()
            cv2.accumulateWeighted(x, res, alpha)
            out.append(
                case(
                    f"accumulateWeighted/{name}/{alpha}",
                    "accumulateWeighted",
                    [x, acc],
                    res,
                    rtol=1e-12,
                    atol=1e-12,
                    alpha=alpha,
                )
            )
    return out
