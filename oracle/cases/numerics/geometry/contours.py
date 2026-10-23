"""Marching squares against skimage.measure.find_contours."""

from importlib.metadata import version
from typing import Literal

import numpy as np
from skimage.measure import find_contours
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"skimage": version("scikit-image")}


def field() -> FloatArray:
    y, x = np.mgrid[-2:2:23j, -3:3:31j]
    return (
        np.exp(-((x - 1) ** 2) - y**2) + 0.8 * np.exp(-((x + 1) ** 2) - 2 * y**2) - 0.3 * np.sin(2 * x) * np.cos(3 * y)
    )


def cases() -> list[Case]:
    z = field()
    out: list[Case] = []
    connectivity: list[Literal["low", "high"]] = ["low", "high"]
    for level in (0.0, 0.3, 0.6):
        for conn in connectivity:
            paths = find_contours(z, level, fully_connected=conn)
            out.append(
                case(
                    f"count/{level}/{conn}",
                    "count",
                    [z],
                    np.array(float(len(paths))),
                    rtol=0,
                    level=level,
                    connected=conn,
                )
            )
            for i, p in enumerate(paths):
                out.append(
                    case(
                        f"path/{level}/{conn}/{i}",
                        "path",
                        [z],
                        np.asarray(p, dtype=np.float64),
                        rtol=1e-12,
                        atol=1e-12,
                        level=level,
                        connected=conn,
                        index=i,
                    )
                )
    return out
