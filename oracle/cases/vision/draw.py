"""Drawing against OpenCV: one-pixel lines (8- and 4-connected, every octant, end points inside the image), polylines
and rectangles outlined and filled, on uint8 canvases (compared as 0/1 masks)."""

from importlib.metadata import version

import cv2
import numpy as np
from mdml_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}

EXACT = {"rtol": 1e-12, "atol": 1e-12}

SEGMENTS = [
    ((2, 3), (20, 9)),
    ((20, 9), (2, 3)),
    ((5, 18), (9, 1)),
    ((12, 2), (12, 17)),
    ((1, 10), (22, 10)),
    ((3, 3), (16, 16)),
    ((22, 1), (4, 15)),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for k, (p, q) in enumerate(SEGMENTS):
        for conn, code in [(8, cv2.LINE_8), (4, cv2.LINE_4)]:
            img = np.zeros((20, 24), np.uint8)
            cv2.line(img, p, q, 1, 1, code)
            out.append(
                case(
                    f"line/{k}/{conn}",
                    "line",
                    [],
                    img.astype(np.float64),
                    **EXACT,
                    p1=list(p),
                    p2=list(q),
                    lineType=conn,
                )
            )
    poly = np.array([[2, 2], [18, 4], [14, 15], [5, 12]], np.int32)
    img = np.zeros((20, 24), np.uint8)
    cv2.polylines(img, [poly], True, 1, 1, cv2.LINE_8)
    out.append(case("polylines/closed", "polylines", [], img.astype(np.float64), **EXACT, points=poly.tolist()))
    for name, thickness in [("outline", 1), ("filled", -1)]:
        img = np.zeros((20, 24), np.uint8)
        cv2.rectangle(img, (3, 4), (17, 13), 1, thickness)
        out.append(
            case(
                f"rectangle/{name}",
                "rectangle",
                [],
                img.astype(np.float64),
                **EXACT,
                p1=[3, 4],
                p2=[17, 13],
                thickness=thickness,
            )
        )
    return out
