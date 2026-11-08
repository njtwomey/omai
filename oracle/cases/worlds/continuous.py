"""Distances from walls and sight in continuous worlds: scipy.ndimage's exact Euclidean distance transform and its
taxicab and chessboard chamfer transforms on random occupancy grids, and shapely's distances from points to the walls
(blocked cells and the map edge) and line-of-sight tests between cell centres (blocked when the segment touches a
blocked cell)."""

import numpy as np
import scipy
import shapely
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from scipy import ndimage as ndi
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "shapely": shapely.__version__}


def random_map(g: np.random.Generator, rows: int, cols: int, wall: float) -> list[str]:
    """A map of floor `.` and walls `#`."""
    return ["".join("#" if x < wall else "." for x in row) for row in g.uniform(size=(rows, cols))]


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(424242)
    for i in range(8):
        rows = random_map(g, int(g.integers(3, 9)), int(g.integers(3, 10)), 0.25)
        free = np.array([[ch == "." for ch in row] for row in rows])
        out.append(
            case(
                f"transforms/{i}",
                "transforms",
                [],
                np.array(0.0),
                rows=rows,
                euclidean=[float(x) for x in ndi.distance_transform_edt(free).ravel()],
                taxicab=[float(x) for x in ndi.distance_transform_cdt(free, metric="taxicab").ravel()],
                chessboard=[float(x) for x in ndi.distance_transform_cdt(free, metric="chessboard").ravel()],
                rtol=1e-12,
                atol=1e-12,
            )
        )
    for i in range(8):
        nr, nc = int(g.integers(3, 8)), int(g.integers(3, 9))
        rows = random_map(g, nr, nc, 0.3)
        blocked = unary_union([box(c, r, c + 1, r + 1) for r in range(nr) for c in range(nc) if rows[r][c] == "#"])
        edge = box(0, 0, nc, nr).exterior
        walls = unary_union([blocked, edge])
        points: list[list[float]] = []
        while len(points) < 12:
            x, y = float(g.uniform(0, nc)), float(g.uniform(0, nr))
            if rows[int(y)][int(x)] == ".":
                points.append([x, y])
        centres = [(c + 0.5, r + 0.5) for r in range(nr) for c in range(nc) if rows[r][c] == "."]
        sight = [
            [int(ia), int(ib), bool(not LineString([a, b]).intersects(blocked))]
            for ia, a in enumerate(centres)
            for ib, b in enumerate(centres)
            if ia < ib
        ]
        out.append(
            case(
                f"geometry/{i}",
                "geometry",
                [],
                np.array(0.0),
                rows=rows,
                points=points,
                distances=[float(Point(p).distance(walls)) for p in points],
                centres=[list(c) for c in centres],
                sight=sight,
            )
        )
    return out
