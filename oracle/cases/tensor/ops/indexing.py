"""Indexing operations against numpy."""

import numpy as np
import numpy.typing as npt
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng, uniform

REFERENCE: dict[str, str] = {"numpy": np.__version__}


def scattered(x: FloatArray, coords: npt.NDArray[np.int32], updates: FloatArray, mode: str) -> FloatArray:
    """numpy's in-place scatters on a copy: assignment for set, the ufunc ``at`` methods otherwise."""
    out = x.copy()
    where = tuple(coords.T)
    if mode == "set":
        out[where] = updates
    else:
        ufunc = {"add": np.add, "mul": np.multiply, "max": np.maximum, "min": np.minimum}[mode]
        ufunc.at(out, where, updates)
    return out


def cases() -> list[Case]:
    m = uniform(rng(600), (4, 5))
    v = uniform(rng(601), (8,))
    idx = np.array([3, 0, 0, -1], dtype=np.int32)
    along = np.array([[4, 0], [1, 1], [2, 3], [0, 0]], dtype=np.int32)
    ties = np.array([3.0, 1.0, 2.0, 1.0, np.nan, 2.0])
    cube = uniform(rng(602), (3, 4, 2))
    pairs = np.array([[2, 1], [0, -1], [2, 1], [1, 0]], dtype=np.int32)
    rows = np.array([[1], [0], [1]], dtype=np.int32)
    distinct = np.array([[2, 1], [0, 3], [1, 0]], dtype=np.int32)
    row_updates = uniform(rng(603), (3, 4, 2))
    pair_updates = uniform(rng(604), (4, 2))
    nan_updates = pair_updates.copy()
    nan_updates[1, 0] = np.nan
    scatters = [
        case(
            f"scatter/{mode}",
            "scatter",
            [cube, pairs, pair_updates],
            scattered(cube, pairs, pair_updates, mode),
            mode=mode,
        )
        for mode in ["add", "mul", "max", "min"]
    ]
    return [
        case("gather/pairs", "gather", [cube, pairs], cube[tuple(pairs.T)]),
        case("gather/rows", "gather", [cube, rows], cube[tuple(rows.T)]),
        case(
            "gather/full",
            "gather",
            [cube, np.array([[2, 3, 1], [0, 0, 0]], dtype=np.int32)],
            cube[(np.array([2, 0]), np.array([3, 0]), np.array([1, 0]))],
        ),
        case(
            "scatter/set",
            "scatter",
            [cube, distinct, pair_updates[:3]],
            scattered(cube, distinct, pair_updates[:3], "set"),
            mode="set",
        ),
        case(
            "scatter/rows-add",
            "scatter",
            [cube, rows, row_updates],
            scattered(cube, rows, row_updates, "add"),
            mode="add",
        ),
        case(
            "scatter/max-nan",
            "scatter",
            [cube, pairs, nan_updates],
            scattered(cube, pairs, nan_updates, "max"),
            mode="max",
        ),
        *scatters,
        case("slice/basic", "slice", [m], m[1:3, ::2], starts=[1, None], stops=[3, None], steps=[1, 2]),
        case("slice/negative", "slice", [m], m[::-1, -2::-2], starts=[None, -2], stops=[None, None], steps=[-1, -2]),
        case("slice/clamped", "slice", [v], v[-100:100], starts=[-100], stops=[100], steps=[1]),
        case("take/axis0", "take", [m, idx], np.take(m, idx, axis=0), axis=0),
        case(
            "take/matrix-indices",
            "take",
            [m, np.array([[4, -1], [0, 2], [1, 1]], dtype=np.int32)],
            np.take(m, [[4, -1], [0, 2], [1, 1]], axis=1),
            axis=1,
        ),
        case("take/axis1", "take", [m, np.array([4, 2], dtype=np.int32)], np.take(m, [4, 2], axis=1), axis=1),
        case("takeAlongAxis", "takeAlongAxis", [m, along], np.take_along_axis(m, along, axis=1), axis=1),
        case("sort", "sort", [ties], np.sort(ties)),
        case("argsort/stable", "argsort", [ties], np.argsort(ties, kind="stable").astype(np.int32)),
        case("sort/axis0", "sort", [m], np.sort(m, axis=0), axis=0),
        case("oneHot", "oneHot", [np.array([0, 2, 1], dtype=np.int32)], np.eye(3)[[0, 2, 1]], n=3),
        case(
            "searchsorted/left",
            "searchsorted",
            [np.sort(v), v[:3]],
            np.searchsorted(np.sort(v), v[:3]).astype(np.int32),
        ),
        case(
            "searchsorted/right",
            "searchsorted",
            [np.array([1.0, 2.0, 2.0, 3.0]), np.array([2.0])],
            np.searchsorted([1.0, 2.0, 2.0, 3.0], [2.0], side="right").astype(np.int32),
            side="right",
        ),
        case(
            "bincount/weights",
            "bincount",
            [np.array([0, 1, 1, 3], dtype=np.int32), np.array([0.5, 1.0, 2.0, 4.0])],
            np.bincount([0, 1, 1, 3], weights=[0.5, 1.0, 2.0, 4.0]),
        ),
    ]
