"""Shape operations against numpy."""

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng, uniform
from numpy.lib.stride_tricks import sliding_window_view

REFERENCE: dict[str, str] = {"numpy": np.__version__}


def dilated_windows(v: FloatArray, w: int, step: int, dilation: int) -> FloatArray:
    """Windows of v[i * step + j * dilation], j < w, by fancy indexing (numpy's view has no dilation)."""
    count = (len(v) - dilation * (w - 1) - 1) // step + 1
    starts = np.arange(count)[:, None] * step
    return v[starts + np.arange(w)[None, :] * dilation]


def cases() -> list[Case]:
    x = uniform(rng(500), (2, 3, 4))
    m = uniform(rng(501), (3, 4))
    v = uniform(rng(502), (5,))
    return [
        case("reshape", "reshape", [x], x.reshape(4, -1), shape=[4, -1]),
        case("transpose/default", "transpose", [x], np.transpose(x)),
        case("transpose/perm", "transpose", [x], np.transpose(x, (1, 2, 0)), perm=[1, 2, 0]),
        case("swapaxes", "swapaxes", [x], np.swapaxes(x, 0, 2), a=0, b=2),
        case("moveaxis", "moveaxis", [x], np.moveaxis(x, 0, -1), source=0, destination=-1),
        case("expandDims", "expandDims", [m], np.expand_dims(m, 1), axis=1),
        case("squeeze", "squeeze", [m.reshape(3, 1, 4)], np.squeeze(m.reshape(3, 1, 4))),
        case("broadcastTo", "broadcastTo", [v], np.broadcast_to(v, (2, 5)), shape=[2, 5]),
        case("concat/axis1", "concat", [m, m[:, :2]], np.concatenate([m, m[:, :2]], axis=1), axis=1),
        case("stack/axis1", "stack", [v, v * 2], np.stack([v, v * 2], axis=1), axis=1),
        case("flip/all", "flip", [m], np.flip(m)),
        case("flip/axis1", "flip", [m], np.flip(m, axis=1), axis=1),
        case("roll/flat", "roll", [m], np.roll(m, 5), shift=5),
        case("roll/axis0", "roll", [m], np.roll(m, -1, axis=0), shift=-1, axis=0),
        case("tile", "tile", [v[:2]], np.tile(v[:2], (2, 3)), reps=[2, 3]),
        case("repeat/flat", "repeat", [m], np.repeat(m, 2), repeats=2),
        case("repeat/axis1", "repeat", [m], np.repeat(m, [1, 0, 2, 1], axis=1), repeats=[1, 0, 2, 1], axis=1),
        case("pad", "pad", [m], np.pad(m, ((1, 2), (0, 1)), constant_values=7.0), widths=[[1, 2], [0, 1]], value=7.0),
        case("slidingWindowView/last", "slidingWindowView", [m], sliding_window_view(m, 3, axis=-1), window=3),
        case("slidingWindowView/axis0", "slidingWindowView", [x], sliding_window_view(x, 2, axis=0), window=2, axis=0),
        case(
            "slidingWindowView/two-axes",
            "slidingWindowView",
            [x],
            sliding_window_view(x, (2, 3), axis=(1, 2)),
            window=[2, 3],
            axis=[1, 2],
        ),
        case(
            "slidingWindowView/step",
            "slidingWindowView",
            [m],
            sliding_window_view(m, 2, axis=1)[:, ::2],
            window=2,
            axis=1,
            step=2,
        ),
        case(
            "slidingWindowView/dilation",
            "slidingWindowView",
            [uniform(rng(503), (11,))],
            dilated_windows(uniform(rng(503), (11,)), 3, 2, 3),
            window=3,
            step=2,
            dilation=3,
        ),
        case("indices", "indices", [], np.indices((2, 3, 4)).astype(np.int32), shape=[2, 3, 4]),
    ]
