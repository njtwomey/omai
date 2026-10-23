"""Shape operations against numpy."""

import numpy as np
from whitebox_oracle.fixture import case
from whitebox_oracle.helpers import rng, uniform

REFERENCE = {"numpy": np.__version__}


def cases():
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
    ]
