"""Small helpers for case modules: seeded arrays and the shape pairs the TypeScript property tests also use."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
"""A float64 array, the default dtype of case inputs."""

type Unary = Callable[[FloatArray], FloatArray]
"""An elementwise function of one array (a numpy or scipy ufunc, or a small def)."""

type Binary = Callable[[FloatArray, FloatArray], FloatArray]
"""An elementwise function of two broadcastable arrays."""

UNARY_SHAPES: list[tuple[int, ...]] = [(), (4,), (2, 3)]
BINARY_SHAPES: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((3,), (3,)),
    ((2, 3), (3,)),
    ((2, 1), (1, 3)),
    ((), (2, 2)),
]


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def uniform(g: np.random.Generator, shape: tuple[int, ...], lo: float = -2.0, hi: float = 2.0) -> FloatArray:
    return np.asarray(g.uniform(lo, hi, size=shape), dtype=np.float64)
