"""Small helpers for case modules: seeded arrays and the shape pairs the TypeScript property tests also use."""

import numpy as np

UNARY_SHAPES: list[tuple[int, ...]] = [(), (4,), (2, 3)]
BINARY_SHAPES: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((3,), (3,)),
    ((2, 3), (3,)),
    ((2, 1), (1, 3)),
    ((), (2, 2)),
]


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def uniform(g: np.random.Generator, shape: tuple[int, ...], lo: float = -2.0, hi: float = 2.0) -> np.ndarray:
    return np.asarray(g.uniform(lo, hi, size=shape), dtype=np.float64)
