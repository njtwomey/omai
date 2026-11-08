"""Small helpers for case modules: seeded arrays and the shape pairs the TypeScript property tests also use."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from .fixture import Case, case

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


def loguniform(g: np.random.Generator, n: int, lo: float, hi: float) -> FloatArray:
    """Values spread evenly in log space between `lo` and `hi` (both positive)."""
    return np.exp(g.uniform(np.log(lo), np.log(hi), size=n))


def reference(
    name: str,
    fn: str,
    f: Callable[..., FloatArray],
    inputs: list[FloatArray],
    *,
    rtol: float = 1e-10,
    atol: float = 1e-300,
    **params: int | str,
) -> Case:
    """A case whose expected output is a prebuilt numpy or scipy function applied to the inputs."""
    return case(name, fn, inputs, np.asarray(f(*inputs), dtype=np.float64), rtol=rtol, atol=atol, **params)


def spd(g: np.random.Generator, n: int, batch: tuple[int, ...] = ()) -> FloatArray:
    """A random symmetric positive-definite matrix (M M^T + n I), well conditioned."""
    m = g.normal(size=(*batch, n, n))
    return m @ np.swapaxes(m, -1, -2) + n * np.eye(n)


def well_conditioned(g: np.random.Generator, m: int, n: int, batch: tuple[int, ...] = ()) -> FloatArray:
    """A random matrix with a strong diagonal, so square ones are comfortably invertible."""
    return g.normal(size=(*batch, m, n)) + 3 * np.eye(m, n)


def sign_fixed(v: FloatArray) -> FloatArray:
    """Flip each column so that its largest-magnitude entry is positive (the convention omai uses)."""
    idx = np.argmax(np.abs(v), axis=-2, keepdims=True)
    signs = np.sign(np.take_along_axis(v, idx, axis=-2))
    return v * np.where(signs == 0, 1, signs)


def softplus(x: FloatArray) -> FloatArray:
    """log(1 + exp(x)), computed stably."""
    return np.logaddexp(0.0, x)
