"""Helpers for the systems case modules: complex values as stacked parts, canonical orderings and random systems."""

import numpy as np
import numpy.typing as npt

from .helpers import FloatArray


def parts(z: npt.ArrayLike) -> FloatArray:
    """Complex values as a `[2, n]` array of real and imaginary parts, in the given order."""
    c = np.atleast_1d(np.asarray(z, dtype=np.complex128))
    return np.stack([c.real, c.imag]).astype(np.float64)


def sorted_parts(z: npt.ArrayLike) -> FloatArray:
    """Complex values as `[2, n]` parts sorted by real then imaginary part (the order whitebox-js returns roots in)."""
    c = np.atleast_1d(np.asarray(z, dtype=np.complex128))
    order = np.lexsort((c.imag, c.real))
    return parts(c[order])


def stable_system(
    g: np.random.Generator, n: int, m: int, p: int, shift: float = 2.0
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """A random continuous-time state-space system whose A has every eigenvalue left of -shift / 2 (roughly)."""
    a = g.normal(size=(n, n)) - shift * np.eye(n)
    while np.max(np.linalg.eigvals(a).real) > -0.2:
        a = a - np.eye(n)
    return a, g.normal(size=(n, m)), g.normal(size=(p, n)), g.normal(size=(p, m))


def schur_stable_system(
    g: np.random.Generator, n: int, m: int, p: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """A random discrete-time state-space system with spectral radius below 0.9."""
    a = g.normal(size=(n, n))
    a = 0.8 * a / np.max(np.abs(np.linalg.eigvals(a)))
    return a, g.normal(size=(n, m)), g.normal(size=(p, n)), g.normal(size=(p, m))
