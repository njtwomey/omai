"""Helpers for the systems case modules: complex values as stacked parts, canonical orderings and random systems."""

from collections.abc import Callable
from typing import cast

import control as ct
import numpy as np
import numpy.typing as npt

from .helpers import FloatArray


def parts(z: npt.ArrayLike) -> FloatArray:
    """Complex values as a `[2, n]` array of real and imaginary parts, in the given order."""
    c = np.atleast_1d(np.asarray(z, dtype=np.complex128))
    return np.stack([c.real, c.imag]).astype(np.float64)


def sorted_parts(z: npt.ArrayLike) -> FloatArray:
    """Complex values as `[2, n]` parts sorted by real then imaginary part (the order mdml-js returns roots in)."""
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


def ss(a: npt.ArrayLike, b: npt.ArrayLike, c: npt.ArrayLike, d: npt.ArrayLike, dt: float = 0.0) -> ct.StateSpace:
    """python-control's ss, typed (its annotations allow nonlinear systems)."""
    return cast(ct.StateSpace, ct.ss(a, b, c, d, dt))


def tf(num: npt.ArrayLike, den: npt.ArrayLike, dt: float = 0.0) -> ct.TransferFunction:
    """python-control's tf, typed."""
    return cast(ct.TransferFunction, ct.tf(num, den, dt))


def feedback(sys1: object, sys2: object = 1, sign: int = -1) -> ct.StateSpace | ct.TransferFunction:
    """python-control's feedback, typed (its annotations take the feedback gain for an int)."""
    fb = cast(Callable[..., ct.StateSpace | ct.TransferFunction], ct.feedback)
    return fb(sys1, sys2, sign=sign)


def system_norm(sys: object, p: int | str) -> float:
    """python-control's system_norm, typed (its annotations take p for an int)."""
    norm = cast(Callable[..., float], ct.system_norm)
    return float(norm(sys, p))
