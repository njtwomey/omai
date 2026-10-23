"""High-precision references with mpmath: elementwise values and derivatives of scalar functions."""

from collections.abc import Callable
from typing import Any

import mpmath as mp
import numpy as np
import numpy.typing as npt

from .fixture import Case, case

DPS = 50

# mpmath is untyped: its numbers (mpf) are named once here so signatures stay readable.
type Mpf = Any
type MpFn = Callable[..., Mpf]
type F64 = npt.NDArray[np.float64]


def mpvec(f: MpFn, *arrays: npt.ArrayLike, dps: int = DPS) -> F64:
    """Evaluate an mpmath function elementwise at high precision, rounding once to float64."""
    with mp.workdps(dps):
        cols = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64) for a in arrays])
        out = [float(f(*[mp.mpf(float(c.flat[i])) for c in cols])) for i in range(cols[0].size)]
    return np.asarray(out, dtype=np.float64).reshape(cols[0].shape)


def mpgrad(f: MpFn, wrt: int, *arrays: npt.ArrayLike, dps: int = DPS) -> F64:
    """The partial derivative of an mpmath function in argument `wrt`, elementwise (mpmath numerical differentiation at
    high precision, accurate to far more digits than float64)."""

    def partial(*args: Mpf) -> Mpf:
        def g(t: Mpf) -> Mpf:
            return f(*[t if j == wrt else a for j, a in enumerate(args)])

        return mp.diff(g, args[wrt])

    return mpvec(partial, *arrays, dps=dps)


def value_and_grads(
    name: str,
    fn: str,
    f: MpFn,
    inputs: list[F64],
    *,
    wrt: list[int],
    rtol: float = 1e-13,
    atol: float = 0.0,
    grad_rtol: float = 1e-10,
    grad_atol: float = 0.0,
    exact: dict[int, MpFn] | None = None,
    dps: int = DPS,
    **params: int | float | str,
) -> list[Case]:
    """A value case and one derivative case per differentiable argument, all from mpmath. `exact` gives closed-form
    derivatives (evaluated in mpmath) for arguments where numerical differentiation would step outside the domain."""
    out = [case(name, fn, list(inputs), mpvec(f, *inputs, dps=dps), rtol=rtol, atol=atol, **params)]
    for i in wrt:
        d = mpvec(exact[i], *inputs, dps=dps) if exact is not None and i in exact else mpgrad(f, i, *inputs, dps=dps)
        out.append(case(f"{name}/d{i}", fn, list(inputs), d, rtol=grad_rtol, atol=grad_atol, wrt=i, **params))
    return out


def _identity(x: Mpf) -> Mpf:
    return x


def _logit(x: Mpf) -> Mpf:
    return mp.log(x / (1 - x))


def _expit(y: Mpf) -> Mpf:
    return 1 / (1 + mp.exp(-y))


_DOMAINS: dict[str, tuple[MpFn, MpFn]] = {
    "real": (_identity, _identity),
    "positive": (mp.log, mp.exp),
    "unit": (_logit, _expit),
}


def invert(cdf: MpFn, sf: MpFn, p: Mpf, x0: float | Mpf, domain: str = "real") -> Mpf:
    """Solve cdf(x) = p in log space on the smaller tail (sf = 1 - cdf), so tiny probabilities keep full precision.
    For `domain` "positive" or "unit" the search runs in log x or logit x, so iterates never leave the domain."""
    to, back = _DOMAINS[domain]

    def residual(y: Mpf) -> Mpf:
        return mp.log(cdf(back(y))) - mp.log(p) if p < 0.5 else mp.log(sf(back(y))) - mp.log(1 - p)

    # Newton from the float64 starting point (already close), with the derivative from mpmath.
    y = to(mp.mpf(x0))
    for _ in range(100):
        step = residual(y) / mp.diff(residual, y)
        y -= step
        if abs(step) < mp.mpf(10) ** (-mp.mp.dps + 5) * (1 + abs(y)):
            break
    if abs(residual(y)) > mp.mpf(10) ** -30:
        raise ValueError(f"invert: residual {residual(y)} at p = {p}")
    return back(y)
