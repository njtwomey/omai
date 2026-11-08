"""Fixture cases for a distribution from a frozen scipy.stats distribution: densities, CDFs, quantiles and moments."""

from collections.abc import Mapping

import numpy as np
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen

from .fixture import Case, case
from .helpers import FloatArray

type Frozen = rv_continuous_frozen | rv_discrete_frozen

METHODS: tuple[str, ...] = ("logProb", "cdf", "sf", "logCdf", "logSf", "ppf", "isf", "mean", "variance", "entropy")


def method_case(
    label: str,
    family: str,
    method: str,
    params: list[FloatArray],
    arg: FloatArray | None,
    value: object,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-300,
) -> Case:
    """One case `<family>.<method>`: the parameters (counted by `arity`) followed by the method's argument, if any."""
    inputs = [*params] if arg is None else [*params, arg]
    out = np.asarray(value, dtype=np.float64)
    return case(f"{label}/{method}", f"{family}.{method}", inputs, out, rtol=rtol, atol=atol, arity=len(params))


def distribution_cases(
    label: str,
    family: str,
    params: list[FloatArray],
    frozen: Frozen,
    x: FloatArray,
    q: FloatArray,
    *,
    rtol: float = 1e-10,
    discrete: bool = False,
    skip: tuple[str, ...] = (),
    tolerances: Mapping[str, float] | None = None,
) -> list[Case]:
    """One case per method of a frozen scipy distribution, with per-method tolerance overrides in `tolerances`."""
    log_prob = frozen.logpmf(x) if discrete else frozen.logpdf(x)  # type: ignore[union-attr]
    values: dict[str, tuple[FloatArray | None, object]] = {
        "logProb": (x, log_prob),
        "cdf": (x, frozen.cdf(x)),
        "sf": (x, frozen.sf(x)),
        "logCdf": (x, frozen.logcdf(x)),
        "logSf": (x, frozen.logsf(x)),
        "ppf": (q, frozen.ppf(q)),
        "isf": (q, frozen.isf(q)),
        "mean": (None, frozen.mean()),
        "variance": (None, frozen.var()),
        "entropy": (None, frozen.entropy()),
    }
    tol = dict(tolerances or {})
    return [
        method_case(label, family, m, params, values[m][0], values[m][1], rtol=tol.get(m, rtol))
        for m in METHODS
        if m not in skip
    ]
