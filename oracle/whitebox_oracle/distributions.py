"""Fixture cases for a distribution from a frozen scipy.stats distribution: densities, CDFs, quantiles and moments."""

import numpy as np
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen

from .fixture import Case, case
from .helpers import FloatArray

type Frozen = rv_continuous_frozen | rv_discrete_frozen


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
) -> list[Case]:
    """One case per method: `fn` is "<family>.<method>", inputs are the parameters followed by the argument."""

    def add(method: str, arg: FloatArray | None, value: object) -> Case:
        inputs = [*params] if arg is None else [*params, arg]
        out = np.asarray(value, dtype=np.float64)
        return case(f"{label}/{method}", f"{family}.{method}", inputs, out, rtol=rtol, atol=1e-300)

    log_prob = frozen.logpmf(x) if discrete else frozen.logpdf(x)  # type: ignore[union-attr]
    methods: list[tuple[str, FloatArray | None, object]] = [
        ("logProb", x, log_prob),
        ("cdf", x, frozen.cdf(x)),
        ("sf", x, frozen.sf(x)),
        ("logCdf", x, frozen.logcdf(x)),
        ("logSf", x, frozen.logsf(x)),
        ("ppf", q, frozen.ppf(q)),
        ("isf", q, frozen.isf(q)),
        ("mean", None, frozen.mean()),
        ("variance", None, frozen.var()),
        ("entropy", None, frozen.entropy()),
    ]
    return [add(m, a, v) for m, a, v in methods if m not in skip]
