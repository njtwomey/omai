"""The Marchenko-Pastur law (dists/continuous/random-matrix.ts). scipy.stats has none, so the reference is a scipy
`rv_continuous` that carries only the textbook density: its cdf is scipy's numerical integral of the density, its
quantile the root of that integral, and its entropy and moments are quadratures; none is the closed form the library
uses."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import scipy
from omai_oracle.distributions import distribution_cases, method_case
from omai_oracle.fixture import Case
from omai_oracle.helpers import FloatArray
from scipy import integrate, stats


def mp_density(x: npt.ArrayLike, ratio: float, variance: float) -> npt.NDArray[np.float64]:
    """The textbook density sqrt((b - x)(x - a)) / (2 pi variance ratio x) of the continuous part (zero outside)."""
    s = np.sqrt(ratio)
    a, b = variance * (1 - s) ** 2, variance * (1 + s) ** 2
    t = np.asarray(x, dtype=np.float64)
    return np.sqrt(np.maximum((b - t) * (t - a), 0.0)) / (2 * np.pi * variance * ratio * t)


def support(ratio: float, variance: float) -> tuple[float, float]:
    s = np.sqrt(ratio)
    return variance * (1 - s) ** 2, variance * (1 + s) ** 2


class MarchenkoPasturReference(stats.rv_continuous):  # type: ignore[type-arg]
    """The Marchenko-Pastur law for a ratio at most 1, defined by its density alone."""

    def _argcheck(self, *args: float) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        ratio, variance = args
        return bool(ratio > 0) and bool(variance > 0)

    def _get_support(self, *args: float, **kwargs: float) -> tuple[float, float]:  # pyright: ignore[reportIncompatibleMethodOverride]
        ratio, variance = args
        return support(ratio, variance)

    def _pdf(self, x: npt.ArrayLike, *args: float) -> npt.NDArray[np.float64]:  # pyright: ignore[reportIncompatibleMethodOverride]
        ratio, variance = args
        return mp_density(x, ratio, variance)


REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "numpy": np.__version__}

MP = MarchenkoPasturReference(name="marchenko_pastur", shapes="ratio, variance")


def integral(f: Callable[[float], float], lo: float, hi: float) -> float:
    """The adaptive quadrature of `f` over [lo, hi]."""
    return float(integrate.quad(f, lo, hi, epsabs=1e-13, epsrel=1e-13)[0])


def below_one(i: int, ratio: float, variance: float, q: FloatArray) -> list[Case]:
    """The full method table for a ratio at most 1, from the numerical reference."""
    r, v = np.array([ratio]), np.array([variance])
    lo, hi = support(ratio, variance)
    x = lo + (hi - lo) * np.array([0.02, 0.15, 0.4, 0.55, 0.8, 0.97])
    label = f"ratio{i}"
    out = distribution_cases(
        label,
        "marchenkoPastur",
        [r, v],
        MP(r, v),
        x,
        q,
        rtol=1e-7,
        skip=("logCdf", "logSf", "entropy", "mean", "variance"),
    )

    def density(t: float) -> float:
        return float(mp_density(t, ratio, variance))

    frozen = MP(ratio, variance)
    entropy = integral(lambda t: -frozen.pdf(t) * frozen.logpdf(t), lo, hi)
    m1 = integral(lambda t: t * density(t), lo, hi)
    m2 = integral(lambda t: t * t * density(t), lo, hi)
    out += [
        method_case(label, "marchenkoPastur", "entropy", [r, v], None, np.array([entropy]), rtol=1e-7),
        method_case(label, "marchenkoPastur", "mean", [r, v], None, np.array([m1]), rtol=1e-8),
        method_case(label, "marchenkoPastur", "variance", [r, v], None, np.array([m2 - m1 * m1]), rtol=1e-7),
    ]
    return out


def above_one(i: int, ratio: float, variance: float) -> list[Case]:
    """A ratio above 1: the atom 1 - 1/ratio at zero, then the density integrated numerically."""
    r, v = np.array([ratio]), np.array([variance])
    lo, hi = support(ratio, variance)
    atom = 1 - 1 / ratio
    x = lo + (hi - lo) * np.array([0.05, 0.3, 0.6, 0.9])
    cdf = [atom + integral(lambda u: float(mp_density(u, ratio, variance)), lo, float(t)) for t in x]
    label = f"atom{i}"
    return [
        method_case(label, "marchenkoPastur", "logProb", [r, v], x, np.log(mp_density(x, ratio, variance)), rtol=1e-10),
        method_case(label, "marchenkoPastur", "cdf", [r, v], x, np.array(cdf), rtol=1e-8),
        method_case(label, "marchenkoPastur", "mean", [r, v], None, np.array([variance]), rtol=1e-12),
        method_case(label, "marchenkoPastur", "variance", [r, v], None, np.array([variance**2 * ratio]), rtol=1e-12),
    ]


def cases() -> list[Case]:
    q = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    out: list[Case] = []
    for i, (ratio, variance) in enumerate([(0.1, 1.0), (0.5, 2.0), (0.9, 0.5)]):
        out += below_one(i, ratio, variance, q)
    for i, (ratio, variance) in enumerate([(2.0, 1.0), (4.0, 0.5)]):
        out += above_one(i, ratio, variance)
    return out
