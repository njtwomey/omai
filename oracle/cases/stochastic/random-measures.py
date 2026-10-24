"""Random measures: the expected number of tables of the Chinese restaurant process (Dirichlet and Pitman-Yor, with
scipy.special.gammaln), the exchangeable partition probability function (scipy.special.poch rising factorials) and
the expected number of Indian buffet dishes (harmonic numbers by scipy.special.digamma)."""

import numpy as np
import scipy
import scipy.special as sp
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "numpy": np.__version__}


def tables(alpha: float, n: int, d: float) -> float:
    if d == 0:
        return float(np.sum(alpha / (alpha + np.arange(n))))
    # (alpha + d)_n / (alpha)_n with rising factorials (x)_n = Gamma(x + n) / Gamma(x)
    log_ratio = sp.gammaln(alpha + d + n) + sp.gammaln(alpha) - sp.gammaln(alpha + d) - sp.gammaln(alpha + n)
    return float(alpha / d * (np.exp(log_ratio) - 1))


def eppf(sizes: list[int], alpha: float, d: float) -> float:
    n = sum(sizes)
    k = len(sizes)
    top = float(np.prod([alpha + i * d for i in range(1, k)])) if k > 1 else 1.0
    bottom = float(sp.poch(alpha + 1, n - 1))
    parts = float(np.prod([sp.poch(1 - d, s - 1) for s in sizes]))
    return float(np.log(top / bottom * parts))


def cases() -> list[Case]:
    out: list[Case] = []
    for alpha, n, d in [(1.0, 10, 0.0), (2.5, 50, 0.0), (1.0, 10, 0.3), (0.5, 200, 0.6)]:
        out.append(
            case(
                f"tables/{alpha}/{n}/{d}",
                "tables",
                [],
                np.array(tables(alpha, n, d)),
                rtol=1e-11,
                alpha=alpha,
                n=n,
                d=d,
            )
        )
    for sizes, alpha, d in [
        ([3, 1, 1], 1.0, 0.0),
        ([2, 2], 0.7, 0.0),
        ([4, 2, 1, 1], 2.0, 0.25),
        ([1, 1, 1], 0.5, 0.5),
    ]:
        out.append(
            case(
                f"eppf/{sizes}/{alpha}/{d}",
                "eppf",
                [],
                np.array(eppf(sizes, alpha, d)),
                rtol=1e-11,
                sizes=sizes,
                alpha=alpha,
                d=d,
            )
        )
    for alpha, n in [(1.0, 1), (2.0, 10), (0.7, 100)]:
        harmonic = float(sp.digamma(n + 1) + np.euler_gamma)
        out.append(case(f"dishes/{alpha}/{n}", "dishes", [], np.array(alpha * harmonic), rtol=1e-12, alpha=alpha, n=n))
    return out
