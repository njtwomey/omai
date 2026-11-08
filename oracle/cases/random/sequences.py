"""Quasi-random sequences against scipy.stats.qmc: unscrambled Sobol' and Halton exactly, and for the randomised
samplers (scrambled Sobol', Poisson disk), whose random streams cannot match scipy's, the statistics they share:
mean centred discrepancy and the number of points that fill the space."""

import numpy as np
import scipy
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray
from scipy.stats import qmc
from scipy.stats._sobol import _initialize_v  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

SEEDS = range(20)


def sobol(n: int, d: int, skip: int, bits: int) -> FloatArray:
    engine = qmc.Sobol(d, scramble=False, bits=bits)
    if skip > 0:
        engine.fast_forward(skip)
    return engine.random(n)


def halton(n: int, d: int, skip: int) -> FloatArray:
    engine = qmc.Halton(d, scramble=False)
    if skip > 0:
        engine.fast_forward(skip)
    return engine.random(n)


def fingerprint(x: FloatArray) -> FloatArray:
    """Two weighted row sums of a wide sample; exact in float64 because every coordinate is a multiple of 2^-30."""
    j = np.arange(x.shape[1])
    w = np.stack([j + 1.0, (j * j % 97) + 1.0], axis=1)
    return x @ w


def direction_fingerprint(d: int, bits: int) -> FloatArray:
    """Two weighted sums over dimensions of scipy's direction numbers v[dim, j] (exact: below 2^53), one per bit j.

    scipy's `fast_forward` walks every skipped point, so points at indices near 2^30 are out of reach in 1111
    dimensions; the direction numbers decide those points, and point 2^(j+1) - 1 of the sequence is v[:, j] / 2^bits.
    """
    v = np.zeros((d, bits), dtype=np.uint32)
    _initialize_v(v, dim=d, bits=bits)
    j = np.arange(d)
    w = np.stack([j + 1.0, (j * j % 97) + 1.0])
    return w @ v.astype(np.float64)


def scrambled_discrepancy(n: int, d: int) -> float:
    values = [qmc.discrepancy(qmc.Sobol(d, rng=np.random.default_rng(s)).random(n)) for s in SEEDS]
    return float(np.mean(values))


def uniform_discrepancy(n: int, d: int) -> float:
    values = [qmc.discrepancy(np.random.default_rng(s).random((n, d))) for s in SEEDS]
    return float(np.mean(values))


def poisson_disk_count(d: int, radius: float) -> float:
    counts = [len(qmc.PoissonDisk(d, radius=radius, rng=np.random.default_rng(s)).fill_space()) for s in range(5)]
    return float(np.mean(counts))


def cases() -> list[Case]:
    out: list[Case] = []
    for n, d, skip, bits in [(16, 5, 0, 30), (32, 2, 0, 30), (8, 40, 5, 30), (8, 3, 0, 32), (8, 3, 1000, 16)]:
        out.append(
            case(f"sobol/{n}x{d}/{skip}/{bits}", "sobol", [], sobol(n, d, skip, bits), n=n, d=d, skip=skip, bits=bits)
        )
    columns = [1, 100, 500, 1000, 1109, 1110]
    wide = sobol(8, 1111, 2**16 - 3, 30)[:, columns]
    out.append(case("sobol/columns", "sobolColumns", [], wide, n=8, d=1111, skip=2**16 - 3, columns=columns))
    out.append(case("sobol/directions", "sobolDirections", [], direction_fingerprint(1111, 30), d=1111, bits=30))
    for skip in [0, 4093, 2**16 - 16]:
        x = sobol(16, 1111, skip, 30)
        out.append(case(f"sobol/fingerprint/{skip}", "sobolFingerprint", [], fingerprint(x), n=16, d=1111, skip=skip))
    for n, d, skip in [(20, 5, 0), (10, 30, 7), (6, 3, 1)]:
        out.append(case(f"halton/{n}x{d}/{skip}", "halton", [], halton(n, d, skip), n=n, d=d, skip=skip))
    for n, d in [(64, 3), (256, 2)]:
        x = sobol(n, d, 0, 30)
        cd = np.asarray(qmc.discrepancy(x))
        out.append(case(f"discrepancy/sobol/{n}x{d}", "centeredDiscrepancy", [x], cd, n=n, d=d))
        # Mean centred discrepancy over seeds: a statistic of the method, compared with a loose band.
        stats = np.array([scrambled_discrepancy(n, d), uniform_discrepancy(n, d)])
        out.append(case(f"discrepancy/scrambled/{n}x{d}", "scrambledDiscrepancy", [], stats, rtol=0.5, n=n, d=d))
    for d, radius in [(2, 0.05), (3, 0.15), (4, 0.3)]:
        count = np.asarray(poisson_disk_count(d, radius))
        # Bridson's algorithm keeps a point active until a round of candidates fails; scipy retires it after one round,
        # so ours fills the box a little more densely. The band allows for that.
        out.append(case(f"poissonDisk/{d}/{radius}", "poissonDiskCount", [], count, rtol=0.25, d=d, radius=radius))
    return out
