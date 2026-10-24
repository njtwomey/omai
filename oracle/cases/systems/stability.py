"""Poles, zeros (invariant zeros by SLICOT) and root loci against python-control; Routh and Jury verdicts against
numpy root counts."""

from typing import cast

import control as ct
import numpy as np
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng
from whitebox_oracle.lti import sorted_parts, ss, stable_system, tf

REFERENCE: dict[str, str] = {"control": ct.__version__, "numpy": np.__version__}


def rhp(c: list[float]) -> FloatArray:
    """The number of roots in the open right half-plane (numpy.roots)."""
    return np.array(float(np.sum(np.roots(c).real > 1e-9)))


def outside(c: list[float]) -> FloatArray:
    """Whether every root is strictly inside the unit circle (numpy.roots), as 1 or 0."""
    return np.array(float(np.all(np.abs(np.roots(c)) < 1)))


def cases() -> list[Case]:
    g = rng(4165)
    out: list[Case] = []
    a, b, c, d = stable_system(g, 4, 2, 2)
    sys = ss(a, b, c, d)
    out.append(case("poles/ss", "poles-ss", [a, b, c, d], sorted_parts(sys.poles()), rtol=1e-10, atol=1e-12))
    out.append(case("zeros/mimo", "zeros-ss", [a, b, c, d], sorted_parts(sys.zeros()), rtol=1e-7, atol=1e-9))
    d0 = np.zeros((2, 2))
    out.append(
        case(
            "zeros/mimo-strictly-proper",
            "zeros-ss",
            [a, b, c, d0],
            sorted_parts(ss(a, b, c, d0).zeros()),
            rtol=1e-7,
            atol=1e-9,
        )
    )
    sa, sb, sc, sd = stable_system(g, 3, 1, 1)
    out.append(
        case(
            "zeros/siso",
            "zeros-ss",
            [sa, sb, sc, sd],
            sorted_parts(ss(sa, sb, sc, sd).zeros()),
            rtol=1e-8,
            atol=1e-10,
        )
    )
    num, den = [1.0, 0.5, 2.0], [1.0, 2.0, 3.0, 1.0]
    out.append(
        case(
            "poles/tf",
            "poles-tf",
            [np.array(num), np.array(den)],
            sorted_parts(tf(num, den).poles()),
            rtol=1e-10,
            atol=1e-12,
        )
    )
    out.append(
        case(
            "zeros/tf",
            "zeros-tf",
            [np.array(num), np.array(den)],
            sorted_parts(tf(num, den).zeros()),
            rtol=1e-10,
            atol=1e-12,
        )
    )

    gains = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 20.0])
    for name, lnum, lden in [
        ("third-order", [1.0], [1.0, 3.0, 2.0, 0.0]),
        ("with-zero", [1.0, 3.0], [1.0, 2.0, 5.0, 0.0]),
        ("complex", [1.0, 1.0], [1.0, 1.0, 4.0]),
    ]:
        data = cast(ct.PoleZeroData, ct.root_locus_map(tf(lnum, lden), gains=gains))
        loci = np.asarray(data.loci)
        for i, k in enumerate(gains):
            out.append(
                case(
                    f"rootLocus/{name}/{k}",
                    "rootLocus",
                    [np.array(lnum), np.array(lden)],
                    sorted_parts(loci[i]),
                    # A gain at a break point gives a double root, which either root finder splits by ~sqrt(eps).
                    rtol=1e-7,
                    atol=1e-7,
                    gain=float(k),
                )
            )

    polys: list[tuple[str, list[float]]] = [
        ("stable", [1.0, 6.0, 11.0, 6.0]),
        ("two-unstable", [1.0, 2.0, 3.0, 10.0]),
        ("one-unstable", [1.0, 1.0, -2.0]),
        ("zero-first-entry", [1.0, 2.0, 2.0, 4.0, 11.0, 10.0]),
        ("zero-row", [1.0, 2.0, 24.0, 48.0, -25.0, -50.0]),
        ("quartic", [1.0, 3.0, 5.0, 4.0, 2.0]),
    ]
    for name, poly in polys:
        out.append(case(f"routh/{name}", "routh", [np.array(poly)], rhp(poly)))
    zpolys: list[tuple[str, list[float]]] = [
        ("stable-2", [1.0, -1.2, 0.5]),
        ("unstable-2", [1.0, -2.1, 1.2]),
        ("stable-4", [1.0, -0.5, 0.3, -0.1, 0.02]),
        ("unstable-3", [1.0, 0.2, -1.0, -0.5]),
        ("stable-3", [2.0, -1.0, 0.4, -0.1]),
    ]
    for name, poly in zpolys:
        out.append(case(f"jury/{name}", "jury", [np.array(poly)], outside(poly)))
    return out
