"""Singular value decomposition and its uses against numpy.linalg (singular vector signs fixed by the convention)."""

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng, sign_fixed

REFERENCE: dict[str, str] = {"numpy": np.__version__}


def fixed(a: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    # Flip U's columns to the convention and the matching rows of Vh with them.
    flips = np.sign(sign_fixed(u) * u).max(axis=-2)
    return u * flips[..., None, :], s, vh * flips[..., :, None]


def low_rank(a: FloatArray, r: int) -> FloatArray:
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    return (u[:, :r] * s[:r]) @ vh[:r]


def cases() -> list[Case]:
    g = rng(26)
    out: list[Case] = []
    for m, n in [(5, 3), (4, 4), (3, 6)]:
        a = g.normal(size=(m, n))
        u, s, vh = fixed(a)
        out.append(case(f"svd/{m}x{n}/s", "svd", [a], s, rtol=1e-10, part="s"))
        out.append(case(f"svd/{m}x{n}/u", "svd", [a], u, rtol=1e-9, atol=1e-11, part="u"))
        out.append(case(f"svd/{m}x{n}/vh", "svd", [a], vh, rtol=1e-9, atol=1e-11, part="vh"))
        out.append(case(f"pinv/{m}x{n}", "pinv", [a], np.linalg.pinv(a), rtol=1e-10, atol=1e-13))
    batch = g.normal(size=(2, 3, 4))
    out.append(case("svdvals/batch", "svdvals", [batch], np.linalg.svd(batch, compute_uv=False), rtol=1e-10))
    deficient = g.normal(size=(5, 2)) @ g.normal(size=(2, 4))
    out.append(case("pinv/rank-deficient", "pinv", [deficient], np.linalg.pinv(deficient), rtol=1e-9, atol=1e-12))
    a = g.normal(size=(5, 4))
    out.append(
        case("lowRankApproximation", "lowRankApproximation", [a], low_rank(a, 2), rtol=1e-10, atol=1e-13, rank=2)
    )
    return out
