"""Symmetric eigendecomposition against numpy.linalg.eigh (eigenvector signs fixed by the shared convention)."""

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng, sign_fixed

REFERENCE: dict[str, str] = {"numpy": np.__version__}


def cases() -> list[Case]:
    g = rng(25)
    out: list[Case] = []
    for n in [2, 5, 8]:
        m = g.normal(size=(n, n))
        a = (m + m.T) / 2
        w, v = np.linalg.eigh(a)
        out.append(case(f"eigh/{n}/values", "eigh", [a], w, rtol=1e-10, atol=1e-13, part="eigenvalues"))
        out.append(case(f"eigh/{n}/vectors", "eigh", [a], sign_fixed(v), rtol=1e-9, atol=1e-11, part="eigenvectors"))
    m = g.normal(size=(2, 4, 4))
    batch = (m + np.swapaxes(m, -1, -2)) / 2
    out.append(case("eigvalsh/batch", "eigvalsh", [batch], np.linalg.eigvalsh(batch), rtol=1e-10, atol=1e-13))
    return out
