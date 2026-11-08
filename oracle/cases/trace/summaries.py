"""Tensor summaries recorded at summary detail, against numpy (moments, norm and np.histogram)."""

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__}


def moments(x: np.ndarray) -> np.ndarray:
    """Count, mean, population standard deviation, extremes, Euclidean norm and fraction of zeros."""
    return np.array(
        [x.size, x.mean(), x.std(), x.min(), x.max(), np.linalg.norm(x), np.mean(x == 0)],
        dtype=np.float64,
    )


def cases() -> list[Case]:
    g = rng(70)
    arrays: list[tuple[str, np.ndarray]] = [
        ("normal", g.normal(size=200)),
        ("uniform", g.uniform(-3.0, 5.0, size=57)),
        ("sparse", np.where(g.uniform(size=100) < 0.7, 0.0, g.normal(size=100))),
        ("heavy", g.standard_t(2.0, size=300)),
    ]
    out: list[Case] = []
    for label, x in arrays:
        out.append(case(f"moments/{label}", "moments", [x], moments(x)))
        for bins in (10, 7):
            counts, edges = np.histogram(x, bins=bins)
            out.append(case(f"histogram/{label}/{bins}", "histogram", [x], counts.astype(np.float64), bins=bins))
            out.append(case(f"edges/{label}/{bins}", "edges", [x], edges, rtol=1e-12, atol=1e-12, bins=bins))
    return out
