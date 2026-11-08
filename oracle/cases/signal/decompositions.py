"""Variational mode decomposition against vmdpy, and singular spectrum analysis by numpy's SVD with diagonal
averaging (the components are invariant to the singular vectors' signs)."""

from importlib.metadata import version

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from vmdpy import VMD

REFERENCE: dict[str, str] = {"vmdpy": version("vmdpy"), "numpy": np.__version__}


def ssa_components(x: FloatArray, window: int) -> FloatArray:
    """Elementary reconstructed components of SSA."""
    n = len(x)
    k = n - window + 1
    traj = np.array([x[i : i + k] for i in range(window)])
    u, s, vh = np.linalg.svd(traj, full_matrices=False)
    out = np.zeros((len(s), n))
    for c in range(len(s)):
        m = s[c] * np.outer(u[:, c], vh[c])
        out[c] = [np.mean(m[::-1].diagonal(d)) for d in range(-(window - 1), k)]
    return out


def cases() -> list[Case]:
    g = rng(417)
    out: list[Case] = []
    t = np.arange(120)
    x = np.cos(2 * np.pi * 0.04 * t) + 0.6 * np.cos(2 * np.pi * 0.21 * t + 0.4) + 0.05 * g.normal(size=120)
    for k, alpha, tau, dc, init in [(2, 2000.0, 0.0, False, 1), (3, 1000.0, 0.1, True, 0), (2, 500.0, 0.0, False, 0)]:
        u, _, omega = VMD(x, alpha, tau, k, dc, init, 1e-7)
        params = {"k": k, "alpha": alpha, "tau": tau, "dc": dc, "init": "uniform" if init == 1 else "zero", "tol": 1e-7}
        out.append(case(f"vmd/{k}/{alpha}/{dc}", "vmd", [x], u, rtol=1e-7, atol=1e-9, **params))
        out.append(case(f"vmdOmega/{k}/{alpha}/{dc}", "vmdOmega", [x], omega[-1], rtol=1e-7, atol=1e-10, **params))
    y = np.cumsum(g.normal(size=30)) + np.sin(np.arange(30) / 2)
    for window in [5, 12]:
        out.append(case(f"ssa/{window}", "ssa", [y], ssa_components(y, window), rtol=1e-9, atol=1e-10, window=window))
    return out
