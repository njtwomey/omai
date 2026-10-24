"""System identification references: SINDy against pysindy (STLSQ on a degree-2 polynomial library, derivatives
given) and exact DMD eigenvalues from numpy's SVD and eigendecomposition (Tu et al. 2014)."""

import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"pysindy": ps.__version__ if hasattr(ps, "__version__") else "installed"}


def lorenz(_t: float, x: FloatArray) -> FloatArray:
    return np.array([10.0 * (x[1] - x[0]), x[0] * (28.0 - x[2]) - x[1], x[0] * x[1] - 8.0 / 3.0 * x[2]])


def lorenz_data() -> tuple[FloatArray, FloatArray]:
    t = np.arange(0.0, 4.0, 0.01)
    sol = solve_ivp(lorenz, (0.0, 4.0), [-8.0, 8.0, 27.0], t_eval=t, rtol=1e-10, atol=1e-10)
    x = np.asarray(sol.y).T
    xdot = np.array([lorenz(0.0, row) for row in x])
    return x, xdot


def sindy_coefficients(x: FloatArray, xdot: FloatArray, threshold: float) -> FloatArray:
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold, alpha=0.05), feature_library=ps.PolynomialLibrary(degree=2)
    )
    model.fit(x, t=0.01, x_dot=xdot)
    return np.asarray(model.coefficients(), dtype=np.float64)


def dmd_eigenvalues(snapshots: FloatArray, rank: int) -> FloatArray:
    x1 = snapshots[:, :-1]
    x2 = snapshots[:, 1:]
    u, s, vh = np.linalg.svd(x1, full_matrices=False)
    ur = u[:, :rank]
    vr = vh[:rank].conj().T
    a = ur.T @ x2 @ vr / s[:rank]
    w = np.linalg.eigvals(a)
    order = np.lexsort((w.imag, w.real))
    w = w[order]
    return np.stack([w.real, w.imag])


def cases() -> list[Case]:
    x, xdot = lorenz_data()
    g = rng(26_6)
    noisy = xdot + 0.01 * g.normal(size=xdot.shape)
    out = [
        case(
            "sindy/lorenz/exact",
            "sindy",
            [x, xdot],
            sindy_coefficients(x, xdot, 0.1),
            rtol=1e-8,
            atol=1e-9,
            threshold=0.1,
        ),
        case(
            "sindy/lorenz/noisy",
            "sindy",
            [x, noisy],
            sindy_coefficients(x, noisy, 0.5),
            rtol=1e-8,
            atol=1e-9,
            threshold=0.5,
        ),
    ]
    # Snapshots of a damped travelling wave on 32 points: two complex pairs.
    grid = np.linspace(0.0, 2.0 * np.pi, 32)
    k = np.arange(40)
    snaps = np.array(
        [np.exp(-0.02 * k) * np.cos(xi - 0.3 * k) + 0.5 * np.exp(-0.05 * k) * np.sin(2.0 * xi + 0.7 * k) for xi in grid]
    )
    out.append(case("dmd/wave", "dmd", [snaps], dmd_eigenvalues(snaps, 4), rtol=1e-9, atol=1e-10, rank=4))
    return out
