"""Preconditioned optimisers (optim/preconditioned.ts): matrix inverse roots and Shampoo against scipy.linalg."""

import numpy as np
import scipy
import scipy.linalg
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng, spd
from omai_oracle.optim_trajectories import A_MATRIX, C_MATRIX, P0, P0_MATRIX, A, C, gradient

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def inverse_root(a: FloatArray, p: int, ridge: float) -> FloatArray:
    """(A + ridge I)^(-1/p) by scipy.linalg.fractional_matrix_power (real part)."""
    m = a + ridge * np.eye(a.shape[0])
    return np.real(np.asarray(scipy.linalg.fractional_matrix_power(m, -1.0 / p), dtype=np.complex128))


def shampoo_path(p0: FloatArray, a: FloatArray, c: FloatArray, steps: int, lr: float, eps: float) -> FloatArray:
    """Shampoo on the shared objective: per-mode statistics eps I + sum G_(i) G_(i)^T, roots by scipy."""
    p = p0.copy()
    k = p.ndim
    stats = [eps * np.eye(d) for d in p.shape]
    out: list[FloatArray] = []
    for _ in range(steps):
        g = gradient(p, a, c)
        pre = g
        for i in range(k):
            gi = np.moveaxis(g, i, 0).reshape(g.shape[i], -1)
            stats[i] = stats[i] + gi @ gi.T
            root = inverse_root(stats[i], 2 * k, 0.0)
            pre = np.moveaxis(np.tensordot(root, np.moveaxis(pre, i, 0), 1), 0, i)
        p = p - lr * pre
        out.append(p.copy())
    return np.stack(out)


def cases() -> list[Case]:
    g = rng(11)
    out: list[Case] = []
    for i, (n, p, ridge) in enumerate([(3, 2, 0.0), (4, 4, 0.0), (3, 4, 0.1), (5, 6, 1e-3)]):
        m = spd(g, n)
        out.append(
            case(
                f"matrixInversePthRoot/{i}",
                "matrixInversePthRoot",
                [m],
                inverse_root(m, p, ridge),
                rtol=1e-10,
                atol=1e-12,
                p=p,
                ridge=ridge,
            )
        )
    for name, (p0, a, c) in {"vector": (P0, A, C), "matrix": (P0_MATRIX, A_MATRIX, C_MATRIX)}.items():
        path = shampoo_path(p0, a, c, 8, 0.3, 1e-2)
        out.append(
            case(
                f"shampoo/{name}",
                "shampoo",
                [p0, a, c],
                path,
                rtol=1e-9,
                atol=1e-11,
                steps=8,
                learningRate=0.3,
                eps=1e-2,
            )
        )
    return out
