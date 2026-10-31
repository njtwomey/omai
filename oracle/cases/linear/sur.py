"""Seemingly unrelated regressions against numpy: Zellner's feasible GLS written with the stacked system and the
Kronecker product of the inverse error covariance (one step from equation-by-equation least squares, and iterated),
compared on the coefficients, the error covariance and the standard errors."""

import numpy as np
import numpy.typing as npt
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from scipy.linalg import block_diag

REFERENCE: dict[str, str] = {"numpy": np.__version__}

type Mat = npt.NDArray[np.float64]


def stacked(x: Mat, y: Mat, equations: list[list[int]], steps: int) -> tuple[Mat, Mat, Mat]:
    n, k = y.shape
    blocks = [np.column_stack([x[:, cols], np.ones(n)]) for cols in equations]
    big = block_diag(*blocks)
    ys = y.T.reshape(-1)
    beta = np.linalg.lstsq(big, ys, rcond=None)[0]
    sigma = np.eye(k)
    cov = np.linalg.inv(big.T @ big)
    for _ in range(steps):
        resid = (ys - big @ beta).reshape(k, n)
        sigma = resid @ resid.T / n
        weight = np.kron(np.linalg.inv(sigma), np.eye(n))
        cov = np.linalg.inv(big.T @ weight @ big)
        beta = cov @ big.T @ weight @ ys
    return beta, sigma, cov


def cases() -> list[Case]:
    g = rng(1962)
    out: list[Case] = []
    n = 90
    x = g.normal(size=(n, 3))
    common = g.normal(size=n)
    y = np.column_stack(
        [
            1.0 + 2.0 * x[:, 0] + common + 0.4 * g.normal(size=n),
            -1.0 + 3.0 * x[:, 1] - x[:, 2] + common + 0.4 * g.normal(size=n),
            0.5 * x[:, 0] + 0.5 * x[:, 2] - common + 0.6 * g.normal(size=n),
        ]
    )
    equations = [[0], [1, 2], [0, 2]]
    for steps in (1, 2, 10):
        beta, sigma, cov = stacked(x, y, equations, steps)
        out.append(
            case(
                f"sur/steps{steps}",
                "sur",
                [x, y],
                {"beta": beta, "sigma": sigma, "bse": np.sqrt(np.diag(cov))},
                rtol=1e-8,
                atol=1e-9,
                maxIter=steps,
            )
        )
    return out
