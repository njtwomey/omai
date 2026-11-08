"""Linear dynamical systems against numpy and scipy.stats: the smoothed means and covariances of the states and the log
likelihood of the observations, by conditioning the joint Gaussian of states and observations (built directly from
A, C, Q, R and the initial moments) on the observations."""

import numpy as np
import scipy
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng, spd
from scipy.stats import multivariate_normal

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def joint(
    a: FloatArray, c: FloatArray, q: FloatArray, r: FloatArray, m0: FloatArray, p0: FloatArray, T: int
) -> tuple[FloatArray, FloatArray]:
    """Mean and covariance of (x_0..x_{T-1}, y_0..y_{T-1}) stacked: Cov(x_t, x_s) = A^(t-s) Var(x_s) for t >= s."""
    n, m = a.shape[0], c.shape[0]
    means = [m0]
    variances = [p0]
    for _ in range(1, T):
        means.append(a @ means[-1])
        variances.append(a @ variances[-1] @ a.T + q)
    sx = np.zeros((n * T, n * T))
    for t in range(T):
        for s in range(t + 1):
            block = np.linalg.matrix_power(a, t - s) @ variances[s]
            sx[t * n : (t + 1) * n, s * n : (s + 1) * n] = block
            sx[s * n : (s + 1) * n, t * n : (t + 1) * n] = block.T
    big_c = np.kron(np.eye(T), c)
    mean = np.concatenate([np.concatenate(means), big_c @ np.concatenate(means)])
    sxy = sx @ big_c.T
    syy = big_c @ sx @ big_c.T + np.kron(np.eye(T), r)
    cov = np.block([[sx, sxy], [sxy.T, syy]])
    del m
    return mean, cov


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(41)
    for t, (n, m, T) in enumerate([(1, 1, 6), (2, 1, 8), (2, 2, 5), (3, 2, 4)]):
        a = g.normal(size=(n, n)) * 0.6
        c = g.normal(size=(m, n))
        q = spd(g, n) / (3 * n)
        r = spd(g, m) / (2 * m)
        m0 = g.normal(size=n)
        p0 = spd(g, n) / n
        mean, cov = joint(a, c, q, r, m0, p0, T)
        y = g.normal(size=(T, m))
        nx = n * T
        mx, my = mean[:nx], mean[nx:]
        sxy = cov[:nx, nx:]
        syy = cov[nx:, nx:]
        post_mean = mx + sxy @ np.linalg.solve(syy, y.ravel() - my)
        post_cov = cov[:nx, :nx] - sxy @ np.linalg.solve(syy, sxy.T)
        blocks = np.array([post_cov[s * n : (s + 1) * n, s * n : (s + 1) * n] for s in range(T)])
        loglik = multivariate_normal(my, syy).logpdf(y.ravel())
        out.append(
            case(
                f"lds/{t}",
                "lds",
                [a, c, q, r, m0, p0, y],
                np.concatenate([post_mean, blocks.ravel(), [loglik]]),
                rtol=1e-9,
                atol=1e-11,
            )
        )
    return out
