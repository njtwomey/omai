"""Matrix completion (ALS, biased SGD, implicit ALS, soft-impute) against dense numpy implementations of the same
updates from the same starting factors."""

import numpy as np
import scipy
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def als(
    x: FloatArray, mask: FloatArray, u: FloatArray, v: FloatArray, lam: float, half_sweeps: int
) -> tuple[FloatArray, FloatArray]:
    r = u.shape[1]
    for step in range(half_sweeps):
        if step % 2 == 0:
            for i in range(x.shape[0]):
                js = np.flatnonzero(mask[i])
                if len(js):
                    u[i] = np.linalg.solve(v[js].T @ v[js] + lam * np.eye(r), v[js].T @ x[i, js])
                else:
                    u[i] = 0
        else:
            for j in range(x.shape[1]):
                is_ = np.flatnonzero(mask[:, j])
                if len(is_):
                    v[j] = np.linalg.solve(u[is_].T @ u[is_] + lam * np.eye(r), u[is_].T @ x[is_, j])
                else:
                    v[j] = 0
    return u, v


def implicit_als(
    counts: FloatArray, u: FloatArray, v: FloatArray, lam: float, alpha: float, half_sweeps: int
) -> tuple[FloatArray, FloatArray]:
    r = u.shape[1]
    pref = (counts > 0).astype(np.float64)
    conf = 1 + alpha * counts
    for step in range(half_sweeps):
        if step % 2 == 0:
            for i in range(u.shape[0]):
                c = np.diag(conf[i])
                u[i] = np.linalg.solve(v.T @ c @ v + lam * np.eye(r), v.T @ c @ pref[i])
        else:
            for j in range(v.shape[0]):
                c = np.diag(conf[:, j])
                v[j] = np.linalg.solve(u.T @ c @ u + lam * np.eye(r), u.T @ c @ pref[:, j])
    return u, v


def sgd(
    triplets: list[tuple[int, int, float]], u: FloatArray, v: FloatArray, bi: FloatArray, bj: FloatArray, mean: float,
    lam: float, lr: float, epochs: int,
) -> None:
    for _ in range(epochs):
        for i, j, val in triplets:
            e = val - (mean + bi[i] + bj[j] + u[i] @ v[j])
            bi[i] += lr * (e - lam * bi[i])
            bj[j] += lr * (e - lam * bj[j])
            ui = u[i].copy()
            u[i] += lr * (e * v[j] - lam * u[i])
            v[j] += lr * (e * ui - lam * v[j])


def soft_impute(x: FloatArray, mask: FloatArray, lam: float, steps: int) -> tuple[FloatArray, float]:
    z = np.zeros_like(x)
    for _ in range(steps):
        filled = np.where(mask > 0, x, z)
        uu, s, vt = np.linalg.svd(filled, full_matrices=False)
        z = (uu * np.maximum(s - lam, 0)) @ vt
    s = np.maximum(np.linalg.svd(z, compute_uv=False), 0)
    sse = float(np.sum(mask * (x - z) ** 2))
    return z, 0.5 * sse + lam * float(s.sum())


def cases() -> list[Case]:
    g = rng(160)
    out: list[Case] = []
    m, n, r = 9, 7, 2
    truth = g.normal(size=(m, r)) @ g.normal(size=(r, n))
    mask = (g.uniform(size=(m, n)) < 0.55).astype(np.float64)
    mask[0] = 0  # a row with no observations
    mask[:, 0] = np.maximum(mask[:, 0], (np.arange(m) % 2).astype(np.float64))
    rows, cols = np.nonzero(mask)
    values = truth[rows, cols] + 0.05 * g.normal(size=len(rows))
    x = np.zeros((m, n))
    x[rows, cols] = values
    u0 = 0.3 * g.normal(size=(m, r))
    v0 = 0.3 * g.normal(size=(n, r))
    rf, cf = rows.astype(np.float64), cols.astype(np.float64)
    common = [rf, cf, values, u0, v0]

    for half in (1, 2, 7):
        u, v = als(x, mask, u0.copy(), v0.copy(), 0.3, half)
        out.append(case(f"als/{half}/u", "alsU", common, u, rtol=1e-9, atol=1e-10, half=half, lam=0.3, m=m, n=n))
        out.append(case(f"als/{half}/v", "alsV", common, v, rtol=1e-9, atol=1e-10, half=half, lam=0.3, m=m, n=n))

    triplets = [(int(i), int(j), float(val)) for i, j, val in zip(rows, cols, values, strict=True)]
    mean = float(values.mean())
    for epochs in (1, 4):
        u, v = u0.copy(), v0.copy()
        bi, bj = np.zeros(m), np.zeros(n)
        sgd(triplets, u, v, bi, bj, mean, 0.05, 0.02, epochs)
        pred = mean + bi[:, None] + bj[None, :] + u @ v.T
        out.append(
            case(f"sgd/{epochs}/prediction", "sgd", common, pred, rtol=1e-9, atol=1e-10, epochs=epochs, lam=0.05, lr=0.02, m=m, n=n)
        )

    counts = np.zeros((m, n))
    counts[rows, cols] = np.floor(1 + 4 * g.uniform(size=len(rows)))
    icommon = [rf, cf, counts[rows, cols], u0, v0]
    for half in (1, 2, 6):
        u, v = implicit_als(counts, u0.copy(), v0.copy(), 0.5, 10.0, half)
        out.append(case(f"implicit/{half}/u", "implicitU", icommon, u, rtol=1e-8, atol=1e-10, half=half, lam=0.5, alpha=10.0, m=m, n=n))
        out.append(case(f"implicit/{half}/v", "implicitV", icommon, v, rtol=1e-8, atol=1e-10, half=half, lam=0.5, alpha=10.0, m=m, n=n))

    for steps in (1, 6, 40):
        z, objective = soft_impute(x, mask, 0.8, steps)
        out.append(case(f"softImpute/{steps}/matrix", "softImpute", common, z, rtol=1e-8, atol=1e-9, steps=steps, lam=0.8, m=m, n=n))
        out.append(case(f"softImpute/{steps}/objective", "softImputeObjective", common, np.array(objective), rtol=1e-8, atol=1e-9, steps=steps, lam=0.8, m=m, n=n))
    return out
