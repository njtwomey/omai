"""Shrinkage towards targets other than a scaled identity (covariance/shrinkage-targets.ts). scikit-learn has the
scaled-identity target only (`ledoit_wolf`, `oas`), so it is the reference there; the diagonal, constant-correlation and
single-factor targets and the Rao-Blackwell Ledoit-Wolf intensity have no installed implementation, and their references
are matrix-form transcriptions of the published formulas (Ledoit and Wolf 2003, 2004; Schafer and Strimmer 2005; Chen,
Wiesel, Eldar and Hero 2010), written independently of the TypeScript, which loops over observations."""

from importlib.metadata import version

import numpy as np
import numpy.typing as npt
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.covariance import ledoit_wolf, oas

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

TOL = {"rtol": 1e-9, "atol": 1e-12}

type Matrix = npt.NDArray[np.float64]


def centred(x: Matrix) -> Matrix:
    return x - x.mean(axis=0)


def diagonal_target(s: Matrix) -> Matrix:
    return np.diag(np.diag(s))


def constant_correlation_target(s: Matrix) -> Matrix:
    sd = np.sqrt(np.diag(s))
    p = s.shape[0]
    corr = s / np.outer(sd, sd)
    rbar = (corr.sum() - p) / (p * (p - 1))
    f = rbar * np.outer(sd, sd)
    np.fill_diagonal(f, np.diag(s))
    return f


def single_factor_target(x: Matrix, s: Matrix) -> Matrix:
    mkt = x.mean(axis=1)
    t = x.shape[0]
    cov_mkt = x.T @ mkt / t
    var_mkt = float(mkt @ mkt / t)
    f = np.outer(cov_mkt, cov_mkt) / var_mkt
    np.fill_diagonal(f, np.diag(s))
    return f


def diagonal_intensity(x: Matrix, s: Matrix) -> float:
    t = x.shape[0]
    y = x**2
    pi = y.T @ y / t - s**2
    off = ~np.eye(s.shape[0], dtype=bool)
    gamma = float(np.sum(s[off] ** 2))
    return float(np.clip(pi[off].sum() / (gamma * t), 0, 1))


def constant_correlation_intensity(x: Matrix, s: Matrix) -> float:
    t, p = x.shape
    sd = np.sqrt(np.diag(s))
    rbar = float((np.sum(s / np.outer(sd, sd)) - p) / (p * (p - 1)))
    f = constant_correlation_target(s)
    y = x**2
    pi_mat = y.T @ y / t - s**2
    pi = float(pi_mat.sum())
    # theta[i, j] = mean_t (x_ti^2 - s_ii)(x_ti x_tj - s_ij) = mean_t x_ti^3 x_tj - s_ii s_ij
    theta = (x**3).T @ x / t - np.diag(s)[:, None] * s
    np.fill_diagonal(theta, 0.0)
    rho = float(np.trace(pi_mat) + rbar * np.sum(np.outer(1 / sd, sd) * theta))
    gamma = float(np.sum((f - s) ** 2))
    return float(np.clip((pi - rho) / gamma / t, 0, 1))


def single_factor_intensity(x: Matrix, s: Matrix) -> float:
    t = x.shape[0]
    mkt = x.mean(axis=1)
    cov_mkt = x.T @ mkt / t
    var_mkt = float(mkt @ mkt / t)
    f = single_factor_target(x, s)
    gamma = float(np.sum((s - f) ** 2))
    y = x**2
    pi = float((y.T @ y / t - s**2).sum())
    rdiag = float((x**4).sum() / t - np.sum(np.diag(s) ** 2))
    z = x * mkt[:, None]
    v1 = y.T @ z / t - cov_mkt[:, None] * s
    roff1 = float(np.sum(v1 * cov_mkt[None, :]) / var_mkt - np.sum(np.diag(v1) * cov_mkt) / var_mkt)
    v3 = z.T @ z / t - var_mkt * s
    roff3 = float(np.sum(v3 * np.outer(cov_mkt, cov_mkt)) / var_mkt**2 - np.sum(np.diag(v3) * cov_mkt**2) / var_mkt**2)
    rho = rdiag + 2 * roff1 - roff3
    return float(np.clip((pi - rho) / gamma / t, 0, 1))


def rblw_intensity(s: Matrix, t: int) -> float:
    p = s.shape[0]
    tr_s, tr_s2 = float(np.trace(s)), float(np.trace(s @ s))
    return float(min(1.0, (((t - 2) / t) * tr_s2 + tr_s**2) / ((t + 2) * (tr_s2 - tr_s**2 / p))))


def data(seed: int, n: int, p: int, factors: int) -> Matrix:
    """Skewed, heavy-tailed returns with a factor structure, so the fourth moments are not Gaussian ones."""
    g = rng(seed)
    load = g.standard_normal((p, factors))
    f = g.standard_t(6, (n, factors))
    e = g.standard_normal((n, p)) ** 3 / 3.0
    return np.asarray(f @ load.T + e + g.uniform(-1, 1, p), dtype=np.float64)


def cases() -> list[Case]:
    out: list[Case] = []
    for name, (seed, n, p, k) in {"a": (5, 40, 10, 2), "b": (6, 60, 15, 1), "c": (7, 25, 12, 3)}.items():
        raw = data(seed, n, p, k)
        x = centred(raw)
        s = x.T @ x / n
        targets = {
            "diagonal": (diagonal_target(s), diagonal_intensity(x, s)),
            "constant-correlation": (constant_correlation_target(s), constant_correlation_intensity(x, s)),
            "single-factor": (single_factor_target(x, s), single_factor_intensity(x, s)),
        }
        for target, (f, delta) in targets.items():
            out.append(case(f"{name}/{target}/target", "target", [raw], f, **TOL, target=target))
            out.append(case(f"{name}/{target}/intensity", "intensity", [raw], np.array(delta), **TOL, target=target))
            out.append(
                case(
                    f"{name}/{target}/covariance",
                    "covariance",
                    [raw],
                    (1 - delta) * s + delta * f,
                    **TOL,
                    target=target,
                )
            )
        mu = float(np.trace(s) / p)
        lw_cov, lw_delta = ledoit_wolf(raw)
        oas_cov, oas_delta = oas(raw)
        d = rblw_intensity(s, n)
        out += [
            case(
                f"{name}/identity/ledoit-wolf/intensity",
                "intensity",
                [raw],
                np.array(lw_delta),
                **TOL,
                target="identity",
                rule="analytic",
            ),
            case(
                f"{name}/identity/ledoit-wolf/covariance",
                "covariance",
                [raw],
                lw_cov,
                **TOL,
                target="identity",
                rule="analytic",
            ),
            case(
                f"{name}/identity/oas/intensity",
                "intensity",
                [raw],
                np.array(oas_delta),
                **TOL,
                target="identity",
                rule="oas",
            ),
            case(f"{name}/identity/oas/covariance", "covariance", [raw], oas_cov, **TOL, target="identity", rule="oas"),
            case(
                f"{name}/identity/rblw/intensity",
                "intensity",
                [raw],
                np.array(d),
                **TOL,
                target="identity",
                rule="rblw",
            ),
            case(
                f"{name}/identity/rblw/covariance",
                "covariance",
                [raw],
                (1 - d) * s + d * mu * np.eye(p),
                **TOL,
                target="identity",
                rule="rblw",
            ),
        ]
    return out
