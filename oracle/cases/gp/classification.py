"""Gaussian-process classification against scikit-learn (the Laplace approximation with the logistic likelihood), and
expectation propagation against an independent numpy implementation of the GPML algorithm."""

from typing import Protocol, cast

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from scipy.special import log_ndtr
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Kernel, Matern


class LaplaceBase(Protocol):
    """The parts of scikit-learn's binary Laplace classifier that the cases read."""

    kernel_: Kernel
    log_marginal_likelihood_value_: float

    def log_marginal_likelihood(
        self, theta: FloatArray, eval_gradient: bool = False
    ) -> float | tuple[float, FloatArray]: ...

    def latent_mean_and_variance(self, x: FloatArray) -> tuple[FloatArray, FloatArray]: ...


REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def problem(g: np.random.Generator, per_class: int, d: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    x = np.vstack([g.normal(-0.8, 1.0, size=(per_class, d)), g.normal(0.8, 1.0, size=(per_class, d))])
    y = np.repeat([0.0, 1.0], per_class)
    xt = g.uniform(-2.5, 2.5, size=(9, d))
    return x, y, xt


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(61)
    x, y, xt = problem(g, 10, 2)
    labels = np.where(y == 0, -3.0, 5.0)
    specs: list[tuple[str, Kernel, dict[str, object]]] = [
        ("rbf", ConstantKernel(1.5) * RBF(1.2), {"constant": 1.5, "lengthScale": 1.2, "family": "rbf"}),
        (
            "matern",
            ConstantKernel(2.0) * Matern(length_scale=0.9, nu=2.5),
            {"constant": 2.0, "lengthScale": 0.9, "family": "matern25"},
        ),
    ]
    for name, kernel, spec in specs:
        model = GaussianProcessClassifier(kernel=kernel, optimizer=cast("str", None)).fit(x, labels)
        base = cast("LaplaceBase", model.base_estimator_)
        theta = base.kernel_.theta
        value, gradient = cast("tuple[float, FloatArray]", base.log_marginal_likelihood(theta, eval_gradient=True))
        mean, var = base.latent_mean_and_variance(xt)
        output = {
            "lml": np.array(value),
            "gradient": np.asarray(gradient, dtype=np.float64),
            "proba": np.asarray(model.predict_proba(xt), dtype=np.float64),
            "latentMean": np.asarray(mean, dtype=np.float64),
            "latentVariance": np.asarray(var, dtype=np.float64),
        }
        out.append(
            case(
                f"laplace/fixed/{name}",
                "gpClassifier",
                [x, labels, xt],
                output,
                rtol=1e-6,
                atol=1e-8,
                optimise=False,
                **spec,
            )
        )
    # The optimised fit, with the optimum found from the start both optimisers share.
    kernel = ConstantKernel(1.0) * RBF(1.0)
    model = GaussianProcessClassifier(kernel=kernel).fit(x, y)
    base = cast("LaplaceBase", model.base_estimator_)
    output = {
        "lml": np.array(base.log_marginal_likelihood_value_),
        "theta": np.asarray(base.kernel_.theta, dtype=np.float64),
        "proba": np.asarray(model.predict_proba(xt), dtype=np.float64),
    }
    out.append(
        case(
            "laplace/optimised/rbf",
            "gpClassifier",
            [x, y, xt],
            output,
            rtol=1e-3,
            atol=1e-4,
            optimise=True,
            constant=1.0,
            lengthScale=1.0,
            family="rbf",
        )
    )
    out.extend(ep_cases(g))
    return out


def positive_columns(p: FloatArray) -> FloatArray:
    return np.column_stack([1 - p, p])


def ep_reference(
    k: FloatArray, t: FloatArray, tol: float = 1e-8, max_sweeps: int = 200
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, float]:
    """GPML Algorithm 3.5 for the probit likelihood, written with explicit loops: the site parameters, the posterior
    mean and covariance, and the log marginal likelihood (equation 3.65)."""
    n = len(t)
    sign = 2 * t - 1
    tau = np.zeros(n)
    nu = np.zeros(n)
    sigma = k.copy()
    mu = np.zeros(n)
    log_z = -np.inf
    for _ in range(max_sweeps):
        change = 0.0
        for i in range(n):
            tau_cav = 1 / sigma[i, i] - tau[i]
            nu_cav = mu[i] / sigma[i, i] - nu[i]
            v_cav = 1 / tau_cav
            m_cav = nu_cav * v_cav
            z = sign[i] * m_cav / np.sqrt(1 + v_cav)
            ratio = np.exp(-0.5 * z * z - 0.5 * np.log(2 * np.pi) - log_ndtr(z))
            m_hat = m_cav + sign[i] * v_cav * ratio / np.sqrt(1 + v_cav)
            v_hat = v_cav - v_cav**2 * ratio / (1 + v_cav) * (z + ratio)
            d_tau = 1 / v_hat - tau_cav - tau[i]
            tau_new = tau[i] + d_tau
            nu_new = m_hat / v_hat - nu_cav
            change = max(change, abs(d_tau), abs(nu_new - nu[i]))
            col = sigma[:, i].copy()
            sigma = sigma - d_tau / (1 + d_tau * sigma[i, i]) * np.outer(col, col)
            tau[i] = tau_new
            nu[i] = nu_new
            mu = sigma @ nu
        s = np.sqrt(tau)
        b = np.eye(n) + s[:, None] * k * s[None, :]
        chol = np.linalg.cholesky(b)
        v = np.linalg.solve(chol, s[:, None] * k)
        sigma = k - v.T @ v
        mu = sigma @ nu
        sii = np.diag(sigma)
        tau_cav_all = 1 / sii - tau
        v_cav_all = 1 / tau_cav_all
        m_cav_all = (mu / sii - nu) * v_cav_all
        z_all = sign * m_cav_all / np.sqrt(1 + v_cav_all)
        sigma_tilde = 1 / tau
        mu_tilde = nu / tau
        a = k + np.diag(sigma_tilde)
        chol_a = np.linalg.cholesky(a)
        alpha = np.linalg.solve(a, mu_tilde)
        total = v_cav_all + sigma_tilde
        log_z = float(
            -np.log(np.diag(chol_a)).sum()
            - 0.5 * mu_tilde @ alpha
            + log_ndtr(z_all).sum()
            + 0.5 * np.log(total).sum()
            + ((m_cav_all - mu_tilde) ** 2 / (2 * total)).sum()
        )
        if change < tol:
            break
    return tau, nu, mu, sigma, log_z


def ep_cases(g: np.random.Generator) -> list[Case]:
    out: list[Case] = []
    x, y, xt = problem(g, 8, 2)
    for name, length in [("short", 0.7), ("long", 1.8)]:
        d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
        k = 1.4 * np.exp(-0.5 * d2 / length**2)
        tau, nu, _, _, log_z = ep_reference(k, y)
        a = k + np.diag(1 / tau)
        weights = np.linalg.solve(a, nu / tau)
        d2s = ((xt[:, None, :] - x[None, :, :]) ** 2).sum(-1)
        ks = 1.4 * np.exp(-0.5 * d2s / length**2)
        mean = ks @ weights
        var = 1.4 - np.einsum("ij,ij->i", ks, np.linalg.solve(a, ks.T).T)
        output = {
            "lml": np.array(log_z),
            "tau": tau,
            "nu": nu,
            "proba": positive_columns(np.exp(log_ndtr(mean / np.sqrt(1 + var)))),
        }
        out.append(
            case(
                f"ep/fixed/{name}",
                "gpClassifier",
                [x, y, xt],
                output,
                rtol=1e-6,
                atol=1e-8,
                constant=1.4,
                lengthScale=length,
                family="rbf",
            )
        )
    return out
