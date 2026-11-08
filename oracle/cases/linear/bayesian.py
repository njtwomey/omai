"""Conjugate Bayesian linear regression against numpy's closed form (Bishop, PRML 3.3): the posterior over the weights
for a zero-mean isotropic Gaussian prior of precision alpha and noise precision beta, the log marginal likelihood, the
posterior predictive mean and variance at new inputs, and MacKay's fixed-point evidence maximisation run for a fixed
number of iterations (numpy has no function for these; they are formed from their definitions with numpy.linalg)."""

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__}


def vandermonde(x: FloatArray, degree: int) -> FloatArray:
    """The polynomial design [1, x, x^2, ...] of a vector, columns increasing in power."""
    return np.vander(x, degree + 1, increasing=True)


def posterior(
    phi: FloatArray, y: FloatArray, alpha: float, beta: float
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """The mean, covariance, precision and log evidence of the weights."""
    n, p = phi.shape
    precision = alpha * np.eye(p) + beta * phi.T @ phi
    covariance = np.linalg.inv(precision)
    mean = beta * covariance @ phi.T @ y
    energy = beta / 2 * float(np.sum((y - phi @ mean) ** 2)) + alpha / 2 * float(mean @ mean)
    _, logdet = np.linalg.slogdet(precision)
    log_evidence = p / 2 * np.log(alpha) + n / 2 * np.log(beta) - energy - logdet / 2 - n / 2 * np.log(2 * np.pi)
    return mean, covariance, precision, float(log_evidence)


def predictive(
    phi_new: FloatArray, mean: FloatArray, covariance: FloatArray, beta: float
) -> tuple[FloatArray, FloatArray]:
    """The predictive mean and variance of a new target at each row."""
    return phi_new @ mean, 1 / beta + np.einsum("ij,jk,ik->i", phi_new, covariance, phi_new)


def maximise_evidence(
    phi: FloatArray, y: FloatArray, alpha: float, beta: float, iterations: int
) -> tuple[float, float, float]:
    """MacKay's fixed point for a fixed number of updates: the final alpha, beta and the last gamma."""
    n = phi.shape[0]
    eigenvalues = np.linalg.eigvalsh(phi.T @ phi)
    gamma = 0.0
    for _ in range(iterations):
        mean, _, _, _ = posterior(phi, y, alpha, beta)
        gamma = float(np.sum(beta * eigenvalues / (alpha + beta * eigenvalues)))
        alpha = gamma / float(mean @ mean)
        beta = (n - gamma) / float(np.sum((y - phi @ mean) ** 2))
    return alpha, beta, gamma


def cases() -> list[Case]:
    g = rng(53)
    out: list[Case] = []
    for name, n, degree, alpha, beta in [
        ("cubic", 30, 3, 2.0, 25.0),
        ("quintic", 12, 5, 0.5, 100.0),
        ("line", 50, 1, 5.0, 4.0),
    ]:
        x = np.sort(g.uniform(-1, 1, size=n))
        y = np.sin(2.5 * x) + 0.2 * g.normal(size=n)
        phi = vandermonde(x, degree)
        test = vandermonde(np.linspace(-1.2, 1.2, 9), degree)
        mean, covariance, precision, log_evidence = posterior(phi, y, alpha, beta)
        pm, pv = predictive(test, mean, covariance, beta)
        out.append(
            case(
                f"posterior/{name}",
                "posterior",
                [phi, y, test],
                {
                    "mean": mean,
                    "covariance": covariance,
                    "precision": precision,
                    "logEvidence": np.array(log_evidence),
                    "predictiveMean": pm,
                    "predictiveVariance": pv,
                },
                rtol=1e-8,
                atol=1e-10,
                alpha=alpha,
                beta=beta,
            )
        )
        iterations = 40
        a, b, gamma = maximise_evidence(phi, y, alpha, beta, iterations)
        out.append(
            case(
                f"evidence/{name}",
                "evidence",
                [phi, y],
                {"alpha": np.array(a), "beta": np.array(b), "gamma": np.array(gamma)},
                rtol=1e-7,
                atol=1e-9,
                alpha=alpha,
                beta=beta,
                iterations=iterations,
            )
        )
    return out
