"""Bayesian-optimisation pieces against closed forms and reference implementations:
- the Gaussian-process posterior with fixed hyperparameters against scikit-learn's GaussianProcessRegressor
  (ConstantKernel * Matern(nu=2.5) with per-dimension length scales, alpha = noise, optimizer=None), and with a
  categorical (Hamming-distance) column against Optuna's GPRegressor;
- expected improvement, probability of improvement and confidence bounds against their closed forms with scipy's
  normal distribution, and log expected improvement against Optuna's `logei` (including z < -25);
- the MAP fit of the surrogate's hyperparameters, compared by the value of the objective it minimises (negative log
  marginal likelihood plus Optuna's log prior) at Optuna's optimum: an invariant, since the optimum is iterative."""

import math

import numpy as np
import optuna
import scipy
import sklearn
import torch
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from optuna._gp.acqf import logei  # pyright: ignore[reportPrivateUsage]
from optuna._gp.gp import GPRegressor, fit_kernel_params  # pyright: ignore[reportPrivateUsage]
from optuna._gp.prior import default_log_prior  # pyright: ignore[reportPrivateUsage]
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

REFERENCE: dict[str, str] = {
    "optuna": optuna.__version__,
    "scikit-learn": sklearn.__version__,
    "scipy": scipy.__version__,
    "torch": torch.__version__,
}


def sklearn_posterior(name: str, seed: int, n: int, d: int, scales: list[float], scale: float, noise: float) -> Case:
    g = rng(seed)
    x = g.uniform(0, 1, size=(n, d))
    y = np.sin(6 * x[:, 0]) + (x[:, 1] ** 2 if d > 1 else 0.0) + g.normal(0, 0.05, size=n)
    xs = g.uniform(0, 1, size=(9, d))
    # scikit-learn annotates length_scale as a float and optimizer as a str; arrays and None are documented inputs.
    kernel = ConstantKernel(scale, "fixed") * Matern(length_scale=scales, length_scale_bounds="fixed", nu=2.5)  # pyright: ignore[reportArgumentType]
    gp = GaussianProcessRegressor(kernel=kernel, alpha=noise, optimizer=None, normalize_y=False)  # pyright: ignore[reportArgumentType]
    gp.fit(x, y)
    mean, std = gp.predict(xs, return_std=True)
    out = {
        "mean": np.asarray(mean, dtype=np.float64),
        "std": np.asarray(std, dtype=np.float64),
        "lml": float(gp.log_marginal_likelihood_value_),
    }
    return case(
        name, "posterior", [x, y, xs], out, rtol=1e-8, atol=1e-10, lengthScales=scales, kernelScale=scale, noise=noise
    )


def optuna_posterior(name: str, seed: int) -> Case:
    g = rng(seed)
    n = 12
    x = np.stack([g.uniform(0, 1, size=n), g.integers(0, 3, size=n).astype(np.float64)], axis=1)
    y = np.cos(4 * x[:, 0]) + 0.5 * x[:, 1]
    xs = np.stack([g.uniform(0, 1, size=7), g.integers(0, 3, size=7).astype(np.float64)], axis=1)
    inv_sq = torch.tensor([1 / 0.4**2, 1 / 1.3**2], dtype=torch.float64)
    gpr = GPRegressor(
        is_categorical=torch.tensor([False, True]),
        X_train=torch.from_numpy(x),
        y_train=torch.from_numpy(y),
        inverse_squared_lengthscales=inv_sq,
        kernel_scale=torch.tensor(0.8, dtype=torch.float64),
        noise_var=torch.tensor(1e-3, dtype=torch.float64),
    )
    gpr._cache_matrix()  # pyright: ignore[reportPrivateUsage]
    mean, var = gpr.posterior(torch.from_numpy(xs))
    out = {"mean": mean.detach().numpy(), "std": np.sqrt(var.detach().numpy())}
    return case(
        name,
        "posteriorCategorical",
        [x, y, xs],
        out,
        rtol=1e-9,
        atol=1e-10,
        lengthScales=[0.4, 1.3],
        kernelScale=0.8,
        noise=1e-3,
    )


def acquisitions() -> list[Case]:
    g = rng(5)
    mean = np.concatenate([g.normal(0, 1, size=12), [0.0, 3.0, 40.0, 60.0]])
    std = np.concatenate([g.uniform(0.05, 2.0, size=12), [1.0, 0.1, 1.0, 1.5]])
    best = -0.3
    z = (best - mean) / std
    ei = (best - mean) * norm.cdf(z) + std * norm.pdf(z)
    pi = norm.cdf(z)
    log_pi = norm.logcdf(z)
    # Optuna maximises: its logei(mean', var, f0) is log E[max(0, y' - f0)], so negate both to minimise.
    log_ei = logei(torch.from_numpy(-mean), torch.from_numpy(std**2), -best).numpy()
    return [
        # The closed form cancels in the far tail (z = -33 loses about three digits), in the reference as here.
        case("acquisition/ei", "ei", [mean, std], np.asarray(ei), rtol=1e-8, atol=1e-300, best=best),
        case("acquisition/pi", "pi", [mean, std], np.asarray(pi), rtol=1e-10, atol=1e-300, best=best),
        case("acquisition/logpi", "logPi", [mean, std], np.asarray(log_pi), rtol=1e-10, atol=1e-12, best=best),
        case("acquisition/logei", "logEi", [mean, std], np.asarray(log_ei), rtol=1e-9, atol=1e-12, best=best),
        case("acquisition/lcb", "lcb", [mean, std], mean - 2.0 * std, best=best),
        case("acquisition/ucb", "ucb", [mean, std], mean + 1.5 * std, best=best, beta=1.5),
    ]


def map_case(name: str, seed: int, n: int, categorical: list[bool]) -> Case:
    g = rng(seed)
    cols = [g.integers(0, 3, size=n).astype(np.float64) if c else g.uniform(0, 1, size=n) for c in categorical]
    x = np.stack(cols, axis=1)
    y = np.sin(5 * x[:, 0]) + 0.3 * x[:, -1] + g.normal(0, 0.05, size=n)
    yn = (y - y.mean()) / y.std()
    gpr = fit_kernel_params(
        x, yn, np.asarray(categorical), default_log_prior, minimum_noise=1e-6, deterministic_objective=False
    )
    loss = float(-(gpr.marginal_log_likelihood() + default_log_prior(gpr)).item()) + n * 0.5 * math.log(2 * math.pi)
    return case(name, "fit", [x, y], loss, rtol=1e-4, atol=1e-6, categorical=categorical)


def cases() -> list[Case]:
    return [
        sklearn_posterior("posterior/1d", 0, 6, 1, [0.3], 1.0, 1e-6),
        sklearn_posterior("posterior/2d", 1, 15, 2, [0.25, 0.8], 1.7, 1e-4),
        optuna_posterior("posterior/categorical", 2),
        *acquisitions(),
        map_case("fit/continuous", 3, 20, [False, False]),
        map_case("fit/mixed", 4, 25, [False, True, False]),
    ]
