"""Automatic relevance determination against sklearn.linear_model.ARDRegression: the weights, intercept, noise and weight
precisions, the posterior covariance of the kept weights, the iteration count, the last objective, and the predictive
mean and standard deviation, for more rows than features and for fewer (the Woodbury path), with and without an
intercept and with a prune threshold that removes irrelevant features."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.linear_model import ARDRegression

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def sparse_problem(seed: int, n: int, d: int, relevant: int, noise: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    g = rng(seed)
    x = g.normal(size=(n, d)) + np.linspace(0.0, 1.0, d)
    w = np.zeros(d)
    w[:relevant] = g.uniform(1.0, 3.0, size=relevant) * g.choice([-1.0, 1.0], size=relevant)
    y = x @ w + 0.7 + noise * g.normal(size=n)
    held = g.normal(size=(7, d)) + np.linspace(0.0, 1.0, d)
    return x, y, held


def cases() -> list[Case]:
    out: list[Case] = []
    for name, seed, n, d, relevant, fit_intercept, noise in [
        ("tall", 1, 60, 8, 3, True, 0.3),
        ("tall/no-intercept", 2, 60, 8, 3, False, 0.3),
        ("wide", 3, 25, 40, 3, True, 0.2),
        ("dense", 4, 80, 4, 4, True, 0.5),
        ("noisy", 5, 50, 10, 2, True, 1.5),
    ]:
        x, y, held = sparse_problem(seed, n, d, relevant, noise)
        model = ARDRegression(fit_intercept=fit_intercept, compute_score=True, tol=1e-6, max_iter=500).fit(x, y)
        mean, std = model.predict(held, return_std=True)
        keep = model.lambda_ < model.threshold_lambda
        out.append(
            case(
                name,
                "ard",
                [x, y, held],
                {
                    "coef": model.coef_,
                    "intercept": np.array(float(model.intercept_)),
                    "alpha": np.array(float(model.alpha_)),
                    "lambda": model.lambda_,
                    "sigma": model.sigma_,
                    "keep": keep.astype(np.float64),
                    "nIter": np.array(float(model.n_iter_)),
                    "score": np.array(float(model.scores_[-1])),
                    "predict": mean,
                    "std": std,
                },
                rtol=1e-6,
                atol=1e-8,
                fitIntercept=fit_intercept,
            )
        )
    return out
