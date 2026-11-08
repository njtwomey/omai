"""Stochastic gradient descent and passive-aggressive linear models against scikit-learn, without shuffling: the
coefficients and intercepts after a few epochs, for several losses, penalties and schedules."""

import warnings
from typing import Any

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier, SGDRegressor

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}

TOL = {"rtol": 1e-9, "atol": 1e-12}


def classification(k: int) -> tuple[FloatArray, FloatArray]:
    g = rng(31 + k)
    y = (np.arange(45) % k).astype(np.float64)
    x = g.normal(size=(45, 3)) + y[:, None] * np.array([1.0, -0.7, 0.4])
    return x, y


def regression() -> tuple[FloatArray, FloatArray]:
    g = rng(37)
    x = g.normal(size=(40, 3))
    y = x @ np.array([1.5, -2.0, 0.5]) + 0.3 + 0.1 * g.normal(size=40)
    return x, y


def classifier_case(name: str, k: int, options: dict[str, Any], rtol: float = 1e-9) -> Case:
    x, y = classification(k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = SGDClassifier(shuffle=False, **options).fit(x, y)
    output = {"coef": np.asarray(m.coef_), "intercept": np.asarray(m.intercept_)}
    # The cumulative L1 penalty clips weights at zero, which amplifies rounding (scikit-learn scales lazily).
    return case(name, "classifier", [x, y], output, rtol=rtol, atol=1e-12, **options)


def regressor_case(name: str, options: dict[str, Any]) -> Case:
    x, y = regression()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = SGDRegressor(shuffle=False, **options).fit(x, y)
    output = {"coef": np.asarray(m.coef_), "intercept": np.asarray(m.intercept_)}
    return case(name, "regressor", [x, y], output, **TOL, **options)


def cases() -> list[Case]:
    return [
        classifier_case("hinge/binary", 2, {"loss": "hinge", "max_iter": 5, "tol": None}),
        classifier_case("hinge/multiclass", 3, {"loss": "hinge", "max_iter": 5, "tol": None}),
        classifier_case(
            "log/elasticnet",
            2,
            {"loss": "log_loss", "penalty": "elasticnet", "alpha": 0.01, "max_iter": 8, "tol": None},
        ),
        classifier_case(
            "modified-huber/l1",
            3,
            {"loss": "modified_huber", "penalty": "l1", "alpha": 0.001, "max_iter": 6, "tol": None},
            rtol=1e-7,
        ),
        classifier_case("hinge/stopping", 2, {"loss": "hinge", "max_iter": 200, "tol": 1e-3}),
        classifier_case(
            "perceptron/constant",
            2,
            {"loss": "perceptron", "penalty": None, "learning_rate": "constant", "eta0": 1.0, "max_iter": 6},
        ),
        classifier_case(
            "pa1",
            3,
            {"loss": "hinge", "penalty": None, "learning_rate": "pa1", "eta0": 0.5, "max_iter": 5, "tol": None},
        ),
        classifier_case(
            "pa2",
            2,
            {"loss": "hinge", "penalty": None, "learning_rate": "pa2", "eta0": 1.0, "max_iter": 5, "tol": None},
        ),
        regressor_case("squared/invscaling", {"loss": "squared_error", "max_iter": 10, "tol": None}),
        regressor_case("huber/adaptive", {"loss": "huber", "learning_rate": "adaptive", "eta0": 0.05, "max_iter": 40}),
        regressor_case(
            "pa1/regressor",
            {
                "loss": "epsilon_insensitive",
                "penalty": None,
                "learning_rate": "pa1",
                "eta0": 1.0,
                "max_iter": 5,
                "tol": None,
            },
        ),
    ]
