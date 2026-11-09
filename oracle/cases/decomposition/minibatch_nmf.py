"""Mini-batch NMF against sklearn.decomposition.MiniBatchNMF from fixed initial factors (`init='custom'`, so nothing is
random: the batches are taken in order): the components, the weights of the training rows, the number of steps and
passes, the reconstruction error and the transform of held-out rows, for the Frobenius, Kullback-Leibler,
Itakura-Saito and an intermediate beta divergence, with penalties, fresh restarts, and each stopping rule."""

import warnings

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.decomposition import MiniBatchNMF
from sklearn.exceptions import ConvergenceWarning

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def parts(seed: int, n: int, d: int, k: int) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    g = rng(seed)
    h_true = g.gamma(1.5, size=(k, d))
    w_true = g.gamma(1.0, size=(n, k))
    x = w_true @ h_true + 0.05 * g.uniform(size=(n, d)) + 0.01
    w0 = g.uniform(0.2, 1.5, size=(n, k))
    h0 = g.uniform(0.2, 1.5, size=(k, d))
    held = g.gamma(1.0, size=(5, k)) @ h_true + 0.02
    return x, w0, h0, held


def cases() -> list[Case]:
    out: list[Case] = []
    configs: list[tuple[str, dict[str, object]]] = [
        ("frobenius/one-batch", {"batch_size": 40, "max_iter": 25, "tol": 0.0, "max_no_improvement": None}),
        ("frobenius/batches", {"batch_size": 12, "max_iter": 8, "tol": 0.0, "max_no_improvement": None}),
        ("frobenius/uneven", {"batch_size": 17, "max_iter": 6, "tol": 0.0, "max_no_improvement": None}),
        (
            "kl",
            {"batch_size": 10, "max_iter": 6, "tol": 0.0, "max_no_improvement": None, "beta_loss": "kullback-leibler"},
        ),
        ("is", {"batch_size": 20, "max_iter": 5, "tol": 0.0, "max_no_improvement": None, "beta_loss": "itakura-saito"}),
        ("beta1.5", {"batch_size": 20, "max_iter": 5, "tol": 0.0, "max_no_improvement": None, "beta_loss": 1.5}),
        (
            "regularised",
            {
                "batch_size": 10,
                "max_iter": 5,
                "tol": 0.0,
                "max_no_improvement": None,
                "alpha_W": 0.01,
                "alpha_H": 0.02,
                "l1_ratio": 0.4,
            },
        ),
        (
            "fresh-restarts",
            {
                "batch_size": 10,
                "max_iter": 5,
                "tol": 0.0,
                "max_no_improvement": None,
                "fresh_restarts": True,
                "fresh_restarts_max_iter": 7,
            },
        ),
        ("forget", {"batch_size": 10, "max_iter": 6, "tol": 0.0, "max_no_improvement": None, "forget_factor": 0.4}),
        ("tol-stop", {"batch_size": 10, "max_iter": 300, "tol": 1e-3, "max_no_improvement": None}),
        ("no-improvement-stop", {"batch_size": 8, "max_iter": 300, "tol": 0.0, "max_no_improvement": 3}),
    ]
    for seed, (name, options) in enumerate(configs, start=1):
        x, w0, h0, held = parts(seed, 40, 9, 3)
        model = MiniBatchNMF(n_components=3, init="custom", **options)  # pyright: ignore[reportArgumentType]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            w = model.fit_transform(x, W=w0.copy(), H=h0.copy())
            transformed = model.transform(held)
        beta = options.get("beta_loss", "frobenius")
        out.append(
            case(
                name,
                "minibatch",
                [x, w0, h0, held],
                {
                    "components": model.components_,
                    "weights": w,
                    "nSteps": np.array(float(model.n_steps_)),
                    "nIter": np.array(float(model.n_iter_)),
                    "error": np.array(float(model.reconstruction_err_)),
                    "transform": transformed,
                },
                rtol=1e-7,
                atol=1e-9,
                betaLoss=beta if isinstance(beta, str) else float(beta),
                batchSize=int(options["batch_size"]),  # pyright: ignore[reportArgumentType]
                maxIter=int(options["max_iter"]),  # pyright: ignore[reportArgumentType]
                tol=float(options["tol"]),  # pyright: ignore[reportArgumentType]
                maxNoImprovement=-1 if options["max_no_improvement"] is None else int(options["max_no_improvement"]),  # pyright: ignore[reportArgumentType]
                alphaW=float(options.get("alpha_W", 0.0)),  # pyright: ignore[reportArgumentType]
                alphaH=float(options.get("alpha_H", options.get("alpha_W", 0.0))),  # pyright: ignore[reportArgumentType]
                l1Ratio=float(options.get("l1_ratio", 0.0)),  # pyright: ignore[reportArgumentType]
                forgetFactor=float(options.get("forget_factor", 0.7)),  # pyright: ignore[reportArgumentType]
                freshRestarts=bool(options.get("fresh_restarts", False)),
                freshRestartsMaxIter=int(options.get("fresh_restarts_max_iter", 30)),  # pyright: ignore[reportArgumentType]
            )
        )
    return out
