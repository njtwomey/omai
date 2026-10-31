"""Kernel ridge regression against sklearn.kernel_ridge: dual coefficients and predictions for several kernels, one and
several outputs; the exact leave-one-out predictions against refitting n times with the same estimator."""

from typing import Any, cast

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from sklearn.kernel_ridge import KernelRidge

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def record(name: str, model: KernelRidge, x: FloatArray, y: FloatArray, xt: FloatArray, **params: object) -> Case:
    return case(
        name,
        "kernelRidge",
        [x, y, xt],
        {"dualCoef": np.asarray(model.dual_coef_, dtype=np.float64), "predict": model.predict(xt)},
        rtol=1e-8,
        atol=1e-9,
        **params,
    )


def loo(x: FloatArray, y: FloatArray, alpha: float, gamma: float) -> FloatArray:
    out = np.empty_like(y)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        model = KernelRidge(alpha=cast(Any, alpha), kernel="rbf", gamma=gamma).fit(x[keep], y[keep])
        out[i] = model.predict(x[i : i + 1])[0]
    return out


def cases() -> list[Case]:
    g = rng(404)
    x = g.uniform(-2, 2, size=(25, 2))
    y = np.sin(x[:, 0]) * np.cos(x[:, 1]) + 0.1 * g.normal(size=25)
    y2 = np.column_stack([y, x[:, 0] - x[:, 1] ** 2])
    xt = g.uniform(-2, 2, size=(10, 2))
    out: list[Case] = []
    configs: list[tuple[str, dict[str, float | str]]] = [
        ("linear", {"kernel": "linear", "alpha": 0.5}),
        ("rbf", {"kernel": "rbf", "alpha": 0.1, "gamma": 0.7}),
        ("poly", {"kernel": "poly", "alpha": 0.3, "gamma": 0.5, "degree": 3, "coef0": 1.0}),
        ("laplacian", {"kernel": "laplacian", "alpha": 0.2, "gamma": 0.4}),
        ("sigmoid", {"kernel": "sigmoid", "alpha": 1.0, "gamma": 0.2, "coef0": 0.5}),
        ("auto", {"kernel": "rbf", "alpha": 0.1}),
    ]
    for name, cfg in configs:
        params = {k: v for k, v in cfg.items()}
        if name == "auto":
            params["gamma"] = 1.0 / x.shape[1]
        model = KernelRidge(**cfg).fit(x, y)  # pyright: ignore[reportArgumentType]
        out.append(record(f"single/{name}", model, x, y, xt, **params))
        if name in ("rbf", "poly"):
            model2 = KernelRidge(**cfg).fit(x, y2)  # pyright: ignore[reportArgumentType]
            out.append(record(f"multi/{name}", model2, x, y2, xt, **params))
    loo_pred = loo(x, y, 0.1, 0.7)
    out.append(
        case(
            "loo/rbf",
            "looRidge",
            [x, y],
            {"predict": loo_pred},
            rtol=1e-8,
            atol=1e-9,
            alpha=0.1,
            gamma=0.7,
        )
    )
    return out
