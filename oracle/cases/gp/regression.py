"""Gaussian-process regression against scikit-learn: the log marginal likelihood and its gradient at fixed kernels,
posterior means, standard deviations and covariances, and the optimised fit."""

from typing import cast

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}

# scikit-learn lists a kernel's hyperparameters alphabetically by name (it reads them from `dir`), as the engine does,
# so theta and its gradient have the same order in both.

type Spec = dict[str, object]
"""A kernel as plain data: `k` is the kind, the rest its hyperparameters, `parts` the operands of a sum or product."""


def parts(spec: Spec) -> list[Spec]:
    value = spec["parts"]
    assert isinstance(value, list)
    return [part for part in value if isinstance(part, dict)]


def number(spec: Spec, name: str) -> float:
    value = spec[name]
    assert isinstance(value, int | float)
    return float(value)


def scales(spec: Spec) -> float | list[float]:
    value = spec["l"]
    if isinstance(value, list):
        return [float(v) for v in value]
    assert isinstance(value, int | float)
    return float(value)


def build(spec: Spec) -> Kernel:
    kind = spec["k"]
    if kind == "rbf":
        return RBF(length_scale=cast("float", scales(spec)))
    if kind == "matern":
        return Matern(length_scale=cast("float", scales(spec)), nu=number(spec, "nu"))
    if kind == "rq":
        return RationalQuadratic(length_scale=number(spec, "l"), alpha=number(spec, "alpha"))
    if kind == "periodic":
        return ExpSineSquared(length_scale=number(spec, "l"), periodicity=number(spec, "p"))
    if kind == "const":
        return ConstantKernel(constant_value=number(spec, "c"))
    if kind == "white":
        return WhiteKernel(noise_level=number(spec, "n"))
    if kind == "dot":
        return DotProduct(sigma_0=number(spec, "s0"))
    operands = [build(part) for part in parts(spec)]
    result = operands[0]
    for operand in operands[1:]:
        result = result + operand if kind == "sum" else result * operand
    return result


def rbf(length: float | list[float]) -> Spec:
    return {"k": "rbf", "l": length}


def kernel_cases() -> list[tuple[str, Spec]]:
    return [
        ("rbf", rbf(1.3)),
        ("rbf-ard", rbf([1.1, 2.2])),
        ("matern15", {"k": "matern", "l": 1.4, "nu": 1.5}),
        ("matern25", {"k": "matern", "l": 0.9, "nu": 2.5}),
        ("matern05", {"k": "matern", "l": 1.7, "nu": 0.5}),
        ("rq", {"k": "rq", "l": 1.2, "alpha": 0.8}),
        ("periodic", {"k": "periodic", "l": 1.1, "p": 2.5}),
        ("const-rbf", {"k": "prod", "parts": [{"k": "const", "c": 2.0}, rbf(1.5)]}),
        (
            "const-rbf-white",
            {
                "k": "sum",
                "parts": [{"k": "prod", "parts": [{"k": "const", "c": 2.0}, rbf(1.5)]}, {"k": "white", "n": 0.1}],
            },
        ),
        ("rbf-plus-dot", {"k": "sum", "parts": [rbf(0.8), {"k": "dot", "s0": 1.1}]}),
        ("periodic-times-rbf", {"k": "prod", "parts": [{"k": "periodic", "l": 1.0, "p": 3.0}, rbf(4.0)]}),
    ]


def dataset(g: np.random.Generator, ard: bool) -> tuple[FloatArray, FloatArray, FloatArray]:
    x = np.sort(g.uniform(-3.0, 3.0, size=(16, 1)), axis=0)
    xt = np.linspace(-3.5, 3.5, 7).reshape(-1, 1)
    if ard:
        x = g.uniform(-2.0, 2.0, size=(16, 2))
        xt = g.uniform(-2.5, 2.5, size=(7, 2))
        y = np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2 + 0.1 * g.normal(size=16)
    else:
        y = np.sin(1.5 * x[:, 0]) + 0.2 * x[:, 0] + 0.1 * g.normal(size=16)
    return x, y, xt


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(53)
    for name, spec in kernel_cases():
        ard = name == "rbf-ard"
        x, y, xt = dataset(g, ard)
        for normalize, alpha in [(False, 1e-2), (True, 0.1)]:
            # No optimiser: the kernel stays as given, and the likelihood is read at its own theta.
            model = GaussianProcessRegressor(
                kernel=build(spec), alpha=alpha, optimizer=cast("str", None), normalize_y=normalize
            )
            model.fit(x, y)
            theta = cast("Kernel", model.kernel_).theta
            value, gradient = cast("tuple[float, FloatArray]", model.log_marginal_likelihood(theta, eval_gradient=True))
            mean, std = model.predict(xt, return_std=True)
            _, cov = model.predict(xt, return_cov=True)
            output = {
                "lml": np.array(value),
                "gradient": np.asarray(gradient, dtype=np.float64),
                "mean": np.asarray(mean, dtype=np.float64),
                "std": np.asarray(std, dtype=np.float64),
                "cov": np.asarray(cov, dtype=np.float64),
            }
            out.append(
                case(
                    f"fixed/{name}/{'normalised' if normalize else 'raw'}",
                    "gpRegression",
                    [x, y, xt],
                    output,
                    rtol=1e-8,
                    atol=1e-9,
                    kernel=spec,
                    alpha=alpha,
                    normalizeY=normalize,
                    optimise=False,
                )
            )
    # The optimised fit, from a start near the optimum, so the two optimisers reach the same local maximum.
    x, y, xt = dataset(g, False)
    start: Spec = {
        "k": "sum",
        "parts": [{"k": "prod", "parts": [{"k": "const", "c": 1.0}, rbf(1.0)]}, {"k": "white", "n": 0.05}],
    }
    model = GaussianProcessRegressor(kernel=build(start), alpha=1e-10, n_restarts_optimizer=0, normalize_y=True)
    model.fit(x, y)
    mean, std = model.predict(xt, return_std=True)
    output = {
        "lml": np.array(model.log_marginal_likelihood_value_),
        "theta": np.asarray(cast("Kernel", model.kernel_).theta, dtype=np.float64),
        "mean": np.asarray(mean, dtype=np.float64),
        "std": np.asarray(std, dtype=np.float64),
    }
    out.append(
        case(
            "optimised/const-rbf-white",
            "gpRegression",
            [x, y, xt],
            output,
            rtol=1e-3,
            atol=1e-4,
            kernel=start,
            alpha=1e-10,
            normalizeY=True,
            optimise=True,
        )
    )
    return out
