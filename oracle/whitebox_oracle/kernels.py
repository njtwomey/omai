"""Shared pieces for the kernels case modules: scikit-learn Gaussian-process kernels evaluated as a user would (Gram
matrix, cross-covariance, diagonal and the gradient in log-hyperparameters from `eval_gradient=True`), and GPyTorch
kernels evaluated in float64."""

from collections.abc import Callable
from typing import cast

import numpy as np
from sklearn.gaussian_process.kernels import Kernel

from .fixture import Case, case
from .helpers import FloatArray


def gp_case(
    name: str,
    reference: Kernel,
    x: FloatArray,
    y: FloatArray,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    **params: object,
) -> Case:
    """A case with K(X), K(X, Y), diag(X) and dK(X)/dtheta of a scikit-learn kernel, its hyperparameter names in theta
    order (`names`) and theta itself (`theta`, log-transformed values)."""
    gram, gradient = cast(tuple[FloatArray, FloatArray], reference(x, eval_gradient=True))
    names = [h.name for h in reference.hyperparameters if not h.fixed]
    return case(
        name,
        "gp",
        [x, y],
        {
            "gram": np.asarray(gram, dtype=np.float64),
            "cross": np.asarray(reference(x, y), dtype=np.float64),
            "diag": np.asarray(reference.diag(x), dtype=np.float64),
            "gradient": np.asarray(gradient, dtype=np.float64),
        },
        rtol=rtol,
        atol=atol,
        names=names,
        theta=[float(v) for v in reference.theta],
        **params,
    )


def call(f: Callable[..., object], *args: object, **kwargs: object) -> FloatArray:
    """A scikit-learn function called as a user would, its result as a float64 array (its stubs annotate some float
    parameters as int)."""
    return np.asarray(f(*args, **kwargs), dtype=np.float64)


def anisotropic(values: list[float]) -> float:
    """Per-dimension length scales for a scikit-learn kernel, which accepts an array where its stubs say float."""
    return cast(float, np.asarray(values, dtype=np.float64))
