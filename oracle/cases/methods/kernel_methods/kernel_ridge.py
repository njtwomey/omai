"""Kernel ridge regression against scikit-learn: dual coefficients and predictions with linear, RBF and polynomial
kernels."""

import numpy as np
import sklearn
from sklearn.kernel_ridge import KernelRidge
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(121)
    x = g.normal(size=(30, 2))
    y = np.sin(x[:, 0]) + x[:, 1] ** 2 + 0.1 * g.normal(size=30)
    q = g.normal(size=(5, 2))
    out: list[Case] = []
    for kernel, alpha, gamma in [("linear", 0.5, None), ("rbf", 0.1, 0.7), ("polynomial", 1.0, 0.5)]:
        kr = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma).fit(x, y)  # pyright: ignore[reportArgumentType]
        out.append(
            case(
                f"kernelRidge/{kernel}",
                "kernelRidge",
                [x, y, q],
                {"dual": kr.dual_coef_, "predict": kr.predict(q)},
                rtol=1e-9,
                atol=1e-10,
                kernel=kernel,
                alpha=alpha,
                gamma=-1.0 if gamma is None else gamma,
            )
        )
    return out
