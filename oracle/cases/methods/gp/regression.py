"""Gaussian-process regression against scikit-learn: the fitted log marginal likelihood and the posterior mean and
standard deviation, with fixed and optimised ConstantKernel * RBF kernels (one and two features)."""

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(101)
    x1 = np.sort(g.uniform(-3, 3, size=(20, 1)), axis=0)
    y1 = np.sin(x1[:, 0]) + 0.1 * g.normal(size=20)
    q1 = np.linspace(-3.5, 3.5, 7)[:, None]
    x2 = g.uniform(-2, 2, size=(25, 2))
    y2 = np.cos(x2[:, 0]) * x2[:, 1] + 0.05 * g.normal(size=25)
    q2 = g.uniform(-2, 2, size=(5, 2))
    out: list[Case] = []
    for name, x, y, q, optimise in [
        ("fixed", x1, y1, q1, False),
        ("fit1d", x1, y1, q1, True),
        ("fit2d", x2, y2, q2, True),
    ]:
        gp = GaussianProcessRegressor(
            ConstantKernel(1.0) * RBF(1.0),
            alpha=0.01,
            optimizer="fmin_l_bfgs_b" if optimise else None,  # pyright: ignore[reportArgumentType]
        ).fit(x, y)
        mean, std = gp.predict(q, return_std=True)
        out.append(
            case(
                f"gpr/{name}",
                "gpr",
                [x, y, q],
                {"lml": np.array(gp.log_marginal_likelihood_value_), "mean": mean, "std": std},
                rtol=1e-5,
                atol=1e-6,
                optimize=optimise,
            )
        )
    return out
