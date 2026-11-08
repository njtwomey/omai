"""Total least squares and Deming regression against scipy.optimize: the orthogonal-distance objective with the latent
features eliminated, `r^2 / (sy^2 + sum beta_j^2 sx_j^2)` summed over the points, minimised by BFGS (scipy.odr, the
usual reference, is deprecated and returns wrong fits in this build); one and several features, equal and unequal
error scales."""

import numpy as np
import numpy.typing as npt
import scipy
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from scipy.optimize import minimize

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type Vec = npt.NDArray[np.float64]


def fit_profile(x: Vec, y: Vec, sx: Vec, sy: float) -> tuple[Vec, float]:
    """Minimise the orthogonal-distance objective with the latent features eliminated: for a hyperplane the squared
    distance of a point, in units of the errors, is r^2 / (sy^2 + sum_j beta_j^2 sx_j^2)."""
    n, d = x.shape

    def objective(theta: Vec) -> float:
        r = y - x @ theta[:d] - theta[d]
        return float(np.sum(r * r / (sy**2 + np.sum((theta[:d] * sx) ** 2))))

    ols = np.linalg.lstsq(np.column_stack([x, np.ones(n)]), y, rcond=None)[0]
    res = minimize(objective, ols, method="BFGS", options={"gtol": 1e-12, "maxiter": 10000})
    return np.asarray(res.x, dtype=np.float64), float(res.fun)


def cases() -> list[Case]:
    g = rng(1943)
    out: list[Case] = []
    for name, n, d, sx, sy in [
        ("one/equal", 80, 1, [0.5], 0.5),
        ("one/ratio4", 80, 1, [0.4], 0.8),
        ("two/mixed", 120, 2, [0.5, 0.3], 0.6),
    ]:
        truth = g.normal(size=(n, d))
        x = truth + np.array(sx) * g.normal(size=(n, d))
        y = truth @ np.linspace(2.0, -1.0, d) + 1.0 + sy * g.normal(size=n)
        beta, _ = fit_profile(x, y, np.array(sx), sy)
        out.append(
            case(
                f"tls/{name}",
                "totalLeastSquares",
                [x, y, np.array(sx)],
                {"coef": beta[:d], "intercept": np.array(float(beta[d]))},
                rtol=1e-5,
                atol=1e-6,
                errorY=sy,
            )
        )
    return out
