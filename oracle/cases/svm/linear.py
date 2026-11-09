"""Linear support vector machines against scikit-learn's liblinear models: the weights and intercepts of LinearSVC
(squared hinge and hinge, L2 and L1 penalties, class and sample weights, intercept scaling, one-versus-rest) and
LinearSVR (epsilon-insensitive and its square). The L2 problems are strictly convex, so the weights are unique and are
compared at 1e-6 from solvers run to a tolerance of 1e-10; the L1 problem is compared at 1e-4."""

import warnings
from importlib.metadata import version
from typing import Any

import numpy as np
from numpy.typing import NDArray
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC, LinearSVR

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

SOLVER: dict[str, Any] = {"tol": 1e-10, "max_iter": 200000, "random_state": 0}


def blobs(n: int, d: int, k: int, seed: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """k overlapping Gaussian classes, so that there are margin violators."""
    g = rng(seed)
    centres = g.normal(0, 1.5, (k, d))
    y = np.arange(n) % k
    x = centres[y] + g.normal(0, 1.2, (n, d))
    return x, y.astype(np.float64)


def svc(c: float, dual: bool, **extra: Any) -> LinearSVC:  # noqa: ANN401
    """A LinearSVC with the solver settings and the given penalty and dual choice."""
    options: dict[str, Any] = {"C": c, "dual": dual, **SOLVER, **extra}
    return LinearSVC(**options)


def classification_cases(
    name: str,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    model: LinearSVC,
    params: dict[str, object],
    loose: bool = False,
) -> list[Case]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, y)
    rtol, atol = (1e-4, 1e-5) if loose else (1e-6, 1e-7)
    inputs = [x, y]
    return [
        case(f"{name}/coef", "coef", inputs, np.asarray(model.coef_), rtol=rtol, atol=atol, **params),
        case(f"{name}/intercept", "intercept", inputs, np.asarray(model.intercept_), rtol=rtol, atol=atol, **params),
        case(
            f"{name}/decision",
            "decision",
            inputs,
            np.asarray(model.decision_function(x)),
            rtol=rtol,
            atol=atol,
            **params,
        ),
    ]


def knobs(**changes: object) -> dict[str, object]:
    """The options the engine reads, from the defaults of LinearSVC."""
    base: dict[str, object] = {
        "C": 1.0,
        "loss": "squared_hinge",
        "penalty": "l2",
        "fit_intercept": True,
        "intercept_scaling": 1.0,
    }
    return {**base, **changes}


def cases() -> list[Case]:
    out: list[Case] = []
    x2, y2 = blobs(40, 4, 2, 11)
    x3, y3 = blobs(45, 4, 3, 12)
    for c in [0.1, 1.0, 10.0]:
        out += classification_cases(f"svc/squared_hinge/C{c}", x2, y2, svc(c, False), knobs(C=c))
    out += classification_cases("svc/squared_hinge/dual", x2, y2, svc(1.0, True), knobs())
    out += classification_cases(
        "svc/squared_hinge/no_intercept", x2, y2, svc(1.0, False, fit_intercept=False), knobs(fit_intercept=False)
    )
    out += classification_cases(
        "svc/squared_hinge/scaling", x2, y2, svc(1.0, False, intercept_scaling=5.0), knobs(intercept_scaling=5.0)
    )
    for c in [0.3, 3.0]:
        out += classification_cases(f"svc/hinge/C{c}", x2, y2, svc(c, True, loss="hinge"), knobs(C=c, loss="hinge"))
    out += classification_cases("svc/l1", x2, y2, svc(0.5, False, penalty="l1"), knobs(C=0.5, penalty="l1"), loose=True)
    out += classification_cases("svc/multiclass", x3, y3, svc(1.0, False), knobs())
    out += classification_cases("svc/multiclass_hinge", x3, y3, svc(1.0, True, loss="hinge"), knobs(loss="hinge"))
    # Class weights: balanced, on an imbalanced problem.
    keep = np.r_[np.arange(0, 40, 2)[:10], np.arange(1, 40, 2)]
    out += classification_cases(
        "svc/balanced",
        x2[keep],
        y2[keep],
        svc(1.0, False, class_weight="balanced"),
        knobs(class_weight="balanced"),
    )
    return out + sample_weight_and_regression(x2, y2)


def sample_weight_and_regression(x2: NDArray[np.float64], y2: NDArray[np.float64]) -> list[Case]:
    out: list[Case] = []
    g = rng(13)
    w = g.uniform(0.2, 2.0, len(y2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        weighted = svc(1.0, False).fit(x2, y2, sample_weight=w)
    params = knobs()
    inputs = [x2, y2, w]
    out.append(
        case(
            "svc/sample_weight/coef", "coefWeighted", inputs, np.asarray(weighted.coef_), rtol=1e-6, atol=1e-7, **params
        )
    )
    out.append(
        case(
            "svc/sample_weight/intercept",
            "interceptWeighted",
            inputs,
            np.asarray(weighted.intercept_),
            rtol=1e-6,
            atol=1e-7,
            **params,
        )
    )

    xr = g.normal(0, 1, (50, 3))
    yr = xr @ np.array([1.5, -2.0, 0.5]) + 0.7 + g.normal(0, 0.4, 50)

    def svr(c: float, eps: float, loss: str, dual: bool) -> LinearSVR:
        options: dict[str, Any] = {"C": c, "epsilon": eps, "loss": loss, "dual": dual, **SOLVER}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return LinearSVR(**options).fit(xr, yr)

    def svr_cases(name: str, model: LinearSVR, c: float, eps: float, loss: str) -> list[Case]:
        p: dict[str, object] = {"C": c, "epsilon": eps, "loss": loss}
        return [
            case(f"{name}/coef", "svrCoef", [xr, yr], np.asarray(model.coef_), rtol=1e-6, atol=1e-7, **p),
            case(
                f"{name}/intercept", "svrIntercept", [xr, yr], np.asarray(model.intercept_), rtol=1e-6, atol=1e-7, **p
            ),
        ]

    for loss in ["epsilon_insensitive", "squared_epsilon_insensitive"]:
        for eps, c in [(0.0, 1.0), (0.2, 10.0)]:
            out += svr_cases(f"svr/{loss}/eps{eps}-C{c}", svr(c, eps, loss, True), c, eps, loss)
    out += svr_cases(
        "svr/primal", svr(1.0, 0.1, "squared_epsilon_insensitive", False), 1.0, 0.1, "squared_epsilon_insensitive"
    )
    return out
