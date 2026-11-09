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

TIGHT = {"rtol": 1e-6, "atol": 1e-7}
LOOSE = {"rtol": 1e-4, "atol": 1e-5}


def blobs(n: int, d: int, k: int, seed: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """k overlapping Gaussian classes, so that there are margin violators."""
    g = rng(seed)
    centres = g.normal(0, 1.5, (k, d))
    y = np.arange(n) % k
    x = centres[y] + g.normal(0, 1.2, (n, d))
    return x, y.astype(np.float64)


def classification_cases(
    name: str, x: NDArray[np.float64], y: NDArray[np.float64], model: LinearSVC, **knobs: object
) -> list[Case]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, y)
    tol = LOOSE if knobs.get("penalty") == "l1" else TIGHT
    inputs = [x, y]
    return [
        case(f"{name}/coef", "coef", inputs, np.asarray(model.coef_), **tol, **knobs),
        case(f"{name}/intercept", "intercept", inputs, np.asarray(model.intercept_), **tol, **knobs),
        case(f"{name}/decision", "decision", inputs, np.asarray(model.decision_function(x)), **tol, **knobs),
    ]


def cases() -> list[Case]:
    out: list[Case] = []
    x2, y2 = blobs(40, 4, 2, 11)
    x3, y3 = blobs(45, 4, 3, 12)
    solver: dict[str, Any] = {"tol": 1e-10, "max_iter": 200000, "random_state": 0}
    for c in [0.1, 1.0, 10.0]:
        knobs = {"C": c, "loss": "squared_hinge", "penalty": "l2", "fit_intercept": True, "intercept_scaling": 1.0}
        out += classification_cases(f"svc/squared_hinge/C{c}", x2, y2, LinearSVC(C=c, dual=False, **solver), **knobs)
    knobs = {"C": 1.0, "loss": "squared_hinge", "penalty": "l2", "fit_intercept": True, "intercept_scaling": 1.0}
    out += classification_cases("svc/squared_hinge/dual", x2, y2, LinearSVC(C=1.0, dual=True, **solver), **knobs)
    knobs = {"C": 1.0, "loss": "squared_hinge", "penalty": "l2", "fit_intercept": False, "intercept_scaling": 1.0}
    out += classification_cases(
        "svc/squared_hinge/no_intercept", x2, y2, LinearSVC(C=1.0, fit_intercept=False, dual=False, **solver), **knobs
    )
    knobs = {"C": 1.0, "loss": "squared_hinge", "penalty": "l2", "fit_intercept": True, "intercept_scaling": 5.0}
    out += classification_cases(
        "svc/squared_hinge/scaling", x2, y2, LinearSVC(C=1.0, intercept_scaling=5.0, dual=False, **solver), **knobs
    )
    for c in [0.3, 3.0]:
        knobs = {"C": c, "loss": "hinge", "penalty": "l2", "fit_intercept": True, "intercept_scaling": 1.0}
        out += classification_cases(
            f"svc/hinge/C{c}", x2, y2, LinearSVC(C=c, loss="hinge", dual=True, **solver), **knobs
        )
    knobs = {"C": 1.0, "loss": "squared_hinge", "penalty": "l1", "fit_intercept": True, "intercept_scaling": 1.0}
    out += classification_cases(
        "svc/l1", x2, y2, LinearSVC(C=0.5, penalty="l1", dual=False, **solver), **{**knobs, "C": 0.5}
    )
    knobs = {"C": 1.0, "loss": "squared_hinge", "penalty": "l2", "fit_intercept": True, "intercept_scaling": 1.0}
    out += classification_cases("svc/multiclass", x3, y3, LinearSVC(C=1.0, dual=False, **solver), **knobs)
    out += classification_cases(
        "svc/multiclass_hinge",
        x3,
        y3,
        LinearSVC(C=1.0, loss="hinge", dual=True, **solver),
        **{**knobs, "loss": "hinge"},
    )
    # Class weights: balanced, on an imbalanced problem.
    keep = np.r_[np.arange(0, 40, 2)[:10], np.arange(1, 40, 2)]
    xb, yb = x2[keep], y2[keep]
    out += classification_cases(
        "svc/balanced",
        xb,
        yb,
        LinearSVC(C=1.0, class_weight="balanced", dual=False, **solver),
        **{**knobs, "class_weight": "balanced"},
    )
    return out + sample_weight_and_regression(x2, y2, solver)


def sample_weight_and_regression(
    x2: NDArray[np.float64], y2: NDArray[np.float64], solver: dict[str, Any]
) -> list[Case]:
    out: list[Case] = []
    g = rng(13)
    w = g.uniform(0.2, 2.0, len(y2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        weighted = LinearSVC(C=1.0, dual=False, **solver).fit(x2, y2, sample_weight=w)
    out.append(case("svc/sample_weight/coef", "coefWeighted", [x2, y2, w], np.asarray(weighted.coef_), **TIGHT, C=1.0))
    out.append(
        case(
            "svc/sample_weight/intercept",
            "interceptWeighted",
            [x2, y2, w],
            np.asarray(weighted.intercept_),
            **TIGHT,
            C=1.0,
        )
    )

    xr = g.normal(0, 1, (50, 3))
    yr = xr @ np.array([1.5, -2.0, 0.5]) + 0.7 + g.normal(0, 0.4, 50)
    for loss in ["epsilon_insensitive", "squared_epsilon_insensitive"]:
        for eps, c in [(0.0, 1.0), (0.2, 10.0)]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = LinearSVR(C=c, epsilon=eps, loss=loss, dual=True, **solver).fit(xr, yr)
            name = f"svr/{loss}/eps{eps}-C{c}"
            knobs = {"C": c, "epsilon": eps, "loss": loss}
            out.append(case(f"{name}/coef", "svrCoef", [xr, yr], np.asarray(model.coef_), **TIGHT, **knobs))
            out.append(
                case(f"{name}/intercept", "svrIntercept", [xr, yr], np.asarray(model.intercept_), **TIGHT, **knobs)
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        primal = LinearSVR(C=1.0, epsilon=0.1, loss="squared_epsilon_insensitive", dual=False, **solver).fit(xr, yr)
    knobs = {"C": 1.0, "epsilon": 0.1, "loss": "squared_epsilon_insensitive"}
    out.append(case("svr/primal/coef", "svrCoef", [xr, yr], np.asarray(primal.coef_), **TIGHT, **knobs))
    out.append(case("svr/primal/intercept", "svrIntercept", [xr, yr], np.asarray(primal.intercept_), **TIGHT, **knobs))
    return out
