"""The penalty, barrier, augmented Lagrangian and projected gradient methods against scipy's solutions of the same
problems: SLSQP for the constrained ones (slsqp.py defines them) and L-BFGS-B for the box-constrained ones."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import scipy
import scipy.optimize as so
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.optim_problems import SMOOTH


def _slsqp() -> ModuleType:
    """The sibling case module with the constrained problems (case modules are loaded by path, not as a package)."""
    spec = importlib.util.spec_from_file_location("whitebox_cases.slsqp_problems", Path(__file__).with_name("slsqp.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load slsqp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    problems = _slsqp()
    for name in ["tutorial", "hs071", "rosen", "sphere", "disc"]:
        r = problems.solve(name)
        x0 = problems.PROBLEMS[name][4]
        bounds = problems.PROBLEMS[name][3]
        lower = [b[0] for b in bounds] if bounds is not None else []
        upper = [b[1] for b in bounds] if bounds is not None else []
        expected = np.concatenate([np.asarray(r.x, dtype=np.float64), [float(r.fun)]])
        for method, tol in [("augmentedLagrangian", 1e-6), ("penaltyMethod", 1e-4)]:
            out.append(
                case(
                    f"{method}/{name}",
                    method,
                    [np.array(x0), np.array(lower), np.array(upper)],
                    expected,
                    rtol=tol,
                    atol=tol,
                    problem=name,
                )
            )
        if name == "disc":
            out.append(
                case(
                    f"barrierMethod/{name}",
                    "barrierMethod",
                    [np.array(x0)],
                    expected,
                    rtol=1e-6,
                    atol=1e-7,
                    problem=name,
                )
            )
    # Box-constrained problems for projected gradients, against L-BFGS-B.
    for name, bounds in [
        ("rosen", [(-2.0, 0.5), (0.0, 2.0)]),
        ("beale", [(-4.5, 2.5), (-4.5, 0.4)]),
        ("spdQuadratic", [(0.0, np.inf)] * 3),
    ]:
        p = SMOOTH[name]
        r = so.minimize(p.f, p.x0, method="L-BFGS-B", jac=p.grad, bounds=bounds, options={"gtol": 1e-12, "ftol": 1e-15})
        out.append(
            case(
                f"projectedGradient/{name}",
                "projectedGradient",
                [p.x0, np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])],
                np.concatenate([np.asarray(r.x, dtype=np.float64), [float(r.fun)]]),
                rtol=1e-6,
                atol=1e-7,
                problem=name,
            )
        )
    return out
