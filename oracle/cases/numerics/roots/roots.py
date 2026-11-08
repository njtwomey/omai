"""Root finding against scipy.optimize, on a shared table of test functions (the TypeScript test defines the same)."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import scipy
import scipy.optimize as so
from omai_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type Scalar = Callable[[float], float]

# name -> (f, f', f'', bracket, start)
FUNCTIONS: dict[str, tuple[Scalar, Scalar, Scalar, tuple[float, float], float]] = {
    "cubic": (lambda x: x**3 - x - 1, lambda x: 3 * x**2 - 1, lambda x: 6 * x, (1.0, 2.0), 1.5),
    "cos": (np.cos, lambda x: -np.sin(x), lambda x: -np.cos(x), (0.0, 2.0), 1.0),
    "exp": (lambda x: np.exp(x) - 5, np.exp, np.exp, (0.0, 3.0), 1.0),
    "kepler": (
        lambda x: x - 0.8 * np.sin(x) - 1.2,
        lambda x: 1 - 0.8 * np.cos(x),
        lambda x: 0.8 * np.sin(x),
        (0.0, 4.0),
        1.2,
    ),
    "flat": (
        lambda x: (x - 1.5) ** 3 + 1e-3 * (x - 1.5),
        lambda x: 3 * (x - 1.5) ** 2 + 1e-3,
        lambda x: 6 * (x - 1.5),
        (0.0, 3.0),
        2.5,
    ),
}


def system(v: npt.NDArray[np.float64]) -> list[float]:
    """x^2 + y^2 = 4, e^x + y = 1."""
    x, y = float(v[0]), float(v[1])
    return [x * x + y * y - 4, np.exp(x) + y - 1]


def cases() -> list[Case]:
    out: list[Case] = []
    for name, (f, fp, fpp, (a, b), x0) in FUNCTIONS.items():
        exact = so.brentq(f, a, b, xtol=1e-15)
        for method, solve in [
            ("bisect", so.bisect),
            ("brentq", so.brentq),
            ("ridder", so.ridder),
            ("toms748", so.toms748),
        ]:
            out.append(
                case(f"{method}/{name}", method, [], np.array(solve(f, a, b)), rtol=1e-10, atol=1e-11, fn_name=name)
            )
        out.append(
            case(
                f"newton/{name}",
                "newton",
                [],
                np.array(so.newton(f, x0, fprime=fp)),
                rtol=1e-8,
                atol=1e-9,
                fn_name=name,
            )
        )
        out.append(case(f"secant/{name}", "secant", [], np.array(so.newton(f, x0)), rtol=1e-8, atol=1e-9, fn_name=name))
        out.append(
            case(
                f"halley/{name}",
                "halley",
                [],
                np.array(so.newton(f, x0, fprime=fp, fprime2=fpp)),
                rtol=1e-8,
                atol=1e-9,
                fn_name=name,
            )
        )
        out.append(case(f"exact/{name}", "exact", [], np.array(exact), rtol=1e-12, fn_name=name))
    sol = so.fsolve(system, np.array([1.0, -1.0]), xtol=1e-14)
    out.append(case("fsolve/system", "fsolve", [np.array([1.0, -1.0])], sol, rtol=1e-10))
    out.append(case("broyden/system", "broyden", [np.array([1.0, -1.0])], sol, rtol=1e-8))
    fp_cos = so.fixed_point(np.cos, np.array([1.0]), xtol=1e-12)
    out.append(case("fixedPoint/cos", "fixedPoint", [np.array([1.0])], fp_cos, rtol=1e-10))
    return out
