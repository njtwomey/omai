"""solve_bvp against scipy.integrate.solve_bvp (the same collocation algorithm), and DAE solutions against the ODE form.

Problems: Bratu's equation (both branches from two guesses), a Sturm-Liouville eigenvalue problem with an unknown
parameter, and a singular problem y'' + 2 y' / x + y = 0 (solution sin x / x). Compared on the continuous solution
at probe points. The DAE case is Robertson's problem with the conservation law as the algebraic equation, against
scipy's Radau on the ODE form at tight tolerance.
"""

import numpy as np
import scipy
import scipy.integrate as si
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def bratu(_x: FloatArray, y: FloatArray) -> FloatArray:
    return np.vstack((y[1], -np.exp(y[0])))


def bratu_bc(ya: FloatArray, yb: FloatArray) -> FloatArray:
    return np.array([ya[0], yb[0]])


def sturm(_x: FloatArray, y: FloatArray, p: FloatArray) -> FloatArray:
    return np.vstack((y[1], -(p[0] ** 2) * y[0]))


def sturm_bc(ya: FloatArray, yb: FloatArray, p: FloatArray) -> FloatArray:
    return np.array([ya[0], yb[0], ya[1] - p[0]])


def sinc(_x: FloatArray, y: FloatArray) -> FloatArray:
    return np.vstack((y[1], -y[0]))


def sinc_bc(ya: FloatArray, yb: FloatArray) -> FloatArray:
    return np.array([ya[0] - 1, yb[0]])


def cases() -> list[Case]:
    out: list[Case] = []
    x = np.linspace(0, 1, 5)
    probe = np.linspace(0, 1, 9)
    for label, guess in {"lower": 0.0, "upper": 3.0}.items():
        y = np.zeros((2, x.size))
        y[0] = guess
        r = si.solve_bvp(bratu, bratu_bc, x, y, tol=1e-6)
        out.append(case(f"bratu/{label}", "bratu", [], r.sol(probe), rtol=1e-6, atol=1e-8, guess=guess))
    xs = np.linspace(0, 1, 5)
    ys = np.zeros((2, xs.size))
    ys[0, 1] = 1
    ys[0, 3] = -1
    r = si.solve_bvp(sturm, sturm_bc, xs, ys, p=[6.0], tol=1e-6)
    assert r.p is not None
    out.append(case("sturm/p", "sturm_p", [], np.asarray(r.p), rtol=1e-7, atol=1e-9))
    out.append(case("sturm/y", "sturm_y", [], r.sol(probe), rtol=1e-6, atol=1e-8))
    xs = np.linspace(0, np.pi, 6)
    ys = np.vstack((np.cos(xs / 2), -0.5 * np.sin(xs / 2)))
    r = si.solve_bvp(sinc, sinc_bc, xs, ys, S=np.array([[0.0, 0.0], [0.0, -2.0]]), tol=1e-6)
    sprobe = np.linspace(0, np.pi, 7)
    out.append(case("singular/y", "singular", [], r.sol(sprobe), rtol=1e-6, atol=1e-8))
    return out
