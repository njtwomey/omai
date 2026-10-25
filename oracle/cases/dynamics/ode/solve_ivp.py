"""solve_ivp (every method, dense output, events), odeint and step counts against scipy.integrate.

The TypeScript golden test defines the same problems by name. Explicit Runge-Kutta methods follow scipy's algorithm
step for step, so they are compared tightly (with exact evaluation counts); Radau and BDF follow it up to the Newton
iterations (scipy is given the analytic Jacobian; mdml-js differentiates the field), and LSODA and odeint use a
different implementation of the same idea, so those compare at a tolerance tied to the solver tolerance.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
import scipy
import scipy.integrate as si
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}

type Field = Callable[[float, FloatArray], FloatArray]


def decay(_t: float, y: FloatArray) -> FloatArray:
    return np.array([-0.5 * y[0], -2.0 * y[1]])


def decay_jac(_t: float, _y: FloatArray) -> FloatArray:
    return np.array([[-0.5, 0.0], [0.0, -2.0]])


def lotka(_t: float, y: FloatArray) -> FloatArray:
    return np.array([1.5 * y[0] - y[0] * y[1], -3.0 * y[1] + y[0] * y[1]])


def lotka_jac(_t: float, y: FloatArray) -> FloatArray:
    return np.array([[1.5 - y[1], -y[0]], [y[1], -3.0 + y[0]]])


def forced(t: float, y: FloatArray) -> FloatArray:
    return np.array([y[1], -y[0] - 0.1 * y[1] + np.cos(1.3 * t)])


def forced_jac(_t: float, _y: FloatArray) -> FloatArray:
    return np.array([[0.0, 1.0], [-1.0, -0.1]])


def vdp(_t: float, y: FloatArray) -> FloatArray:
    return np.array([y[1], 10.0 * (1 - y[0] ** 2) * y[1] - y[0]])


def vdp_jac(_t: float, y: FloatArray) -> FloatArray:
    return np.array([[0.0, 1.0], [-20.0 * y[0] * y[1] - 1.0, 10.0 * (1 - y[0] ** 2)]])


def robertson(_t: float, y: FloatArray) -> FloatArray:
    return np.array(
        [
            -0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
            3e7 * y[1] ** 2,
        ]
    )


def robertson_jac(_t: float, y: FloatArray) -> FloatArray:
    return np.array(
        [
            [-0.04, 1e4 * y[2], 1e4 * y[1]],
            [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
            [0.0, 6e7 * y[1], 0.0],
        ]
    )


# name -> (field, jacobian, y0, span, output times)
PROBLEMS: dict[str, tuple[Field, Field, list[float], tuple[float, float], list[float]]] = {
    "decay": (decay, decay_jac, [1.0, 2.0], (0.0, 3.0), [0.5, 1.0, 2.0, 3.0]),
    "lotka": (lotka, lotka_jac, [10.0, 5.0], (0.0, 4.0), [0.7, 1.9, 3.1, 4.0]),
    "forced": (forced, forced_jac, [1.0, 0.0], (0.0, 6.0), [1.0, 2.5, 4.0, 6.0]),
    "vdp": (vdp, vdp_jac, [2.0, 0.0], (0.0, 3.0), [0.5, 1.5, 3.0]),
    "robertson": (robertson, robertson_jac, [1.0, 0.0, 0.0], (0.0, 40.0), [0.4, 4.0, 40.0]),
}

type Method = Literal["RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"]

EXPLICIT: list[Method] = ["RK23", "RK45", "DOP853"]
IMPLICIT: list[Method] = ["Radau", "BDF"]


def ys(sol: object) -> FloatArray:
    return np.asarray(sol.y, dtype=np.float64).T  # type: ignore[attr-defined]


def first_crossing(_t: float, y: FloatArray) -> float:
    return float(y[0])


def speed_zero(_t: float, y: FloatArray) -> float:
    return float(y[1])


def cases() -> list[Case]:
    out: list[Case] = []
    for name in ["decay", "lotka", "forced"]:
        f, _jac, y0, span, t_eval = PROBLEMS[name]
        for method in EXPLICIT:
            for rtol in [1e-4, 1e-8]:
                sol = si.solve_ivp(f, span, y0, method=method, rtol=rtol, atol=rtol * 1e-2, t_eval=t_eval)
                label = f"{method}/{name}/{rtol:g}"
                out.append(
                    case(
                        f"{label}/y",
                        "solve_ivp",
                        [],
                        ys(sol),
                        rtol=1e-9,
                        atol=1e-11,
                        problem=name,
                        method=method,
                        tol=rtol,
                        what="y",
                    )
                )
                out.append(
                    case(
                        f"{label}/nfev",
                        "solve_ivp",
                        [],
                        np.array(float(sol.nfev)),
                        rtol=0,
                        atol=0,
                        problem=name,
                        method=method,
                        tol=rtol,
                        what="nfev",
                    )
                )
                steps = si.solve_ivp(f, span, y0, method=method, rtol=rtol, atol=rtol * 1e-2)
                # Step times agree to the rounding of the error estimate (which cancels heavily), about 1e-8.
                out.append(
                    case(
                        f"{label}/t",
                        "solve_ivp",
                        [],
                        np.asarray(steps.t),
                        rtol=1e-6,
                        atol=1e-9,
                        problem=name,
                        method=method,
                        tol=rtol,
                        what="t",
                    )
                )
            dense = si.solve_ivp(f, span, y0, method=method, rtol=1e-6, atol=1e-8, dense_output=True)
            probe = np.linspace(span[0], span[1], 7)
            assert dense.sol is not None
            out.append(
                case(
                    f"{method}/{name}/dense",
                    "solve_ivp",
                    [],
                    dense.sol(probe).T,
                    rtol=1e-9,
                    atol=1e-11,
                    problem=name,
                    method=method,
                    tol=1e-6,
                    what="dense",
                )
            )
    for name in ["decay", "vdp", "robertson"]:
        f, jac, y0, span, t_eval = PROBLEMS[name]
        for method in IMPLICIT:
            for rtol in [1e-4, 1e-7]:
                atol = 1e-10 if name == "robertson" else rtol * 1e-2
                sol = si.solve_ivp(f, span, y0, method=method, rtol=rtol, atol=atol, t_eval=t_eval, jac=jac)
                tol = 20 * rtol
                out.append(
                    case(
                        f"{method}/{name}/{rtol:g}/y",
                        "solve_ivp",
                        [],
                        ys(sol),
                        rtol=tol,
                        atol=50 * atol,
                        problem=name,
                        method=method,
                        tol=rtol,
                        atol_solver=atol,
                        what="y",
                    )
                )
        for rtol in [1e-6, 1e-9]:
            atol = 1e-12 if name == "robertson" else rtol * 1e-2
            sol = si.solve_ivp(f, span, y0, method="LSODA", rtol=rtol, atol=atol, t_eval=t_eval, jac=jac)
            out.append(
                case(
                    f"LSODA/{name}/{rtol:g}/y",
                    "solve_ivp",
                    [],
                    ys(sol),
                    rtol=200 * rtol,
                    atol=200 * atol,
                    problem=name,
                    method="LSODA",
                    tol=rtol,
                    atol_solver=atol,
                    what="y",
                )
            )
    # Events: the first downward zero of x and every zero of the velocity of the forced oscillator.
    f = PROBLEMS["forced"][0]
    terminal: Callable[[float, FloatArray], float] = first_crossing
    terminal.terminal = True  # type: ignore[attr-defined]
    terminal.direction = -1  # type: ignore[attr-defined]
    for method in EXPLICIT:
        sol = si.solve_ivp(
            f, (0.0, 20.0), [1.0, 0.0], method=method, rtol=1e-9, atol=1e-11, events=[terminal, speed_zero]
        )
        out.append(
            case(
                f"{method}/events/terminal",
                "events",
                [],
                np.asarray(sol.t_events[0]),
                rtol=1e-9,
                atol=1e-11,
                method=method,
                which=0,
            )
        )
        out.append(
            case(
                f"{method}/events/speed",
                "events",
                [],
                np.asarray(sol.t_events[1]),
                rtol=1e-9,
                atol=1e-11,
                method=method,
                which=1,
            )
        )
        out.append(
            case(
                f"{method}/events/end",
                "events",
                [],
                np.array(float(sol.t[-1])),
                rtol=1e-9,
                atol=1e-11,
                method=method,
                which=-1,
            )
        )
    # odeint at its default tolerances (LSODA in both libraries, compared to their tolerance).
    for name in ["lotka", "vdp"]:
        f, _jac, y0, span, t_eval = PROBLEMS[name]
        times = np.array([span[0], *t_eval])
        y = si.odeint(f, y0, times, tfirst=True)
        # scipy's odeint is itself only good to about 5e-6 here (against DOP853 at 1e-13), so the tolerance follows it.
        out.append(case(f"odeint/{name}", "odeint", [], np.asarray(y), rtol=2e-5, atol=1e-6, problem=name))
    return out
