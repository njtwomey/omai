"""Named systems against scipy.integrate.solve_ivp (DOP853, rtol 1e-13): Rossler, Thomas, the double pendulum (the
closed-form equations, and the same system derived from its Lagrangian), Kuramoto oscillators, and the double pendulum's
time to its first flip located with an integration event.
"""

from collections.abc import Callable

import numpy as np
import scipy
import scipy.integrate as si
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}

type Field = Callable[[float, FloatArray], FloatArray]

PENDULUM: dict[str, float] = {"m1": 1.3, "m2": 0.7, "l1": 0.9, "l2": 1.4, "g": 9.81}


def rossler(_t: float, y: FloatArray) -> FloatArray:
    a, b, c = 0.2, 0.2, 5.7
    return np.array([-y[1] - y[2], y[0] + a * y[1], b + y[2] * (y[0] - c)])


def thomas(_t: float, y: FloatArray) -> FloatArray:
    b = 0.208186
    return np.array([np.sin(y[1]) - b * y[0], np.sin(y[2]) - b * y[1], np.sin(y[0]) - b * y[2]])


def pendulum(p: dict[str, float]) -> Field:
    m1, m2, l1, l2, g = p["m1"], p["m2"], p["l1"], p["l2"], p["g"]

    def f(_t: float, y: FloatArray) -> FloatArray:
        t1, t2, w1, w2 = y
        d = t1 - t2
        den = 2 * m1 + m2 - m2 * np.cos(2 * d)
        a1 = (
            -g * (2 * m1 + m2) * np.sin(t1)
            - m2 * g * np.sin(t1 - 2 * t2)
            - 2 * np.sin(d) * m2 * (w2**2 * l2 + w1**2 * l1 * np.cos(d))
        ) / (l1 * den)
        a2 = (2 * np.sin(d) * (w1**2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(t1) + w2**2 * l2 * m2 * np.cos(d))) / (
            l2 * den
        )
        return np.array([w1, w2, a1, a2])

    return f


def kuramoto(omega: FloatArray, k: float) -> Field:
    n = len(omega)

    def f(_t: float, y: FloatArray) -> FloatArray:
        diff = y[None, :] - y[:, None]
        return omega + (k / n) * np.sin(diff).sum(axis=1)

    return f


def trajectory(f: Field, y0: FloatArray, times: FloatArray) -> FloatArray:
    sol = si.solve_ivp(f, (0.0, float(times[-1])), y0, method="DOP853", t_eval=times, rtol=1e-13, atol=1e-14)
    return np.asarray(sol.y.T, dtype=np.float64)


def flip_time(theta1: float, theta2: float) -> float:
    f = pendulum({"m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0, "g": 9.81})

    def over_inner(t: float, y: FloatArray) -> float:
        return float(abs(y[0]) - np.pi)

    def over_outer(t: float, y: FloatArray) -> float:
        return float(abs(y[1]) - np.pi)

    over_inner.terminal = True  # type: ignore[attr-defined]
    over_outer.terminal = True  # type: ignore[attr-defined]
    sol = si.solve_ivp(
        f,
        (0.0, 10.0),
        np.array([theta1, theta2, 0.0, 0.0]),
        method="DOP853",
        events=[over_inner, over_outer],
        rtol=1e-13,
        atol=1e-14,
    )
    hits = [float(e[0]) for e in sol.t_events if len(e) > 0]
    return min(hits) if hits else float("inf")


def cases() -> list[Case]:
    out: list[Case] = []
    times = np.linspace(0.0, 10.0, 6)
    out.append(
        case(
            "rossler",
            "rossler",
            [np.array([1.0, 1.0, 1.0]), times],
            trajectory(rossler, np.array([1.0, 1.0, 1.0]), times),
            rtol=1e-6,
            atol=1e-7,
            dt=0.005,
        )
    )
    times = np.linspace(0.0, 20.0, 6)
    out.append(
        case(
            "thomas",
            "thomas",
            [np.array([1.1, 1.1, -0.01]), times],
            trajectory(thomas, np.array([1.1, 1.1, -0.01]), times),
            rtol=1e-6,
            atol=1e-7,
            dt=0.005,
        )
    )
    times = np.linspace(0.0, 4.0, 5)
    y0 = np.array([1.2, -0.5, 0.3, 0.8])
    reference = trajectory(pendulum(PENDULUM), y0, times)
    for fn in ["double-pendulum", "double-pendulum-lagrangian"]:
        out.append(case(f"{fn}", fn, [y0, times], reference, rtol=1e-6, atol=1e-7, dt=0.005, **PENDULUM))
    g = np.random.default_rng(5)
    omega = g.normal(size=12)
    theta0 = g.uniform(-np.pi, np.pi, size=12)
    times = np.linspace(0.0, 6.0, 7)
    out.append(
        case(
            "kuramoto",
            "kuramoto",
            [omega, theta0, times],
            trajectory(kuramoto(omega, 1.8), theta0, times),
            rtol=1e-8,
            atol=1e-9,
            K=1.8,
            dt=0.005,
        )
    )
    starts = np.array([[2.9, 2.9], [3.0, 1.0], [2.5, 2.0], [2.0, 2.8], [3.1, -2.0], [1.5, 3.0]])
    flips = np.array([flip_time(float(a), float(b)) for a, b in starts])
    out.append(case("flip-time", "flip-time", [starts], flips, rtol=1e-5, atol=1e-5))
    return out
