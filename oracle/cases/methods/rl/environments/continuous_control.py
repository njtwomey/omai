"""Continuous-control dynamics against independent references.

- The double integrator and the damped point mass: `scipy.signal.cont2discrete` (zero-order hold) of the continuous
  linear systems.
- The cart double pendulum and the 2-D reacher: equations of motion derived here with sympy from each system's
  Lagrangian (and Rayleigh dissipation function), evaluated with `numpy.linalg.solve`; a step against
  `scipy.integrate.solve_ivp` (DOP853 at 1e-12) at a tolerance matching RK4's local error; and RK4 trajectories on the
  sympy equations (the environments integrate by RK4) at 1e-10.
- The planar quadrotor: Newton's equations in numpy, `solve_ivp` steps and RK4 trajectories.
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import scipy
import sympy as sp
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "sympy": sp.__version__, "numpy": np.__version__}

type Rhs = Callable[[FloatArray, FloatArray], FloatArray]


def lagrangian_rhs(
    q: Sequence[sp.Symbol],
    qd: Sequence[sp.Symbol],
    kinetic: sp.Expr,
    potential: sp.Expr,
    rayleigh: sp.Expr,
    forces: Sequence[sp.Expr],
    inputs: Sequence[sp.Symbol],
) -> Rhs:
    """`d/dt [q, qd]` from the Euler-Lagrange equations `d/dt dL/dqd - dL/dq + dR/dqd = Q`, as a numpy function."""
    lag = kinetic - potential
    n = len(q)
    mass = sp.Matrix([[sp.diff(lag, qd[i], qd[j]) for j in range(n)] for i in range(n)])
    # Everything in d/dt dL/dqd except the qdd terms, minus dL/dq, plus the dissipation, minus the inputs.
    rest = sp.Matrix(
        [
            sp.Add(*[sp.diff(lag, qd[i], q[j]) * qd[j] for j in range(n)])
            - sp.diff(lag, q[i])
            + sp.diff(rayleigh, qd[i])
            - forces[i]
            for i in range(n)
        ]
    )
    m_fn = cast(Callable[..., Any], sp.lambdify([*q, *qd, *inputs], mass, "numpy"))
    r_fn = cast(Callable[..., Any], sp.lambdify([*q, *qd, *inputs], rest, "numpy"))

    def rhs(s: FloatArray, u: FloatArray) -> FloatArray:
        args = [*s.tolist(), *u.tolist()]
        qdd = np.linalg.solve(np.asarray(m_fn(*args), dtype=np.float64), -np.asarray(r_fn(*args), dtype=np.float64))
        return np.concatenate([s[n:], qdd.ravel()])

    return rhs


def cart_double_pendulum(
    cart_mass: float = 1.0,
    m1: float = 0.1,
    m2: float = 0.1,
    l1: float = 0.6,
    l2: float = 0.6,
    g: float = 9.81,
    cart_damping: float = 0.0,
    joint_damping: float = 0.0,
) -> Rhs:
    """A cart with two uniform poles at absolute angles from upright, pushed by a horizontal force."""
    x, t1, t2, xd, w1, w2, f = sp.symbols("x t1 t2 xd w1 w2 f")
    c1 = (x + l1 / 2 * sp.sin(t1), l1 / 2 * sp.cos(t1))
    c2 = (x + l1 * sp.sin(t1) + l2 / 2 * sp.sin(t2), l1 * sp.cos(t1) + l2 / 2 * sp.cos(t2))
    q = [x, t1, t2]
    qd = [xd, w1, w2]

    def velocity(p: tuple[sp.Expr, sp.Expr]) -> tuple[sp.Expr, sp.Expr]:
        return (
            sp.Add(*[sp.diff(p[0], a) * b for a, b in zip(q, qd, strict=True)]),
            sp.Add(*[sp.diff(p[1], a) * b for a, b in zip(q, qd, strict=True)]),
        )

    v1 = velocity(c1)
    v2 = velocity(c2)
    kinetic = (
        cart_mass * xd**2 / 2
        + m1 * (v1[0] ** 2 + v1[1] ** 2) / 2
        + m2 * (v2[0] ** 2 + v2[1] ** 2) / 2
        + (m1 * l1**2 / 12) * w1**2 / 2
        + (m2 * l2**2 / 12) * w2**2 / 2
    )
    potential = g * (m1 * c1[1] + m2 * c2[1])
    rayleigh = cart_damping * xd**2 / 2 + joint_damping * w1**2 / 2 + joint_damping * (w2 - w1) ** 2 / 2
    return lagrangian_rhs(q, qd, kinetic, potential, rayleigh, [f, sp.Integer(0), sp.Integer(0)], [f])


def reacher(
    l1: float = 0.1,
    l2: float = 0.11,
    m1: float = 1.0,
    m2: float = 1.0,
    armature: float = 0.01,
    damping: float = 0.05,
    gain: float = 0.2,
) -> Rhs:
    """A two-link arm of uniform links in a horizontal plane with rotor inertia and viscous joint damping."""
    q1, q2, qd1, qd2, u1, u2 = sp.symbols("q1 q2 qd1 qd2 u1 u2")
    q = [q1, q2]
    qd = [qd1, qd2]
    c1 = (l1 / 2 * sp.cos(q1), l1 / 2 * sp.sin(q1))
    c2 = (l1 * sp.cos(q1) + l2 / 2 * sp.cos(q1 + q2), l1 * sp.sin(q1) + l2 / 2 * sp.sin(q1 + q2))

    def speed2(p: tuple[sp.Expr, sp.Expr]) -> sp.Expr:
        vx = sp.Add(*[sp.diff(p[0], a) * b for a, b in zip(q, qd, strict=True)])
        vy = sp.Add(*[sp.diff(p[1], a) * b for a, b in zip(q, qd, strict=True)])
        return sp.Add(vx**2, vy**2)

    kinetic = (
        m1 * speed2(c1) / 2
        + m2 * speed2(c2) / 2
        + (m1 * l1**2 / 12) * qd1**2 / 2
        + (m2 * l2**2 / 12) * (qd1 + qd2) ** 2 / 2
        + armature * (qd1**2 + qd2**2) / 2
    )
    rayleigh = damping * (qd1**2 + qd2**2) / 2
    return lagrangian_rhs(q, qd, kinetic, sp.Integer(0), rayleigh, [gain * u1, gain * u2], [u1, u2])


def quadrotor(mass: float = 1.0, inertia: float = 0.01, arm: float = 0.2, g: float = 9.81) -> Rhs:
    """Newton's equations of the planar quadrotor under left and right thrusts."""

    def rhs(s: FloatArray, u: FloatArray) -> FloatArray:
        thrust = u[0] + u[1]
        return np.array(
            [
                s[3],
                s[4],
                s[5],
                -thrust * np.sin(s[2]) / mass,
                thrust * np.cos(s[2]) / mass - g,
                arm * (u[1] - u[0]) / inertia,
            ]
        )

    return rhs


def rk4(rhs: Rhs, s: FloatArray, u: FloatArray, dt: float, substeps: int) -> FloatArray:
    h = dt / substeps
    y = s.copy()
    for _ in range(substeps):
        k1 = rhs(y, u)
        k2 = rhs(y + h / 2 * k1, u)
        k3 = rhs(y + h / 2 * k2, u)
        k4 = rhs(y + h * k3, u)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def ivp_step(rhs: Rhs, s: FloatArray, u: FloatArray, dt: float) -> FloatArray:
    def fun(_t: float, y: FloatArray) -> FloatArray:
        return rhs(y, u)

    sol = solve_ivp(fun, (0.0, dt), s, method="DOP853", rtol=1e-12, atol=1e-12)
    return np.asarray(sol.y[:, -1], dtype=np.float64)


def system_cases(
    out: list[Case],
    name: str,
    rhs: Rhs,
    label: str,
    options: dict[str, object],
    points: list[tuple[FloatArray, FloatArray]],
    start: FloatArray,
    actions: list[FloatArray],
    steps: tuple[float, int],
) -> None:
    """Derivative and solve_ivp step cases at `points` and an RK4 trajectory from `start`, for one setting."""
    dt, substeps = steps
    for i, (s, u) in enumerate(points):
        out.append(case(f"{name}/{label}/derivatives/{i}", f"{name}Derivatives", [s, u], rhs(s, u), options=options))
        # One step of RK4 against an accurate solution: RK4's local error is O(h^5).
        out.append(
            case(
                f"{name}/{label}/ivp/{i}",
                f"{name}Step",
                [s, u],
                ivp_step(rhs, s, u, dt),
                rtol=1e-6,
                atol=1e-6,
                options=options,
            )
        )
    s = start.copy()
    trajectory: list[FloatArray] = []
    for u in actions:
        s = rk4(rhs, s, u, dt, substeps)
        trajectory.append(s)
    out.append(
        case(
            f"{name}/{label}/rk4",
            f"{name}Trajectory",
            [start, np.stack(actions)],
            np.stack(trajectory),
            rtol=1e-10,
            atol=1e-12,
            options=options,
        )
    )


def cases() -> list[Case]:
    g = rng(71)
    out: list[Case] = []
    # Exact discretisations of the linear systems.
    for mass, dt in [(1.0, 0.1), (2.0, 0.05), (0.5, 0.5)]:
        a = np.array([[0.0, 1.0], [0.0, 0.0]])
        b = np.array([[0.0], [1.0 / mass]])
        ad, bd, *_ = cont2discrete((a, b, np.eye(2), np.zeros((2, 1))), dt, method="zoh")
        out.append(
            case(
                f"double-integrator/{mass}/{dt}", "doubleIntegratorMatrices", [], np.hstack([ad, bd]), mass=mass, dt=dt
            )
        )
    for mass, damping, gain, dt in [(0.3, 1.0, 0.1, 0.02), (1.0, 0.5, 1.0, 0.1), (0.3, 0.0, 0.1, 0.02)]:
        a = np.array([[0.0, 1.0], [0.0, -damping / mass]])
        b = np.array([[0.0], [gain / mass]])
        ad, bd, *_ = cont2discrete((a, b, np.eye(2), np.zeros((2, 1))), dt, method="zoh")
        for i in range(2):
            s = g.uniform(-0.1, 0.1, 4)
            u = g.uniform(-1.0, 1.0, 2)
            nxt = [ad @ s[[0, 2]] + bd[:, 0] * u[0], ad @ s[[1, 3]] + bd[:, 0] * u[1]]
            out.append(
                case(
                    f"point-mass/{mass}/{damping}/{i}",
                    "pointMassStep",
                    [s, u],
                    np.array([nxt[0][0], nxt[1][0], nxt[0][1], nxt[1][1]]),
                    mass=mass,
                    damping=damping,
                    gain=gain,
                    dt=dt,
                )
            )
    # The cart double pendulum, frictionless and with friction.
    cdp: list[tuple[str, dict[str, float], dict[str, object]]] = [
        ("default", {}, {}),
        (
            "friction",
            {"cart_mass": 2.0, "m1": 0.3, "m2": 0.2, "l1": 0.5, "l2": 0.7, "cart_damping": 0.4, "joint_damping": 0.05},
            {"cartMass": 2.0, "masses": [0.3, 0.2], "lengths": [0.5, 0.7], "cartDamping": 0.4, "jointDamping": 0.05},
        ),
    ]
    for label, kwargs, options in cdp:
        rhs = cart_double_pendulum(**kwargs)
        points = [
            (
                np.concatenate([g.uniform(-1, 1, 1), g.uniform(-np.pi, np.pi, 2), g.uniform(-2, 2, 3)]),
                g.uniform(-20, 20, 1),
            )
            for _ in range(4)
        ]
        start = np.concatenate([np.zeros(1), g.uniform(-0.3, 0.3, 2), np.zeros(3)])
        actions = [g.uniform(-20, 20, 1) for _ in range(25)]
        system_cases(out, "cartDoublePendulum", rhs, label, options, points, start, actions, (0.02, 2))
    # The reacher, with gymnasium-like links and with heavier undamped ones.
    arms: list[tuple[str, dict[str, float], dict[str, object]]] = [
        ("default", {}, {}),
        (
            "heavy",
            {"m1": 2.0, "m2": 0.5, "armature": 0.005, "damping": 0.0, "gain": 0.5},
            {"masses": [2.0, 0.5], "armature": 0.005, "damping": 0.0, "gain": 0.5},
        ),
    ]
    for label, kwargs, options in arms:
        rhs = reacher(**kwargs)
        points = [
            (np.concatenate([g.uniform(-np.pi, np.pi, 2), g.uniform(-5, 5, 2)]), g.uniform(-1, 1, 2)) for _ in range(4)
        ]
        start = np.concatenate([g.uniform(-1, 1, 2), np.zeros(2)])
        actions = [g.uniform(-1, 1, 2) for _ in range(25)]
        system_cases(out, "reacher2d", rhs, label, options, points, start, actions, (0.02, 1))
    rhs = quadrotor()
    points = [(np.concatenate([g.uniform(-1, 1, 3), g.uniform(-2, 2, 3)]), g.uniform(3.9, 5.9, 2)) for _ in range(4)]
    start = np.concatenate([g.uniform(-0.5, 0.5, 3), np.zeros(3)])
    actions = [g.uniform(3, 7, 2) for _ in range(25)]
    system_cases(out, "planarQuadrotor", rhs, "default", {}, points, start, actions, (0.02, 1))
    return out
