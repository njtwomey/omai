"""Celestial mechanics against scipy: Kepler's equation (elliptic, hyperbolic, parabolic) solved by scipy.optimize's
Brent root-finder; two-body orbits propagated by the analytic solution against scipy.integrate.solve_ivp (DOP853 at
rtol 1e-13); an n-body integration and a restricted three-body trajectory against the same integrator; and the
collinear Lagrange points by Brent's method on the axis derivative of the pseudo-potential.
"""

import numpy as np
import scipy
import scipy.integrate as si
import scipy.optimize as so
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}

BRENT: dict[str, float] = {"xtol": 1e-15, "rtol": 4.5e-16}


def elliptic(m: float, e: float) -> float:
    """E - e sin E = M for any real M: the root is within e of M."""
    root: float = so.brentq(lambda x: x - e * np.sin(x) - m, m - 1.0, m + 1.0, xtol=BRENT["xtol"], rtol=1e-15)
    return root


def hyperbolic(m: float, e: float) -> float:
    """e sinh H - H = M: monotone, so a wide bracket will do."""
    root: float = so.brentq(lambda x: e * np.sinh(x) - x - m, -60.0, 60.0, xtol=BRENT["xtol"], rtol=1e-15)
    return root


def parabolic(m: float) -> float:
    """Barker's equation D + D^3 / 3 = M, returned as the true anomaly 2 atan D."""
    d: float = so.brentq(lambda x: x + x**3 / 3 - m, -1e4, 1e4, xtol=BRENT["xtol"], rtol=1e-15)
    return float(2 * np.arctan(d))


def kepler_field(mu: float, y: FloatArray) -> FloatArray:
    r = y[:3]
    return np.concatenate([y[3:], -mu * r / np.linalg.norm(r) ** 3])


def propagate(mu: float, y0: FloatArray, times: FloatArray) -> FloatArray:
    sol = si.solve_ivp(
        lambda _t, y: kepler_field(mu, y),
        (0.0, float(times[-1])),
        y0,
        method="DOP853",
        t_eval=times,
        rtol=1e-13,
        atol=1e-14,
    )
    return np.asarray(sol.y.T, dtype=np.float64)


def nbody_field(masses: FloatArray, g: float, y: FloatArray) -> FloatArray:
    n = len(masses)
    q = y[: 3 * n].reshape(n, 3)
    p = y[3 * n :].reshape(n, 3)
    diff = q[None, :, :] - q[:, None, :]
    r2 = (diff**2).sum(axis=2) + np.eye(n)
    w = g * masses[:, None] * masses[None, :] * r2**-1.5 * (1 - np.eye(n))
    force = (w[:, :, None] * diff).sum(axis=1)
    return np.concatenate([(p / masses[:, None]).ravel(), force.ravel()])


def cr3bp_field(mu: float, y: FloatArray) -> FloatArray:
    x, yy, vx, vy = y
    r1 = np.hypot(x + mu, yy)
    r2 = np.hypot(x - 1 + mu, yy)
    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    ay = -2 * vx + yy - (1 - mu) * yy / r1**3 - mu * yy / r2**3
    return np.array([vx, vy, ax, ay])


def collinear(mu: float) -> FloatArray:
    def d_omega(x: float) -> float:
        return x - (1 - mu) * (x + mu) / abs(x + mu) ** 3 - mu * (x - 1 + mu) / abs(x - 1 + mu) ** 3

    tiny = 1e-9
    roots = [
        so.brentq(d_omega, -mu + tiny, 1 - mu - tiny, xtol=1e-15, rtol=1e-15),
        so.brentq(d_omega, 1 - mu + tiny, 2.0, xtol=1e-15, rtol=1e-15),
        so.brentq(d_omega, -2.0, -mu - tiny, xtol=1e-15, rtol=1e-15),
    ]
    return np.array(roots, dtype=np.float64)


def cases() -> list[Case]:
    out: list[Case] = []
    ms = np.array([-7.5, -3.0, -0.4, 0.0, 1e-4, 0.5, 1.0, 2.0, 3.0, 3.14, 5.5, 12.0])
    for e in [0.0, 0.3, 0.7, 0.95, 0.999]:
        ans = np.array([elliptic(float(m), e) for m in ms])
        out.append(case(f"kepler/elliptic/{e}", "kepler-elliptic", [ms], ans, rtol=1e-11, atol=1e-11, e=e))
    mh = np.array([-30.0, -1.0, 0.0, 0.1, 2.0, 100.0])
    for e in [1.01, 1.5, 5.0]:
        ans = np.array([hyperbolic(float(m), e) for m in mh])
        out.append(case(f"kepler/hyperbolic/{e}", "kepler-hyperbolic", [mh], ans, rtol=1e-10, atol=1e-11, e=e))
    mp = np.array([-20.0, -1.0, 0.0, 0.3, 4.0, 1000.0])
    out.append(
        case(
            "kepler/parabolic",
            "kepler-parabolic",
            [mp],
            np.array([parabolic(float(m)) for m in mp]),
            rtol=1e-11,
            atol=1e-11,
        )
    )

    # Two-body orbits from a state vector: elliptic, parabolic (escape speed exactly) and hyperbolic.
    mu = 1.7
    times = np.linspace(0.0, 9.0, 10)
    r0 = np.array([1.2, -0.4, 0.5])
    direction = np.array([0.3, 0.8, -0.2])
    direction = direction - direction.dot(r0) / r0.dot(r0) * r0  # tangential
    direction = direction / np.linalg.norm(direction)
    speed_escape = float(np.sqrt(2 * mu / np.linalg.norm(r0)))
    for name, factor in {"ellipse": 0.8, "parabola": 1.0, "hyperbola": 1.4}.items():
        v0 = factor * speed_escape * direction
        y0 = np.concatenate([r0, v0])
        out.append(
            case(
                f"two-body/{name}",
                "two-body",
                [y0, times],
                propagate(mu, y0, times),
                rtol=1e-8,
                atol=1e-9,
                mu=mu,
            )
        )

    # Four bodies in three dimensions.
    g = np.random.default_rng(11)
    masses = np.array([1.0, 0.4, 2.2, 0.7])
    q0 = g.normal(size=(4, 3)) * 1.5
    p0 = g.normal(size=(4, 3)) * 0.2 * masses[:, None]
    grav = 0.9
    y0 = np.concatenate([q0.ravel(), p0.ravel()])
    sol = si.solve_ivp(
        lambda _t, y: nbody_field(masses, grav, y),
        (0.0, 1.0),
        y0,
        method="DOP853",
        rtol=1e-13,
        atol=1e-14,
    )
    final = np.asarray(sol.y[:, -1], dtype=np.float64)
    out.append(
        case("nbody/final", "nbody", [masses, q0, p0], final, rtol=1e-7, atol=1e-8, G=grav, t=1.0, dt=1.0 / 400.0)
    )

    # The restricted three-body problem.
    mus = np.array([1e-6, 0.01, 0.0385, 0.1215058560962404, 0.3, 0.5])
    out.append(
        case(
            "cr3bp/collinear",
            "lagrange-collinear",
            [mus],
            np.stack([collinear(float(m)) for m in mus]),
            rtol=1e-10,
            atol=1e-12,
        )
    )
    mu3 = 0.0121505856
    y3 = np.array([0.8, 0.0, 0.0, 0.35])
    t3 = np.linspace(0.0, 3.0, 7)
    sol3 = si.solve_ivp(
        lambda _t, y: cr3bp_field(mu3, y), (0.0, 3.0), y3, method="DOP853", t_eval=t3, rtol=1e-13, atol=1e-14
    )
    out.append(
        case("cr3bp/trajectory", "cr3bp", [y3, t3], np.asarray(sol3.y.T), rtol=1e-8, atol=1e-9, mu=mu3, dt=0.0005)
    )
    return out
