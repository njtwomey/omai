"""H2 and H-infinity norms against python-control (system_norm, with SLICOT ab13dd for H-infinity)."""

import warnings

import control as ct
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from omai_oracle.lti import schur_stable_system, stable_system, system_norm

REFERENCE: dict[str, str] = {"control": ct.__version__}


def cases() -> list[Case]:
    g = rng(4168)
    out: list[Case] = []
    a, b, c, d = stable_system(g, 4, 2, 2)
    da, db, dc, dd = schur_stable_system(g, 3, 1, 2)
    systems = {
        "mimo": (a, b, c, d, 0.0),
        "mimo-strictly-proper": (a, b, c, np.zeros((2, 2)), 0.0),
        "lightly-damped": (
            np.array([[0.0, 1.0], [-1.0, -0.2]]),
            np.array([[0.0], [1.0]]),
            np.array([[1.0, 0.0]]),
            np.zeros((1, 1)),
            0.0,
        ),
        "discrete": (da, db, dc, dd, 0.2),
    }
    for name, (sa, sb, sc, sd, dt) in systems.items():
        sys = ct.ss(sa, sb, sc, sd, dt)
        args = [sa, sb, sc, sd]
        with warnings.catch_warnings():
            # python-control warns that a continuous system with feedthrough has an infinite H2 norm.
            warnings.simplefilter("ignore", UserWarning)
            h2 = system_norm(sys, 2)
        out.append(case(f"h2/{name}", "h2", args, np.array(h2), rtol=1e-9, dt=dt))
        out.append(case(f"hinf/{name}", "hinf", args, np.array(system_norm(sys, "inf")), rtol=1e-6, dt=dt))
    return out
