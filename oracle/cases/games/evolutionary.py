"""Evolutionary dynamics: the replicator (and replicator-mutator) equation and the asymmetric two-population replicator
dynamics against nashpy (integrated by scipy's odeint; the engine uses Runge-Kutta, so the tolerance is that of an
ODE solution), and projected replicator dynamics against OpenSpiel's `projected_replicator_dynamics` (exact
arithmetic of a discrete iteration)."""

from importlib.metadata import version

import nashpy as nash
import numpy as np
from open_spiel.python.algorithms import projected_replicator_dynamics as prd
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"nashpy": nash.__version__, "open_spiel": version("open_spiel")}


def cases() -> list[Case]:
    out: list[Case] = []
    times = np.linspace(0, 10, 101)
    rps = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    symmetric: dict[str, tuple[FloatArray, FloatArray, FloatArray | None]] = {
        "rock-paper-scissors": (rps, np.array([0.5, 0.3, 0.2]), None),
        "hawk-dove": (np.array([[-1.0, 2.0], [0.0, 1.0]]), np.array([0.1, 0.9]), None),
        "prisoners-dilemma": (np.array([[3.0, 0.0], [5.0, 1.0]]), np.array([0.95, 0.05]), None),
        "random-4": (rng(800).uniform(-1, 1, size=(4, 4)), np.array([0.1, 0.2, 0.3, 0.4]), None),
        "mutation": (
            rps + 0.5,
            np.array([0.6, 0.3, 0.1]),
            np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]),
        ),
    }
    for name, (a, x0, q) in symmetric.items():
        xs = nash.Game(a).replicator_dynamics(y0=x0, timepoints=times, mutation_matrix=q)
        params: dict[str, object] = {"x0": x0.tolist(), "timepoints": times.tolist()}
        if q is not None:
            params["mutation"] = q.tolist()
        out.append(
            case(f"replicator/{name}", "replicatorDynamics", [a], np.asarray(xs), rtol=1e-6, atol=1e-7, **params)
        )
    asymmetric: dict[str, tuple[FloatArray, FloatArray, FloatArray, FloatArray]] = {
        "battle-of-the-sexes": (
            np.array([[3.0, 0.0], [0.0, 2.0]]),
            np.array([[2.0, 0.0], [0.0, 3.0]]),
            np.array([0.3, 0.7]),
            np.array([0.6, 0.4]),
        ),
        "random-2x3": (
            rng(801).uniform(-1, 1, size=(2, 3)),
            rng(802).uniform(-1, 1, size=(2, 3)),
            np.array([0.5, 0.5]),
            np.array([0.2, 0.3, 0.5]),
        ),
    }
    for name, (a, b, x0, y0) in asymmetric.items():
        xs, ys = nash.Game(a, b).asymmetric_replicator_dynamics(x0=x0, y0=y0, timepoints=times)
        out.append(
            case(
                f"asymmetric/{name}",
                "asymmetricReplicatorDynamics",
                [a, b],
                np.hstack([np.asarray(xs), np.asarray(ys)]),
                x0=x0.tolist(),
                y0=y0.tolist(),
                timepoints=times.tolist(),
                rtol=1e-6,
                atol=1e-7,
            )
        )
    games: dict[str, list[FloatArray]] = {
        "rock-paper-scissors": [rps, -rps],
        "random-3x3": [rng(803).uniform(-1, 1, size=(3, 3)), rng(804).uniform(-1, 1, size=(3, 3))],
        "random-2x2x3": [rng(805 + p).uniform(-1, 1, size=(2, 2, 3)) for p in range(3)],
    }
    for name, payoffs in games.items():
        average = prd.projected_replicator_dynamics(payoffs, prd_iterations=300, prd_dt=0.01, prd_gamma=1e-3)
        out.append(
            case(
                f"projected/{name}",
                "projectedReplicatorDynamics",
                payoffs,
                np.array(0.0),
                iterations=300,
                dt=0.01,
                gamma=1e-3,
                average=[np.asarray(x, dtype=np.float64).tolist() for x in average],
                rtol=1e-10,
                atol=1e-12,
            )
        )
    return out
