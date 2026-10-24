"""Symplectic integrator trajectories, kinetic energies and the U-turn criterion against BlackJAX (float64)."""

import importlib.metadata
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.mcmc import integrators, metrics
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng, spd

jax.config.update("jax_enable_x64", True)

REFERENCE: dict[str, str] = {
    "blackjax": importlib.metadata.version("blackjax"),
    "jax": importlib.metadata.version("jax"),
}

SCHEMES: dict[str, Callable[..., Any]] = {
    "velocityVerlet": integrators.velocity_verlet,
    "mcLachlan": integrators.mclachlan,
    "yoshida": integrators.yoshida,
    "omelyan": integrators.omelyan,
}


def log_density(x: jax.Array) -> jax.Array:
    """A banana-shaped density: x0 ~ N(0, 4), x1 | x0 ~ N(x0^2 / 2, 1) (the TypeScript test uses the same one)."""
    return -0.5 * (x[0] ** 2 / 4.0 + (x[1] - 0.5 * x[0] ** 2) ** 2)


def trajectory(scheme: str, imm: FloatArray, x0: FloatArray, p0: FloatArray, eps: float, steps: int) -> FloatArray:
    """Positions and momenta `[steps + 1, 4]` of `steps` integrator steps from (x0, p0)."""
    metric = metrics.gaussian_euclidean(jnp.asarray(imm))
    step = SCHEMES[scheme](log_density, metric.kinetic_energy)
    state = integrators.new_integrator_state(log_density, jnp.asarray(x0), jnp.asarray(p0))
    rows = [np.concatenate([np.asarray(state.position), np.asarray(state.momentum)])]
    for _ in range(steps):
        state = step(state, eps)
        rows.append(np.concatenate([np.asarray(state.position), np.asarray(state.momentum)]))
    return np.asarray(rows, dtype=np.float64)


def cases() -> list[Case]:
    g = rng(11)
    out: list[Case] = []
    x0 = np.array([0.7, -0.3])
    p0 = np.array([0.4, 1.1])
    diag = np.array([1.5, 0.6])
    dense = spd(g, 2) / 3.0
    for scheme in SCHEMES:
        for name, imm in [("identity", np.ones(2)), ("diagonal", diag), ("dense", dense)]:
            for eps in [0.1, -0.25]:
                traj = trajectory(scheme, imm, x0, p0, eps, 12)
                out.append(
                    case(
                        f"trajectory/{scheme}/{name}/{eps}",
                        "trajectory",
                        [x0, p0, imm],
                        traj,
                        rtol=1e-12,
                        atol=1e-12,
                        scheme=scheme,
                        stepSize=eps,
                        numSteps=12,
                    )
                )
    for name, imm in [("diagonal", diag), ("dense", dense)]:
        metric = metrics.gaussian_euclidean(jnp.asarray(imm))
        p = g.normal(size=(6, 2))
        energies = np.array([float(metric.kinetic_energy(jnp.asarray(q))) for q in p])
        out.append(case(f"kineticEnergy/{name}", "kineticEnergy", [p, imm], energies))
        triples = g.normal(size=(20, 3, 2))
        turning = np.array(
            [
                bool(metric.check_turning(jnp.asarray(t[0]), jnp.asarray(t[1]), jnp.asarray(t[2] + t[0] + t[1])))
                for t in triples
            ]
        )
        out.append(case(f"isTurning/{name}", "isTurning", [triples, imm], turning.astype(np.float64)))
    return out
