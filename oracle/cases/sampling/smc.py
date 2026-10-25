"""Resampling schemes and the adaptive-tempering step against BlackJAX (float64), with the same uniforms."""

import importlib.metadata

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.smc import ess, resampling, solver
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

jax.config.update("jax_enable_x64", True)

REFERENCE: dict[str, str] = {
    "blackjax": importlib.metadata.version("blackjax"),
    "jax": importlib.metadata.version("jax"),
}


def next_temperature(loglik: FloatArray, temperature: float, target_ess: float) -> float:
    """BlackJAX's adaptive tempering: lambda plus the clipped ESS-solver increment."""
    max_delta = 1.0 - temperature

    def loglikelihood(x: jax.Array) -> jax.Array:
        return x

    delta = ess.ess_solver(loglikelihood, jnp.asarray(loglik), target_ess, max_delta, solver.dichotomy)
    return temperature + float(jnp.clip(delta, 0.0, max_delta))


def cases() -> list[Case]:
    g = rng(29)
    out: list[Case] = []
    for i, (k, n) in enumerate([(5, 5), (8, 12), (20, 20), (3, 7)]):
        w = g.dirichlet(np.ones(k) * 0.7)
        key = jax.random.key(i)
        u = np.asarray(jax.random.uniform(key, ()), dtype=np.float64)
        idx = np.asarray(resampling.systematic(key, jnp.asarray(w), n), dtype=np.float64)
        out.append(case(f"systematic/{k}/{n}", "systematic", [w, u], idx, n=n))
        us = np.asarray(jax.random.uniform(key, (n,)), dtype=np.float64)
        idx = np.asarray(resampling.stratified(key, jnp.asarray(w), n), dtype=np.float64)
        out.append(case(f"stratified/{k}/{n}", "stratified", [w, us], idx, n=n))
        um = np.asarray(jax.random.uniform(key, (n + 1,)), dtype=np.float64)
        idx = np.asarray(resampling.multinomial(key, jnp.asarray(w), n), dtype=np.float64)
        out.append(case(f"multinomial/{k}/{n}", "multinomial", [w, um], idx, n=n))
    for j, (scale, temperature, target) in enumerate(
        [(1.0, 0.0, 0.5), (5.0, 0.1, 0.5), (30.0, 0.4, 0.7), (0.2, 0.0, 0.5)]
    ):
        loglik = -scale * g.exponential(size=50 + 10 * j)
        lam = next_temperature(loglik, temperature, target)
        out.append(
            case(
                f"nextTemperature/{scale}/{temperature}/{target}",
                "nextTemperature",
                [loglik],
                np.array(lam),
                rtol=1e-12,
                temperature=temperature,
                targetEss=target,
            )
        )
    return out
