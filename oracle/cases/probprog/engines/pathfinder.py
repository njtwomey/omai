"""Pathfinder (probprog/engines/pathfinder.ts) against BlackJAX's `pathfinder.approximate` on NumPyro's unconstrained
potential: the moments (mean and sd of every unconstrained coordinate) of 20,000 draws from the approximation of the
best ELBO along the L-BFGS path, and that ELBO. The two are approximations of the same posterior from different starts,
so the TypeScript test compares them within a fraction of the posterior sd, not to digits. Coordinates follow the
program's execution order (as the TypeScript layout)."""

import importlib.metadata
import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.infer.util import potential_energy, unconstrain_fn
from omai_oracle.fixture import Case, case
from omai_oracle.pgm import stable

_spec = importlib.util.spec_from_file_location("probprog_models", Path(__file__).parent.parent / "_models.py")
assert _spec is not None and _spec.loader is not None
_models: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models)
MODELS = cast(dict[str, Callable[[], None]], _models.MODELS)

REFERENCE: dict[str, str] = {
    "blackjax": importlib.metadata.version("blackjax"),
    "numpyro": importlib.metadata.version("numpyro"),
    "jax": importlib.metadata.version("jax"),
}

NAMES = ["normalMean", "mvn", "logistic", "gammaPoisson"]


def layout(model: Callable[[], None]) -> list[tuple[str, tuple[int, ...]]]:
    """Latent sites in execution order with their unconstrained shapes."""
    tr = handlers.trace(handlers.seed(model, 0)).get_trace()
    latent = {k: v["value"] for k, v in tr.items() if v["type"] == "sample" and not v["is_observed"]}
    u = unconstrain_fn(model, (), {}, latent)
    return [(k, tuple(np.shape(u[k]))) for k in latent]


def cases() -> list[Case]:
    out: list[Case] = []
    for name in NAMES:
        model = MODELS[name]
        sites = layout(model)
        sizes = [int(np.prod(s)) for _, s in sites]

        def unpack(
            x: jax.Array, sites: list[tuple[str, tuple[int, ...]]] = sites, sizes: list[int] = sizes
        ) -> dict[str, jax.Array]:
            out_: dict[str, jax.Array] = {}
            at = 0
            for (k, shape), n in zip(sites, sizes, strict=True):
                out_[k] = jnp.reshape(x[at : at + n], shape)
                at += n
            return out_

        def logdensity(
            x: jax.Array,
            model: Callable[[], None] = model,
            unpack: Callable[[jax.Array], dict[str, jax.Array]] = unpack,
        ) -> jax.Array:
            return -jnp.asarray(potential_energy(model, (), {}, unpack(x)))

        d = sum(sizes)
        state, _ = blackjax.pathfinder.approximate(
            jax.random.key(0), logdensity, jnp.zeros(d), num_samples=500, maxiter=60, maxcor=10
        )
        draws, _ = blackjax.pathfinder.sample(jax.random.key(1), state, 20000)
        draws = np.asarray(draws, dtype=np.float64).reshape(20000, d)
        moments = np.concatenate([draws.mean(axis=0), draws.std(axis=0)])
        out.append(case(f"moments/{name}", "moments", [], stable(moments), rtol=0.0, atol=0.0, model=name, dim=d))
        out.append(case(f"elbo/{name}", "elbo", [], stable(float(state.elbo)), rtol=0.0, atol=0.0, model=name))
    return out
