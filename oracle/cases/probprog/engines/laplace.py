"""Laplace approximations (probprog/engines/laplace.ts) against NumPyro's unconstrained potential: the MAP point by
scipy's BFGS on `potential_energy` with JAX gradients, the covariance as the inverse JAX Hessian there, and the
Laplace evidence. Coordinates follow the program's execution order (as the TypeScript layout)."""

import importlib.metadata
import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.pgm import stable
from numpyro import handlers
from numpyro.infer.util import potential_energy, unconstrain_fn
from scipy.optimize import minimize

_spec = importlib.util.spec_from_file_location("probprog_models", Path(__file__).parent.parent / "_models.py")
assert _spec is not None and _spec.loader is not None
_models: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models)
MODELS = cast(dict[str, Callable[[], None]], _models.MODELS)

REFERENCE: dict[str, str] = {
    "numpyro": importlib.metadata.version("numpyro"),
    "jax": importlib.metadata.version("jax"),
    "scipy": importlib.metadata.version("scipy"),
}

NAMES = ["normalMean", "gammaPoisson", "bounded", "logistic", "simplex", "mvn"]


def layout(model: Callable[[], None]) -> list[tuple[str, tuple[int, ...]]]:
    """Latent sites in execution order with their unconstrained shapes."""
    tr = handlers.trace(handlers.seed(model, 0)).get_trace()
    latent = {k: v["value"] for k, v in tr.items() if v["type"] == "sample" and not v["is_observed"]}
    u = unconstrain_fn(model, (), {}, latent)
    return [(k, tuple(np.shape(u[k]))) for k in latent]


def map_point(pe: Callable[[jax.Array], jax.Array], d: int) -> np.ndarray:
    """The minimiser of a potential from zero, by BFGS with JAX gradients and a tight gradient tolerance."""
    f = jax.jit(pe)
    g = jax.jit(jax.grad(pe))

    def value(x: np.ndarray) -> float:
        return float(f(jnp.asarray(x)))

    def gradient(x: np.ndarray) -> np.ndarray:
        return np.asarray(g(jnp.asarray(x)), dtype=np.float64)

    r = minimize(value, np.zeros(d), jac=gradient, method="BFGS", options={"gtol": 1e-12, "maxiter": 10000})
    return np.asarray(r.x, dtype=np.float64)


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

        def pe(
            x: jax.Array,
            model: Callable[[], None] = model,
            unpack: Callable[[jax.Array], dict[str, jax.Array]] = unpack,
        ) -> jax.Array:
            return jnp.asarray(potential_energy(model, (), {}, unpack(x)))

        f = jax.jit(pe)
        d = sum(sizes)
        mode = map_point(pe, d)
        hess = np.asarray(jax.hessian(pe)(jnp.asarray(mode)), dtype=np.float64)
        cov = np.linalg.inv(hess)
        _, logdet = np.linalg.slogdet(cov)
        evidence = -float(f(jnp.asarray(mode))) + 0.5 * d * np.log(2 * np.pi) + 0.5 * logdet
        out.append(case(f"mode/{name}", "mode", [], stable(mode), rtol=1e-6, atol=1e-7, model=name))
        out.append(case(f"covariance/{name}", "covariance", [], stable(cov), rtol=1e-6, atol=1e-8, model=name))
        out.append(case(f"evidence/{name}", "evidence", [], stable(evidence), rtol=1e-8, model=name))
    return out
