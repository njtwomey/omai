"""Log joint densities, potential energies and constrain/unconstrain maps of programs (probprog/density.ts) against
NumPyro's `log_density`, `potential_energy`, `constrain_fn` and `unconstrain_fn`; enumerated log densities against
brute-force sums of NumPyro log densities over every discrete configuration."""

import importlib.metadata
import importlib.util
import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.pgm import stable
from numpyro import handlers
from numpyro.infer.util import constrain_fn, log_density, potential_energy, unconstrain_fn
from scipy.special import logsumexp

_spec = importlib.util.spec_from_file_location("probprog_models", Path(__file__).with_name("_models.py"))
assert _spec is not None and _spec.loader is not None
_models: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models)
MODELS = cast(dict[str, Callable[[], None]], _models.MODELS)

REFERENCE: dict[str, str] = {
    "numpyro": importlib.metadata.version("numpyro"),
    "jax": importlib.metadata.version("jax"),
}

SUBSAMPLE = [1, 4, 7]


def program(name: str) -> Callable[[], None]:
    """The named program, with the subsampled plate's indices fixed."""
    model = MODELS[name]
    if name == "subsampled":
        return cast(Callable[[], None], handlers.substitute(model, data={"data": jnp.array(SUBSAMPLE)}))
    return model


def latent_draws(name: str, seed: int) -> dict[str, np.ndarray]:
    """Constrained values of the latent sites from one prior run."""
    tr = handlers.trace(handlers.seed(program(name), jax.random.PRNGKey(seed))).get_trace()
    return {k: stable(v["value"]) for k, v in tr.items() if v["type"] == "sample" and not v["is_observed"]}


def as_numpy(d: dict[str, Any]) -> dict[str, np.ndarray]:
    """Values as float64 arrays rounded to 14 digits (JAX's reductions vary in the last bits between runs)."""
    return {k: stable(v) for k, v in d.items()}


def enumerated_mixture(locs: np.ndarray) -> tuple[float, np.ndarray]:
    """log p(y | locs) summing the three assignments out, and P(z_i = 1 | y), by brute force."""
    model = _models.DISCRETE["mixture"]
    logs: list[float] = []
    configs = list(itertools.product([0, 1], repeat=3))
    for z in configs:
        ld, _ = log_density(model, (), {}, {"locs": jnp.asarray(locs), "z": jnp.array(z)})
        logs.append(float(ld))
    lw = np.array(logs)
    total = float(logsumexp(lw))
    post = np.exp(lw - total)
    marg = np.array([sum(p for p, z in zip(post, configs, strict=True) if z[i] == 1) for i in range(3)])
    return total, marg


def enumerated_hmm() -> tuple[float, np.ndarray]:
    """log p(y) of the four-step HMM over all 16 paths, and P(z_t = 1 | y)."""
    model = _models.DISCRETE["hmm"]
    logs: list[float] = []
    configs = list(itertools.product([0, 1], repeat=4))
    for z in configs:
        ld, _ = log_density(model, (), {}, {f"z{t}": jnp.array(v) for t, v in enumerate(z)})
        logs.append(float(ld))
    lw = np.array(logs)
    total = float(logsumexp(lw))
    post = np.exp(lw - total)
    marg = np.array([sum(p for p, z in zip(post, configs, strict=True) if z[t] == 1) for t in range(4)])
    return total, marg


def cases() -> list[Case]:
    out: list[Case] = []
    for name in MODELS:
        model = program(name)
        for i, seed in enumerate([3, 17]):
            params = latent_draws(name, seed)
            ld, _ = log_density(model, (), {}, params)
            out.append(
                case(f"logDensity/{name}/{i}", "logDensity", [params], stable(float(ld)), rtol=1e-12, model=name)
            )
            u = as_numpy(unconstrain_fn(model, (), {}, params))
            out.append(case(f"unconstrain/{name}/{i}", "unconstrain", [params], u, rtol=1e-10, atol=1e-12, model=name))
            pe = float(potential_energy(model, (), {}, u))
            out.append(case(f"potential/{name}/{i}", "potential", [u], stable(pe), rtol=1e-11, model=name))
            c = as_numpy(constrain_fn(model, (), {}, u, return_deterministic=True))
            out.append(case(f"constrain/{name}/{i}", "constrain", [u], c, rtol=1e-11, atol=1e-13, model=name))
    for i, locs in enumerate([np.array([-1.0, 1.5]), np.array([0.5, 2.5])]):
        total, marg = enumerated_mixture(locs)
        out.append(
            case(f"enumerated/mixture/{i}", "enumerated", [{"locs": locs}], stable(total), rtol=1e-11, model="mixture")
        )
        out.append(
            case(
                f"marginals/mixture/{i}",
                "marginals",
                [{"locs": locs}],
                marg,
                rtol=1e-10,
                atol=1e-13,
                model="mixture",
                site="z",
            )
        )
    total, marg = enumerated_hmm()
    out.append(case("enumerated/hmm", "enumerated", [{}], stable(total), rtol=1e-11, model="hmm"))
    out.append(case("marginals/hmm", "chainMarginals", [{}], stable(marg), rtol=1e-10, atol=1e-13, model="hmm"))
    return out
