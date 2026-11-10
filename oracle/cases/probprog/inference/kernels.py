"""The kernel methods of probprog/inference (the ensemble, Langevin, Barker, microcanonical and stochastic-gradient
samplers) against the samplers of NumPyro (AIES, ESS) and BlackJAX (MALA, Barker, MCLMC, SGLD, SGHMC) on the same
targets: the two real-valued programs `mvn` and `logistic` of `_models.py`. A fixture is a summary of long chains, not
draws: per coordinate the posterior mean and standard deviation, the effective sample size (NumPyro's) and the Monte
Carlo standard errors, so the TypeScript test compares its own chains' summaries within the standard errors of both."""

import importlib.metadata
import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import handlers
from numpyro.diagnostics import effective_sample_size
from numpyro.infer import AIES, ESS, MCMC
from numpyro.infer.util import potential_energy, unconstrain_fn
from omai_oracle.fixture import Case, case
from omai_oracle.pgm import stable

_spec = importlib.util.spec_from_file_location("probprog_models", Path(__file__).parent.parent / "_models.py")
assert _spec is not None and _spec.loader is not None
_models: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models)
MODELS = cast(dict[str, Callable[[], None]], _models.MODELS)

REFERENCE: dict[str, str] = {
    "numpyro": importlib.metadata.version("numpyro"),
    "blackjax": importlib.metadata.version("blackjax"),
    "jax": importlib.metadata.version("jax"),
}

NAMES = ["mvn", "logistic"]
NUM_CHAINS = 4
NUM_DRAWS = 4000
NUM_WARMUP = 1000

Array = jax.Array
LogDensity = Callable[[Array], Array]
# BlackJAX is loosely typed: its samplers are used through this untyped handle.
BJ: Any = blackjax


def position_of(state: Any) -> Array:  # noqa: ANN401
    """The position of a BlackJAX state."""
    return cast(Array, state.position)


def identity(x: object) -> Array:
    """A state that is its own position."""
    return cast(Array, x)


def layout(model: Callable[[], None]) -> list[tuple[str, tuple[int, ...]]]:
    """Latent sites in execution order with their unconstrained shapes."""
    tr = handlers.trace(handlers.seed(model, 0)).get_trace()
    latent = {k: v["value"] for k, v in tr.items() if v["type"] == "sample" and not v["is_observed"]}
    u = unconstrain_fn(model, (), {}, latent)
    return [(k, tuple(np.shape(u[k]))) for k in latent]


def logdensity_of(model: Callable[[], None]) -> tuple[LogDensity, int]:
    """The log density of the program on its unconstrained coordinates, flattened in execution order."""
    sites = layout(model)
    sizes = [int(np.prod(s)) for _, s in sites]

    def unpack(x: Array) -> dict[str, Array]:
        out: dict[str, Array] = {}
        at = 0
        for (k, shape), n in zip(sites, sizes, strict=True):
            out[k] = jnp.reshape(x[at : at + n], shape)
            at += n
        return out

    def logdensity(x: Array) -> Array:
        return -jnp.asarray(potential_energy(model, (), {}, unpack(x)))

    return logdensity, sum(sizes)


def summarise(draws: np.ndarray, name: str, sampler: str) -> Case:
    """Mean, sd, effective sample size and Monte Carlo standard errors per coordinate of `[chain, draw, dim]` draws."""
    mean = draws.mean(axis=(0, 1))
    sd = draws.std(axis=(0, 1), ddof=1)
    ess = np.asarray(effective_sample_size(draws), dtype=np.float64)
    mcse_mean = sd / np.sqrt(ess)
    mcse_sd = sd / np.sqrt(2.0 * ess)
    out = np.stack([mean, sd, ess, mcse_mean, mcse_sd])
    return case(f"{sampler}/{name}", "kernelMoments", [], stable(out), rtol=0.0, atol=0.0, model=name, sampler=sampler)


def scan_chains(
    step: Callable[..., tuple[object, Array]],
    init: Callable[[Array], object],
    starts: Array,
    key: Array,
    num_draws: int,
) -> np.ndarray:
    """Run `NUM_CHAINS` chains of a step function (`step(key, state) -> (state, position)`) from the starts."""
    keys = jax.random.split(key, NUM_CHAINS)

    def one(k: Array, x0: Array) -> Array:
        def body(state: object, kk: Array) -> tuple[object, Array]:
            return step(kk, state)

        _, xs = jax.lax.scan(body, init(x0), jax.random.split(k, num_draws))
        return xs

    return np.asarray(jax.vmap(one)(keys, starts), dtype=np.float64)


def blackjax_cases(name: str, logdensity: LogDensity, dim: int) -> list[Case]:
    """MALA, Barker, SGLD and SGHMC of BlackJAX on four chains."""
    out: list[Case] = []
    key = jax.random.key(0)
    starts = jax.random.normal(jax.random.key(1), (NUM_CHAINS, dim))
    burn = NUM_WARMUP

    def run(
        kernel_step: Callable[..., tuple[object, object]], init: Callable[[Array], object], pos: Callable[..., Array]
    ) -> np.ndarray:
        def step(k: Array, state: object) -> tuple[object, Array]:
            new, _ = kernel_step(k, state)
            return new, pos(new)

        draws = scan_chains(step, init, starts, key, burn + NUM_DRAWS)
        return draws[:, burn:, :]

    mala = BJ.mala(logdensity, 0.25)
    out.append(summarise(run(mala.step, mala.init, position_of), name, "mala"))
    barker = BJ.barker(logdensity, 1.0)
    out.append(summarise(run(barker.step, barker.init, position_of), name, "barker"))
    # SGLD and SGHMC with the exact gradient (the minibatch is a placeholder): their step-size bias is compared.
    grad = jax.grad(logdensity)

    def grad_estimator(position: Array, _batch: object) -> Array:
        return grad(position)

    sgld = BJ.sgld(grad_estimator)

    def sgld_step(k: Array, x: Array) -> tuple[Array, None]:
        return sgld.step(k, x, 0.0, 0.01), None

    out.append(summarise(run(sgld_step, identity, identity), name, "sgld"))
    sghmc = BJ.sghmc(grad_estimator, num_integration_steps=10, alpha=0.5)

    def sghmc_step(k: Array, x: Array) -> tuple[Array, None]:
        return sghmc.step(k, x, 0.0, 0.05), None

    out.append(summarise(run(sghmc_step, identity, identity), name, "sghmc"))
    return out


def mclmc_case(name: str, logdensity: LogDensity, dim: int) -> Case:
    """MCLMC tuned by BlackJAX's `mclmc_find_L_and_step_size`, then run on four chains."""
    key = jax.random.key(2)
    tune_key, run_key, init_key = jax.random.split(key, 3)
    kernel = BJ.mcmc.mclmc.build_kernel(integrator=BJ.mcmc.integrators.isokinetic_mclachlan)
    state0 = BJ.mcmc.mclmc.init(jax.random.normal(init_key, (dim,)), logdensity, init_key)
    state, params, _ = BJ.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel, num_steps=NUM_DRAWS + NUM_WARMUP, state=state0, rng_key=tune_key, logdensity_fn=logdensity
    )
    sampler = BJ.mclmc(
        logdensity, L=params.L, step_size=params.step_size, inverse_mass_matrix=params.inverse_mass_matrix
    )
    keys = jax.random.split(run_key, NUM_CHAINS)

    def one(k: Array) -> Array:
        k_init, k_run = jax.random.split(k)
        s = sampler.init(state.position + 0.01 * jax.random.normal(k_init, (dim,)), k_init)

        def body(carry: object, kk: Array) -> tuple[object, Array]:
            new, _ = sampler.step(kk, carry)
            return new, new.position

        _, xs = jax.lax.scan(body, s, jax.random.split(k_run, NUM_DRAWS))
        return xs

    return summarise(np.asarray(jax.vmap(one)(keys), dtype=np.float64), name, "mclmc")


def ensemble_cases(name: str, model: Callable[[], None]) -> list[Case]:
    """NumPyro's ensemble samplers: every walker is a chain."""
    out: list[Case] = []
    for sampler_name, kernel in (("aies", AIES(model)), ("ess", ESS(model))):
        mcmc = MCMC(
            kernel,
            num_warmup=NUM_WARMUP,
            num_samples=NUM_DRAWS,
            num_chains=16,
            chain_method="vectorized",
            progress_bar=False,
        )
        mcmc.run(jax.random.key(3))
        grouped = mcmc.get_samples(group_by_chain=True)
        names = [k for k, _ in layout(model)]
        columns = [np.asarray(grouped[k], dtype=np.float64).reshape(16, NUM_DRAWS, -1) for k in names]
        out.append(summarise(np.concatenate(columns, axis=2), name, sampler_name))
    return out


def cases() -> list[Case]:
    numpyro.enable_x64()
    out: list[Case] = []
    for name in NAMES:
        model = MODELS[name]
        logdensity, dim = logdensity_of(model)
        out.extend(ensemble_cases(name, model))
        out.extend(blackjax_cases(name, logdensity, dim))
        out.append(mclmc_case(name, logdensity, dim))
    return out
