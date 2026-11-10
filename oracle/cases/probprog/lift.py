"""The `lift` handler (probprog/lift.ts) against NumPyro's `handlers.lift` and Pyro's `poutine.lift`: the log joint of a
program whose parameters are lifted to random variables with priors."""

import importlib.metadata
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pyro
import pyro.distributions as pdist
import pyro.poutine as poutine
import torch
from numpyro import handlers
from numpyro.infer.util import log_density
from omai_oracle.fixture import Case, case
from omai_oracle.pgm import stable

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)

REFERENCE: dict[str, str] = {
    "numpyro": importlib.metadata.version("numpyro"),
    "pyro-ppl": importlib.metadata.version("pyro-ppl"),
}


def program() -> None:
    s = numpyro.param("s", jnp.asarray(1.0), constraint=dist.constraints.positive)
    numpyro.sample("y", dist.Normal(0.0, s), obs=jnp.asarray(0.5))


def density(model: Callable[..., Any], params: dict[str, Any]) -> float:
    ld, _ = log_density(model, (), {}, {k: jnp.asarray(v) for k, v in params.items()})
    return float(ld)


def pyro_density(rate: float, s: float) -> float:
    def model() -> None:
        scale = pyro.param("s", torch.tensor(1.0, dtype=torch.float64), constraint=pdist.constraints.positive)
        pyro.sample("y", pdist.Normal(0.0, scale), obs=torch.tensor(0.5, dtype=torch.float64))

    lifted = poutine.lift(model, prior={"s": pdist.Exponential(torch.tensor(rate, dtype=torch.float64))})
    conditioned = poutine.condition(lifted, data={"s": torch.tensor(s, dtype=torch.float64)})
    return float(poutine.trace(conditioned).get_trace().log_prob_sum())


def cases() -> list[Case]:
    out: list[Case] = []
    pyro.clear_param_store()
    for rate in [0.5, 2.0]:
        for s in [0.4, 1.0, 2.2]:
            lifted = handlers.lift(program, prior={"s": dist.Exponential(rate)})
            ld = density(lifted, {"s": s})
            other = pyro_density(rate, s)
            assert abs(ld - other) < 1e-10, (rate, s, ld, other)
            out.append(case(f"lift/exponential/rate={rate}/s={s}", "lift", [], stable(ld), rtol=1e-12, rate=rate, s=s))
    return out
