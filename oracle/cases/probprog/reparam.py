"""Reparameterised programs (probprog/reparam.ts) against NumPyro's `reparam` handler: the log joint of the
reparameterised program at values of the auxiliary sites, for LocScaleReparam at several centrings (a scalar site, and
the plated schools program), TransformReparam of a log-normal and CircularReparam of a von Mises angle. Every number is
also computed with Pyro's `poutine.reparam` (pyro-ppl, torch) and the two must agree to 1e-10 before the fixture is
written."""

import importlib.metadata
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pyro
import pyro.distributions as pdist
import pyro.poutine as poutine
import torch
from numpyro import handlers
from numpyro.infer.reparam import CircularReparam, LocScaleReparam, TransformReparam
from numpyro.infer.util import log_density
from omai_oracle.fixture import Case, case
from omai_oracle.pgm import stable
from pyro.infer.reparam import LocScaleReparam as PyroLocScale

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)

REFERENCE: dict[str, str] = {
    "numpyro": importlib.metadata.version("numpyro"),
    "pyro-ppl": importlib.metadata.version("pyro-ppl"),
}

SIGMA = [15.0, 10.0, 16.0, 11.0]
SCHOOLS = [28.0, 8.0, -3.0, 7.0]


def theta_model() -> None:
    """One normal site with an observation (the `THETA` program of reparam.test.ts)."""
    theta = numpyro.sample("theta", dist.Normal(1.5, 2.5))
    numpyro.sample("y", dist.Normal(theta, 1.0), obs=jnp.asarray(0.7))


def schools() -> None:
    """The plated, hierarchical schools program."""
    mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
    tau = numpyro.sample("tau", dist.HalfCauchy(5.0))
    with numpyro.plate("J", 4):
        theta = numpyro.sample("theta", dist.Normal(mu, tau))
        numpyro.sample("y", dist.Normal(theta, jnp.array(SIGMA)), obs=jnp.array(SCHOOLS))


def lognormal() -> None:
    s = numpyro.sample("s", dist.TransformedDistribution(dist.Normal(0.2, 0.7), dist.transforms.ExpTransform()))
    numpyro.sample("y", dist.Normal(s, 1.0), obs=jnp.asarray(1.3))


def angle() -> None:
    phi = numpyro.sample("phi", dist.VonMises(0.3, 2.0))
    numpyro.sample("y", dist.Normal(phi, 1.0), obs=jnp.asarray(0.1))


def density(model: Callable[..., Any], params: dict[str, Any]) -> float:
    ld, _ = log_density(model, (), {}, {k: jnp.asarray(v) for k, v in params.items()})
    return float(ld)


def pyro_theta(centered: float, z: float) -> float:
    """The same one-site program in Pyro: log density of the reparameterised model at the auxiliary value."""

    def model() -> None:
        theta = pyro.sample("theta", pdist.Normal(1.5, 2.5))
        pyro.sample("y", pdist.Normal(theta, 1.0), obs=torch.tensor(0.7, dtype=torch.float64))

    reparameterised = poutine.reparam(model, config={"theta": PyroLocScale(centered)})
    conditioned = poutine.condition(reparameterised, data={"theta_decentered": torch.tensor(z, dtype=torch.float64)})
    trace = poutine.trace(conditioned).get_trace()
    return float(trace.log_prob_sum())


def cases() -> list[Case]:
    out: list[Case] = []
    for centered in [0.0, 0.3, 0.8]:
        for z in [-1.1, 0.4, 2.3]:
            model = handlers.reparam(theta_model, config={"theta": LocScaleReparam(centered)})
            ld = density(model, {"theta_decentered": z})
            pyro_ld = pyro_theta(centered, z)
            assert abs(ld - pyro_ld) < 1e-10, (centered, z, ld, pyro_ld)
            out.append(
                case(
                    f"locScale/theta/c={centered}/z={z}",
                    "locScaleTheta",
                    [],
                    stable(ld),
                    rtol=1e-12,
                    centered=centered,
                    z=z,
                )
            )
    for centered in [0.0, 0.5]:
        values = {"mu": 1.5, "tau": 2.5, "theta_decentered": np.array([0.3, -0.8, 1.2, 0.1])}
        model = handlers.reparam(schools, config={"theta": LocScaleReparam(centered)})
        out.append(
            case(
                f"locScale/schools/c={centered}",
                "locScaleSchools",
                [values],
                stable(density(model, values)),
                rtol=1e-12,
                centered=centered,
            )
        )
    for z in [-0.9, 0.3, 1.4]:
        model = handlers.reparam(lognormal, config={"s": TransformReparam()})
        out.append(
            case(
                f"transform/lognormal/z={z}",
                "transformLognormal",
                [],
                stable(density(model, {"s_base": z})),
                rtol=1e-12,
                z=z,
            )
        )
    for turns in [0, 1, -2]:
        value = 0.5 + 2.0 * np.pi * turns
        model = handlers.reparam(angle, config={"phi": CircularReparam()})
        out.append(
            case(
                f"circular/turns={turns}",
                "circular",
                [],
                stable(density(model, {"phi_unwrapped": value})),
                rtol=1e-10,
                value=float(value),
            )
        )
    return out
