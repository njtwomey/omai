"""ELBO values (probprog/elbo.ts) at fixed guide parameters and fixed guide draws against NumPyro: `Trace_ELBO` and
`TraceMeanField_ELBO` with an `AutoNormal` guide whose draws are substituted (fixed reparameterised noise), and the
Renyi and importance-weighted bounds of fixed particles from NumPyro's log densities."""

import importlib.metadata
import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.infer import Trace_ELBO, TraceMeanField_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import log_density
from scipy.special import logsumexp
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.pgm import stable

_spec = importlib.util.spec_from_file_location("probprog_models", Path(__file__).with_name("_models.py"))
assert _spec is not None and _spec.loader is not None
_models: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models)
MODELS = cast(dict[str, Callable[[], None]], _models.MODELS)

REFERENCE: dict[str, str] = {
    "numpyro": importlib.metadata.version("numpyro"),
    "jax": importlib.metadata.version("jax"),
}

NAMES = ["normalMean", "schools", "gammaPoisson", "bounded", "simplex"]


def guide_params(guide: AutoNormal, g: np.random.Generator) -> dict[str, np.ndarray]:
    """Random locations and scales of the guide's parameters, with their shapes from an initialising run."""
    with handlers.seed(rng_seed=0):
        tr = handlers.trace(guide).get_trace()
    out: dict[str, np.ndarray] = {}
    for name, site in tr.items():
        if site["type"] != "param":
            continue
        shape = np.shape(site["value"])
        if name.endswith("_auto_scale"):
            out[name] = g.uniform(0.3, 1.2, size=shape)
        else:
            out[name] = g.normal(size=shape) * 0.7
    return out


def guide_draw(guide: AutoNormal, params: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    tr = handlers.trace(
        handlers.seed(handlers.substitute(guide, data={k: jnp.asarray(v) for k, v in params.items()}), seed)
    ).get_trace()
    return {k: stable(v["value"]) for k, v in tr.items() if v["type"] == "sample"}


def cases() -> list[Case]:
    g = rng(5)
    out: list[Case] = []
    key = jax.random.PRNGKey(0)
    for name in NAMES:
        model = MODELS[name]
        guide = AutoNormal(model)
        params = guide_params(guide, g)
        jp = {k: jnp.asarray(v) for k, v in params.items()}
        draws = [guide_draw(guide, params, s) for s in [1, 2, 3]]
        values = draws[0]
        fixed = handlers.substitute(guide, data={k: jnp.asarray(v) for k, v in values.items()})
        elbo = -float(np.asarray(Trace_ELBO().loss(key, jp, model, fixed)))
        out.append(case(f"traceElbo/{name}", "elboAt", [params, values], stable(elbo), rtol=1e-11, model=name))
        mf = -float(np.asarray(TraceMeanField_ELBO().loss(key, jp, model, fixed)))
        out.append(case(f"meanField/{name}", "meanField", [params, values], stable(mf), rtol=1e-11, model=name))
        logw = []
        for d in draws:
            lp, _ = log_density(model, (), {}, d)
            lq, _ = log_density(guide, (), {}, {**params, **d})
            logw.append(float(lp) - float(lq))
        w = np.array(logw)
        for alpha in [0.0, 0.5]:
            bound = float((logsumexp((1 - alpha) * w) - np.log(len(w))) / (1 - alpha))
            out.append(
                case(
                    f"renyi/{name}/{alpha}",
                    "renyi",
                    [params, *draws],
                    stable(bound),
                    rtol=1e-11,
                    model=name,
                    alpha=alpha,
                )
            )
    return out
