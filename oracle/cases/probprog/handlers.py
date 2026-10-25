"""Effect-handler semantics (probprog/handlers.ts) against NumPyro's handlers: log densities under condition,
substitute, scale, mask and do; which sites block hides and do records; trace shapes and plate dimensions; and the
sites config_enumerate marks."""

import importlib.metadata
import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from mdml_oracle.fixture import Case, case
from mdml_oracle.pgm import stable
from numpyro import handlers
from numpyro.infer.util import log_density

_spec = importlib.util.spec_from_file_location("probprog_models", Path(__file__).with_name("_models.py"))
assert _spec is not None and _spec.loader is not None
_models: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models)
MODELS = cast(dict[str, Callable[[], None]], _models.MODELS)

REFERENCE: dict[str, str] = {"numpyro": importlib.metadata.version("numpyro")}

SCHOOL_PARAMS: dict[str, Any] = {
    "mu": np.array(1.5),
    "tau": np.array(2.5),
    "theta": np.array([3.0, -1.0, 0.5, 2.0]),
}


def density(model: Callable[..., Any], params: dict[str, Any]) -> np.ndarray:
    ld, _ = log_density(model, (), {}, params)
    return stable(float(ld))


def shapes_model() -> None:
    """Sample shapes, nested plates with explicit and automatic dims, and a subsampled plate."""
    numpyro.sample("a", dist.Normal(0.0, 1.0), sample_shape=(3,))
    with numpyro.plate("outer", 2):
        numpyro.sample("b", dist.Normal(0.0, 1.0))
        with numpyro.plate("inner", 4):
            numpyro.sample("c", dist.Normal(jnp.zeros(1), 1.0))
    with numpyro.plate("rows", 5, dim=-2):
        numpyro.sample("d", dist.Dirichlet(jnp.ones(3)))
    with numpyro.plate("big", 10, subsample_size=4):
        numpyro.sample("e", dist.Exponential(1.0))


def mixed() -> None:
    """A program with discrete and continuous latent sites, for config_enumerate."""
    p = numpyro.sample("p", dist.Beta(2.0, 2.0))
    with numpyro.plate("n", 3):
        numpyro.sample("c", dist.Bernoulli(probs=p))
        numpyro.sample("k", dist.Poisson(2.0))
    numpyro.sample("d", dist.CategoricalProbs(jnp.array([0.2, 0.8])))


def enumerate_config(site: dict[str, Any]) -> dict[str, str]:
    """NumPyro's `config_enumerate` rule (without importing funsor): latent sites with enumerate support."""
    if site["type"] == "sample" and not site["is_observed"] and site["fn"].has_enumerate_support:
        return {"enumerate": "parallel"}
    return {}


def cases() -> list[Case]:
    out: list[Case] = []
    schools = MODELS["schools"]
    rest = {k: v for k, v in SCHOOL_PARAMS.items() if k != "tau"}
    out.append(
        case(
            "condition/schools",
            "condition",
            [rest],
            density(handlers.condition(schools, {"tau": 2.5}), rest),
            rtol=1e-12,
        )
    )
    tr = handlers.trace(handlers.seed(handlers.condition(schools, {"tau": 2.5}), 0)).get_trace()
    observed = np.array([1.0 if tr[k]["is_observed"] else 0.0 for k in ["mu", "tau", "theta", "y"]])
    out.append(case("condition/observed", "observedFlags", [], observed, site="tau"))
    sub = {"mu": SCHOOL_PARAMS["mu"]}
    out.append(
        case(
            "substitute/schools",
            "substitute",
            [rest | {"tau": SCHOOL_PARAMS["tau"]}],
            density(handlers.substitute(schools, sub), SCHOOL_PARAMS),
            rtol=1e-12,
        )
    )
    for factor in [0.5, 3.0]:
        out.append(
            case(
                f"scale/{factor}",
                "scale",
                [SCHOOL_PARAMS],
                density(handlers.scale(schools, scale=factor), SCHOOL_PARAMS),
                rtol=1e-12,
                factor=factor,
            )
        )
    mask = np.array([True, False, True, True])

    def masked() -> None:
        mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
        tau = numpyro.sample("tau", dist.HalfCauchy(5.0))
        with numpyro.plate("J", 4), handlers.mask(mask=jnp.asarray(mask)):
            theta = numpyro.sample("theta", dist.Normal(mu, tau))
            numpyro.sample("y", dist.Normal(theta, jnp.array(_models.SIGMA)), obs=jnp.array(_models.SCHOOLS))

    out.append(
        case("mask/plate", "mask", [SCHOOL_PARAMS], density(masked, SCHOOL_PARAMS), rtol=1e-12, mask=mask.tolist())
    )
    out.append(
        case(
            "mask/false",
            "maskAll",
            [SCHOOL_PARAMS],
            density(handlers.mask(schools, mask=False), SCHOOL_PARAMS),
            rtol=1e-12,
        )
    )
    out.append(
        case(
            "do/schools",
            "do",
            [SCHOOL_PARAMS],
            density(handlers.do(schools, {"mu": 4.0}), SCHOOL_PARAMS),
            rtol=1e-12,
        )
    )
    with handlers.seed(rng_seed=0):
        kept = list(handlers.trace(handlers.block(schools, hide=["mu", "theta"])).get_trace().keys())
        exposed = list(handlers.trace(handlers.block(schools, expose=["tau"])).get_trace().keys())
    out.append(case("block/hide", "blockHide", [], np.array([len(kept)]), sites=",".join(kept)))
    out.append(case("block/expose", "blockExpose", [], np.array([len(exposed)]), sites=",".join(exposed)))
    tr = handlers.trace(handlers.seed(shapes_model, 0)).get_trace()
    for name in ["a", "b", "c", "d", "e"]:
        site = tr[name]
        dims = [f.dim for f in site["cond_indep_stack"]]
        out.append(
            case(
                f"shape/{name}",
                "shape",
                [],
                np.array(site["value"].shape, dtype=np.int32),
                site=name,
                dims=",".join(str(d) for d in sorted(dims)),
                scale=float(site["scale"]) if site["scale"] is not None else 1.0,
            )
        )
    tr = handlers.trace(handlers.seed(handlers.infer_config(mixed, enumerate_config), 0)).get_trace()
    marked = [k for k, v in tr.items() if v["type"] == "sample" and v["infer"].get("enumerate") == "parallel"]
    out.append(case("enumerate/marks", "enumerateMarks", [], np.array([len(marked)]), sites=",".join(marked)))
    return out
