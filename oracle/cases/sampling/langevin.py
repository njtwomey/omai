"""One step of Stein variational gradient descent and the median bandwidth against BlackJAX (float64)."""

import importlib.metadata
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from blackjax.vi import svgd
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

jax.config.update("jax_enable_x64", True)

REFERENCE: dict[str, str] = {
    "blackjax": importlib.metadata.version("blackjax"),
    "optax": optax.__version__,
}

MEAN = jnp.array([1.0, -0.5])
PRECISION = jnp.array([[2.0, 0.6], [0.6, 1.0]])


def log_density(x: jax.Array) -> jax.Array:
    """A correlated Gaussian (the TypeScript test uses the same one)."""
    d = x - MEAN
    return -0.5 * d @ PRECISION @ d


def svgd_step(particles: FloatArray, length_scale: float, lr: float) -> FloatArray:
    """BlackJAX's SVGD kernel with plain gradient steps of size lr and a fixed RBF bandwidth."""
    optimizer = optax.sgd(lr)
    state = svgd.init(jnp.asarray(particles), {"length_scale": length_scale}, optimizer)
    step: Any = svgd.build_kernel(optimizer)
    grad = jax.grad(log_density)
    # BlackJAX passes the state's kernel parameters (the length scale) to the kernel as keywords.
    new_state = step(state, grad, svgd.rbf_kernel)
    return np.asarray(new_state.particles, dtype=np.float64)


def cases() -> list[Case]:
    g = rng(17)
    out: list[Case] = []
    for n in [2, 5, 12]:
        particles = g.normal(size=(n, 2)) * 1.5
        params: Any = svgd.median_heuristic({"length_scale": 1.0}, jnp.asarray(particles))
        h = float(params["length_scale"])
        out.append(case(f"medianBandwidth/{n}", "medianBandwidth", [particles], np.array(h)))
        for lr in [0.05, 0.5]:
            moved = svgd_step(particles, h, lr)
            out.append(
                case(f"svgdStep/{n}/{lr}", "svgdStep", [particles], moved, rtol=1e-12, lengthScale=h, stepSize=lr)
            )
    return out
