"""NumPyro versions of the probabilistic programs the probprog golden tests share (the TypeScript tests define the same
programs under the same names, with the same data)."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers

jax.config.update("jax_enable_x64", True)

Y4 = [0.3, -1.2, 2.1, 0.8]
SIGMA = [15.0, 10.0, 16.0, 11.0]
SCHOOLS = [28.0, 8.0, -3.0, 7.0]
X = [[0.5, -1.0], [1.2, 0.3], [-0.7, 0.8], [0.1, -0.4], [1.5, 1.1]]
YB = [1.0, 0.0, 0.0, 1.0, 1.0]
COUNTS = [3.0, 5.0, 4.0, 6.0]
CATS = [0.0, 2.0, 2.0, 1.0, 2.0]
FLIPS = [1.0, 0.0, 1.0, 1.0]
Y3 = [0.5, 1.5, -0.5]
MASK3 = [True, False, True]
Y10 = [0.2, -0.4, 1.1, 0.7, -1.3, 0.5, 2.0, -0.1, 0.9, 0.3]
YN = [[0.1, -0.2], [0.4, 0.9], [-1.0, 0.3]]
SIMPLEX_DATA = [3.0, 0.0, 2.0, 3.0, 1.0, 3.0]
MIX_W = [0.3, 0.7]
MIX_Y = [-1.1, 0.4, 2.3]
HMM_T = [[0.8, 0.2], [0.3, 0.7]]
HMM_Y = [0.2, 1.7, 2.2, -0.3]


def normal_mean() -> None:
    mu = numpyro.sample("mu", dist.Normal(0.0, 2.0))
    with numpyro.plate("data", 4):
        numpyro.sample("y", dist.Normal(mu, 1.5), obs=jnp.array(Y4))


def schools() -> None:
    mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
    tau = numpyro.sample("tau", dist.HalfCauchy(5.0))
    with numpyro.plate("J", 4):
        theta = numpyro.sample("theta", dist.Normal(mu, tau))
        numpyro.sample("y", dist.Normal(theta, jnp.array(SIGMA)), obs=jnp.array(SCHOOLS))


def logistic() -> None:
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
    with numpyro.plate("features", 2):
        beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
    logits = alpha + jnp.array(X) @ beta
    with numpyro.plate("data", 5):
        numpyro.sample("y", dist.Bernoulli(logits=logits), obs=jnp.array(YB))


def gamma_poisson() -> None:
    rate = numpyro.sample("rate", dist.Gamma(2.0, 0.5))
    numpyro.deterministic("double", 2.0 * rate)
    with numpyro.plate("data", 4):
        numpyro.sample("k", dist.Poisson(rate), obs=jnp.array(COUNTS))


def dirichlet_categorical() -> None:
    p = numpyro.sample("p", dist.Dirichlet(jnp.array([1.0, 2.0, 3.0])))
    with numpyro.plate("data", 5):
        numpyro.sample("z", dist.CategoricalProbs(p), obs=jnp.array(CATS, dtype=jnp.int32))


def beta_factor() -> None:
    p = numpyro.sample("p", dist.Beta(2.0, 3.0))
    numpyro.factor("penalty", -(p**2))
    with numpyro.plate("data", 4):
        numpyro.sample("y", dist.Bernoulli(probs=p), obs=jnp.array(FLIPS))


def scale_mask() -> None:
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with handlers.scale(scale=2.0), numpyro.plate("data", 3), handlers.mask(mask=jnp.array(MASK3)):
        numpyro.sample("y", dist.Normal(mu, 1.0), obs=jnp.array(Y3))


def subsampled() -> None:
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with numpyro.plate("data", 10, subsample_size=3) as idx:
        numpyro.sample("y", dist.Normal(mu, 1.0), obs=jnp.array(Y10)[idx])


def nested() -> None:
    with numpyro.plate("a", 2), numpyro.plate("b", 3):
        x = numpyro.sample("x", dist.Normal(0.0, 1.0))
        numpyro.sample("y", dist.Normal(x, 0.5), obs=jnp.array(YN))


def bounded() -> None:
    u = numpyro.sample("u", dist.Uniform(-1.0, 4.0))
    s = numpyro.sample("s", dist.LogNormal(0.0, 1.0))
    numpyro.sample("y", dist.Normal(u, s), obs=1.3)


def simplex() -> None:
    w = numpyro.sample("w", dist.Dirichlet(jnp.array([2.0, 1.0, 1.5, 3.0])))
    with numpyro.plate("data", 6):
        numpyro.sample("z", dist.CategoricalProbs(w), obs=jnp.array(SIMPLEX_DATA, dtype=jnp.int32))


def mvn() -> None:
    x = jnp.asarray(
        numpyro.sample(
            "x", dist.MultivariateNormal(jnp.array([0.0, 1.0]), covariance_matrix=jnp.array([[1.0, 0.3], [0.3, 2.0]]))
        )
    )
    numpyro.sample("y", dist.Normal(x[0] + 2.0 * x[1], 1.0), obs=0.7)


def mixture() -> None:
    locs = jnp.asarray(numpyro.sample("locs", dist.Independent(dist.Normal(jnp.zeros(2), 3.0), 1)))
    with numpyro.plate("data", 3):
        z = jnp.asarray(numpyro.sample("z", dist.CategoricalProbs(jnp.array(MIX_W))))
        numpyro.sample("y", dist.Normal(locs[z], 1.0), obs=jnp.array(MIX_Y))


def hmm() -> None:
    prev: jax.Array | None = None
    for t, y in enumerate(HMM_Y):
        probs = jnp.array([0.5, 0.5]) if prev is None else jnp.array(HMM_T)[prev]
        z = jnp.asarray(numpyro.sample(f"z{t}", dist.CategoricalProbs(probs)))
        numpyro.sample(f"y{t}", dist.Normal(jnp.array([0.0, 2.0])[z], 1.0), obs=y)
        prev = z


MODELS: dict[str, Callable[[], None]] = {
    "normalMean": normal_mean,
    "schools": schools,
    "logistic": logistic,
    "gammaPoisson": gamma_poisson,
    "dirichletCategorical": dirichlet_categorical,
    "betaFactor": beta_factor,
    "scaleMask": scale_mask,
    "subsampled": subsampled,
    "nested": nested,
    "bounded": bounded,
    "simplex": simplex,
    "mvn": mvn,
}

DISCRETE: dict[str, Callable[[], None]] = {"mixture": mixture, "hmm": hmm}
