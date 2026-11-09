"""Latent Dirichlet allocation by variational Bayes (batch and online) from fixed initial topics against scikit-learn.

Scikit-learn draws the initial topic-word parameters and each training E-step's document starts at random, so the
reference runs with a random state that returns the given initial parameters for the topics and ones for the
documents; the engine takes the same initial parameters and a constant document start.
"""

from importlib.metadata import version
from typing import Any

import numpy as np
from numpy.typing import NDArray
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.decomposition import LatentDirichletAllocation

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

# Scikit-learn's digamma (Cython, "optimised for speed, not accuracy") is good to about 1e-8, so parity is to 1e-6.
TOL = {"rtol": 1e-6, "atol": 1e-7}


class FixedState(np.random.RandomState):
    """A random state whose gamma draws are the given topics (shape K x V) or ones (the documents' constant start)."""

    def __init__(self, components: NDArray[np.float64]) -> None:
        super().__init__(0)
        self._components = components

    def gamma(self, shape: Any, scale: Any = 1.0, size: Any = None) -> Any:  # noqa: ANN401
        if tuple(size) == self._components.shape:
            return self._components.copy()
        return np.full(size, shape * scale)


def corpus(n: int, v: int, k: int, seed: int) -> NDArray[np.float64]:
    """Counts from k planted topics over v words."""
    g = rng(seed)
    topics = g.dirichlet(np.full(v, 0.2), k)
    theta = g.dirichlet(np.full(k, 0.3), n)
    lengths = g.integers(15, 40, n)
    x = np.zeros((n, v))
    for i in range(n):
        x[i] = g.multinomial(lengths[i], theta[i] @ topics)
    return x


def cases() -> list[Case]:
    out: list[Case] = []
    n, v, k = 24, 12, 3
    x = corpus(n, v, k, 5)
    g = rng(6)
    start = g.gamma(100.0, 0.01, (k, v))
    held = corpus(6, v, k, 7)
    settings: list[dict[str, Any]] = [
        {"name": "batch", "learning_method": "batch", "max_iter": 12, "batch_size": 128},
        {
            "name": "online",
            "learning_method": "online",
            "max_iter": 4,
            "batch_size": 7,
            "learning_offset": 5.0,
            "learning_decay": 0.8,
        },
        {
            "name": "priors",
            "learning_method": "batch",
            "max_iter": 6,
            "batch_size": 128,
            "doc_topic_prior": 0.2,
            "topic_word_prior": 0.05,
        },
    ]
    for s in settings:
        options: dict[str, Any] = {
            "n_components": k,
            "mean_change_tol": 1e-6,
            "max_doc_update_iter": 200,
            "learning_method": s["learning_method"],
            "max_iter": s["max_iter"],
            "batch_size": s["batch_size"],
            "learning_offset": s.get("learning_offset", 10.0),
            "learning_decay": s.get("learning_decay", 0.7),
            "doc_topic_prior": s.get("doc_topic_prior"),
            "topic_word_prior": s.get("topic_word_prior"),
            "random_state": FixedState(start),
        }
        model = LatentDirichletAllocation(**options)
        model.fit(x)
        name = str(s["name"])
        knobs: dict[str, Any] = {key: val for key, val in s.items() if key != "name"}
        inputs = [x, start, held]
        out.append(case(f"{name}/components", "components", inputs, np.asarray(model.components_), **TOL, **knobs))
        out.append(case(f"{name}/doc_topic", "docTopic", inputs, model.transform(x), **TOL, **knobs))
        out.append(case(f"{name}/transform_held", "transformHeld", inputs, model.transform(held), **TOL, **knobs))
        out.append(case(f"{name}/perplexity", "perplexity", inputs, np.float64(model.bound_), **TOL, **knobs))
        out.append(
            case(
                f"{name}/perplexity_held", "perplexityHeld", inputs, np.float64(model.perplexity(held)), **TOL, **knobs
            )
        )
        out.append(case(f"{name}/score", "score", inputs, np.float64(model.score(x)), **TOL, **knobs))
        out.append(case(f"{name}/score_held", "scoreHeld", inputs, np.float64(model.score(held)), **TOL, **knobs))
    return out
