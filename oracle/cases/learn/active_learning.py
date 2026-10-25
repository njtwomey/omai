"""Active-learning queries against modAL: least-confidence, margin and entropy scores and selections, ranked batch
sampling, and query-by-committee (vote entropy, consensus entropy, maximum disagreement) for naive Bayes models."""

from typing import Any

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from modAL.batch import uncertainty_batch_sampling
from modAL.disagreement import (
    consensus_entropy_sampling,
    max_disagreement_sampling,
    vote_entropy_sampling,
)
from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import (
    classifier_entropy,
    classifier_margin,
    classifier_uncertainty,
    entropy_sampling,
    margin_sampling,
    uncertainty_sampling,
)
from sklearn.naive_bayes import GaussianNB

REFERENCE: dict[str, str] = {"modAL-python": "0.4.2.1"}


def indices(selected: Any) -> np.ndarray[Any, np.dtype[np.float64]]:  # noqa: ANN401  (modAL returns untyped tuples)
    """The selected pool indices (modAL returns (indices, rows) or indices)."""
    idx = selected[0] if isinstance(selected, tuple) else selected
    return np.asarray(idx, dtype=np.float64)


def cases() -> list[Case]:
    g = rng(4347)
    labeled = np.vstack(
        [g.normal(size=(6, 2)) + [-1, 0], g.normal(size=(6, 2)) + [1, 0], g.normal(size=(6, 2)) + [0, 2]]
    )
    y = np.array([0.0] * 6 + [1.0] * 6 + [2.0] * 6)
    pool = g.normal(size=(25, 2)) * 1.5
    out: list[Case] = []
    learner = ActiveLearner(estimator=GaussianNB(), X_training=labeled, y_training=y)
    scores = {
        "uncertainty": classifier_uncertainty(learner, pool),
        "margin": classifier_margin(learner, pool),
        "entropy": classifier_entropy(learner, pool),
    }
    out.append(case("scores", "scores", [labeled, y, pool], scores, rtol=1e-10, atol=1e-12))
    for name, fn in [("uncertainty", uncertainty_sampling), ("margin", margin_sampling), ("entropy", entropy_sampling)]:
        sel = np.sort(indices(fn(learner, pool, n_instances=3)))
        out.append(case(f"query/{name}", "query", [labeled, y, pool], {"indices": sel}, strategy=name, n=3))
    batch = indices(uncertainty_batch_sampling(learner, pool, n_instances=4))
    out.append(
        case("query/ranked-batch", "query", [labeled, y, pool], {"indices": batch}, strategy="ranked-batch", n=4)
    )
    subsets = [np.arange(0, 18, 2), np.arange(1, 18, 2), np.r_[0:4, 6:10, 12:16]]
    members = [ActiveLearner(estimator=GaussianNB(), X_training=labeled[s], y_training=y[s]) for s in subsets]
    committee = Committee(learner_list=members)
    masks = np.array([np.isin(np.arange(18), s) for s in subsets], dtype=np.float64)
    for name, fn in [
        ("vote-entropy", vote_entropy_sampling),
        ("consensus-entropy", consensus_entropy_sampling),
        ("max-disagreement", max_disagreement_sampling),
    ]:
        sel = np.sort(indices(fn(committee, pool, n_instances=3)))
        out.append(
            case(f"committee/{name}", "committee", [labeled, y, pool, masks], {"indices": sel}, strategy=name, n=3)
        )
    return out
