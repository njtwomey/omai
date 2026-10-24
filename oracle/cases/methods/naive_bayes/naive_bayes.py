"""Naive Bayes against scikit-learn: joint log-likelihoods of the Gaussian, multinomial, Bernoulli, complement and
categorical variants."""

import numpy as np
import sklearn
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(71)
    y = (np.arange(60) % 3).astype(np.float64)
    gx = g.normal(size=(60, 3)) + y[:, None] * np.array([0.8, -0.5, 0.2])
    counts = g.poisson(1.0 + y[:, None] * np.array([1.0, 0.0, 2.0, 0.5]), size=(60, 4)).astype(np.float64)
    cats = np.minimum(counts, 3)
    out: list[Case] = []
    gnb = GaussianNB().fit(gx, y)
    out.append(
        case(
            "gaussian",
            "nb",
            [gx, y],
            {"jll": gnb.predict_joint_log_proba(gx[:6])},
            rtol=1e-10,
            atol=1e-10,
            kind="gaussian",
        )
    )
    for kind, model, x in [
        ("multinomial", MultinomialNB(alpha=0.5), counts),
        ("bernoulli", BernoulliNB(alpha=1.0, binarize=1.0), counts),
        ("complement", ComplementNB(alpha=1.0), counts),
        ("categorical", CategoricalNB(alpha=1.0), cats),
    ]:
        model.fit(x, y)
        out.append(
            case(
                kind,
                "nb",
                [x, y],
                {"jll": model.predict_joint_log_proba(x[:6]), "proba": model.predict_proba(x[:6])},
                rtol=1e-10,
                atol=1e-10,
                kind=kind,
                alpha=float(model.alpha),
            )
        )
    return out
