"""Regularisation penalties against numpy.linalg and scipy.stats, and their definitions (sklearn's elastic net)."""

import numpy as np
import scipy
import scipy.special as sp
import scipy.stats as st
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(49)
    w = g.normal(size=7)
    m = g.normal(size=(4, 3))
    groups = np.array([0.0, 0, 1, 1, 1, 2, 2])
    sizes = np.array([2, 3, 2])
    norms = np.array([np.linalg.norm(w[groups == k]) for k in range(3)])
    logits = g.normal(size=(5, 4))
    probs = sp.softmax(logits, axis=-1)
    return [
        case("l1Penalty", "l1Penalty", [w, w], np.asarray(0.3 * np.abs(w).sum()), strength=0.3),
        case("l2Penalty", "l2Penalty", [w, w], np.asarray(0.5 * 0.3 * (w**2).sum()), strength=0.3),
        case(
            "elasticNet",
            "elasticNet",
            [w, w],
            np.asarray(0.7 * (0.2 * np.abs(w).sum() + 0.5 * 0.8 * (w**2).sum())),
            alpha=0.7,
            l1Ratio=0.2,
        ),
        case("groupLasso", "groupLasso", [w, groups], np.asarray(0.5 * (np.sqrt(sizes) * norms).sum()), strength=0.5),
        case(
            "groupLasso/sparse",
            "groupLasso",
            [w, groups],
            np.asarray(0.75 * (np.sqrt(sizes) * norms).sum() + 0.25 * np.abs(w).sum()),
            l1Ratio=0.25,
        ),
        case("nuclearNorm", "nuclearNorm", [m, m], np.asarray(np.linalg.norm(m, "nuc")), rtol=1e-10),
        case("orthogonality", "orthogonality", [m, m], np.asarray(((m.T @ m - np.eye(3)) ** 2).sum())),
        case(
            "orthogonality/double",
            "orthogonality",
            [m, m],
            np.asarray(((m.T @ m - np.eye(3)) ** 2).sum() + ((m @ m.T - np.eye(4)) ** 2).sum()),
            double=True,
        ),
        case("entropyPenalty", "entropyPenalty", [logits, logits], np.asarray(-st.entropy(probs, axis=-1).mean())),
        case(
            "entropyPenalty/probs",
            "entropyPenalty",
            [probs, probs],
            np.asarray(-0.1 * st.entropy(probs, axis=-1).mean()),
            strength=0.1,
            logits=False,
        ),
    ]
