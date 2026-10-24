"""Entropies, divergences and mutual information of probability tables against scipy, sklearn.metrics and pyitlib."""

import importlib.metadata
from typing import Any

import numpy as np
import scipy
import scipy.spatial.distance as ssd
import scipy.special as sp
import scipy.stats as st
import sklearn
import sklearn.metrics as skm
from pyitlib import discrete_random_variable as drv  # pyright: ignore[reportMissingTypeStubs]
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

# pyitlib annotates `base` as int through its default; any positive base works, and e gives nats.
NATS: Any = np.e

REFERENCE: dict[str, str] = {
    "scipy": scipy.__version__,
    "scikit-learn": sklearn.__version__,
    "pyitlib": importlib.metadata.version("pyitlib"),
}


def samples_of(counts: FloatArray) -> FloatArray:
    """The samples (one row per variable) whose contingency table is `counts` (non-negative integers)."""
    cells = np.argwhere(np.ones(counts.shape, dtype=bool))
    reps = counts.ravel().astype(np.int64)
    return np.repeat(cells, reps, axis=0).T.astype(np.float64)


def probs(g: np.random.Generator, shape: tuple[int, ...], zeros: bool = False) -> FloatArray:
    p = g.uniform(0.05, 1.0, size=shape)
    if zeros:
        p[..., 0] = 0.0
    return p / p.sum(axis=-1, keepdims=True)


def cases() -> list[Case]:
    g = rng(400)
    out: list[Case] = []
    p, q = probs(g, (6,)), probs(g, (6,))
    pz = probs(g, (6,), zeros=True)
    batch_p, batch_q = probs(g, (3, 5)), probs(g, (3, 5))
    weights = g.uniform(0.1, 3.0, size=6)
    for base in [None, 2.0, 10.0]:
        kw: dict[str, float] = {} if base is None else {"base": base}
        tag = "e" if base is None else str(int(base))
        out.append(case(f"entropy/base={tag}", "entropy", [p], np.array(st.entropy(p, base=base)), **kw))
        out.append(
            case(f"relativeEntropy/base={tag}", "relativeEntropy", [p, q], np.array(st.entropy(p, q, base=base)), **kw)
        )
        js = ssd.jensenshannon(p, q, base=base)
        out.append(case(f"jensenShannonDistance/base={tag}", "jensenShannonDistance", [p, q], np.array(js), **kw))
        out.append(
            case(f"jensenShannonDivergence/base={tag}", "jensenShannonDivergence", [p, q], np.array(js**2), **kw)
        )
        cross = st.entropy(p, base=base) + st.entropy(p, q, base=base)
        out.append(case(f"crossEntropy/base={tag}", "crossEntropy", [p, q], np.array(cross), **kw))
    out.append(case("entropy/unnormalised", "entropy", [weights], np.array(st.entropy(weights))))
    out.append(case("entropy/zeros", "entropy", [pz], np.array(st.entropy(pz))))
    out.append(case("relativeEntropy/zeros", "relativeEntropy", [pz, q], np.array(st.entropy(pz, q))))
    out.append(case("relativeEntropy/infinite", "relativeEntropy", [q, pz], np.array(st.entropy(q, pz))))
    out.append(case("entropy/axis=-1", "entropy", [batch_p], st.entropy(batch_p, axis=-1), axis=-1))
    out.append(case("entropy/axis=0", "entropy", [batch_p.T], st.entropy(batch_p.T, axis=0), axis=0))
    out.append(
        case(
            "relativeEntropy/axis=-1",
            "relativeEntropy",
            [batch_p, batch_q],
            st.entropy(batch_p, batch_q, axis=-1),
            axis=-1,
        )
    )
    out.append(
        case(
            "jensenShannonDistance/axis=-1",
            "jensenShannonDistance",
            [batch_p, batch_q],
            ssd.jensenshannon(batch_p, batch_q, axis=-1),
            axis=-1,
        )
    )
    tv = 0.5 * ssd.cityblock(p, q)
    out.append(case("totalVariation", "totalVariation", [p, q], np.array(tv)))
    u = g.uniform(0.01, 0.99, size=10)
    out.append(case("binaryEntropy", "binaryEntropy", [u], sp.entr(u) + sp.entr(1 - u)))
    out.append(case("binaryEntropy/base=2", "binaryEntropy", [u], (sp.entr(u) + sp.entr(1 - u)) / np.log(2), base=2))

    # Tables of counts: sklearn's mutual_info_score on the contingency table, pyitlib on the matching samples.
    counts2 = g.integers(0, 6, size=(3, 4)).astype(np.float64)
    counts2[0, 0] = 0.0
    xy = samples_of(counts2)
    mi = skm.mutual_info_score(None, None, contingency=counts2.astype(np.int64))
    out.append(case("mutualInformation", "mutualInformation", [counts2], np.array(mi)))
    out.append(case("mutualInformation/base=2", "mutualInformation", [counts2], np.array(mi / np.log(2)), base=2))
    out.append(case("jointEntropy", "jointEntropy", [counts2], np.array(st.entropy(counts2.ravel()))))
    out.append(
        case(
            "conditionalEntropy/given=1",
            "conditionalEntropy",
            [counts2],
            np.array(drv.entropy_conditional(xy[0], xy[1], base=NATS)),
            given=1,
        )
    )
    out.append(
        case(
            "conditionalEntropy/given=0",
            "conditionalEntropy",
            [counts2],
            np.array(drv.entropy_conditional(xy[1], xy[0], base=NATS)),
            given=0,
        )
    )
    counts3 = g.integers(0, 5, size=(2, 3, 3)).astype(np.float64)
    xyz = samples_of(counts3)
    cmi = drv.information_mutual_conditional(xyz[0], xyz[1], xyz[2], base=NATS)
    out.append(case("conditionalMutualInformation", "conditionalMutualInformation", [counts3], np.array(cmi)))
    co = drv.information_co(xyz, base=2)
    out.append(case("interactionInformation/3", "interactionInformation", [counts3], np.array(co), base=2))
    co2 = drv.information_co(xy, base=NATS)
    out.append(case("interactionInformation/2", "interactionInformation", [counts2], np.array(co2)))
    counts4 = g.integers(1, 4, size=(2, 2, 2, 2)).astype(np.float64)
    co4 = drv.information_co(samples_of(counts4), base=NATS)
    out.append(case("interactionInformation/4", "interactionInformation", [counts4], np.array(co4)))
    out.append(
        case(
            "conditionalEntropy/given=[1,2]",
            "conditionalEntropy",
            [counts3],
            np.array(drv.entropy_conditional(xyz[0], xyz[1] * 10 + xyz[2], base=NATS)),
            given=[1, 2],
        )
    )
    # Information gain of a split: rows are children, columns classes.
    split = g.integers(0, 8, size=(3, 2)).astype(np.float64)
    child_class = samples_of(split)
    gain = drv.information_mutual(child_class[1], child_class[0], base=2)
    out.append(case("informationGain", "informationGain", [split], np.array(gain), base=2))
    out.append(case("gainRatio", "gainRatio", [split], np.array(gain / drv.entropy(child_class[0], base=2))))
    return out
