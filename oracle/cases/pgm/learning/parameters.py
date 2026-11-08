"""Parameter learning against pgmpy: maximum likelihood and Bayesian (BDeu, K2, Dirichlet) CPDs from complete data,
expectation maximisation with a latent variable from fixed starting CPDs (a fixed number of iterations, so both sides
take the same EM steps), and least-squares fits of linear-Gaussian networks."""

from typing import Any

import numpy as np
import pandas as pd
import pgmpy
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from omai_oracle.pgm import STUDENT, NodeSpec, cpd_rows, sample_rows
from pgmpy.base import DAG
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator, DiscreteEM, DiscreteMLE

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__}

# A latent class h with an observed root r and three observed children.
LATENT: list[NodeSpec] = [
    ("r", ["0", "1"], [], [[0.4, 0.6]]),
    ("h", ["0", "1"], ["r"], [[0.8, 0.2], [0.3, 0.7]]),
    ("x1", ["0", "1", "2"], ["h"], [[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]]),
    ("x2", ["0", "1"], ["h"], [[0.85, 0.15], [0.2, 0.8]]),
    ("x3", ["0", "1"], ["h", "r"], [[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.15, 0.85]]),
]
EM_INIT: dict[str, list[list[float]]] = {
    "h": [[0.6, 0.4], [0.45, 0.55]],
    "x1": [[0.5, 0.3, 0.2], [0.25, 0.35, 0.4]],
    "x2": [[0.7, 0.3], [0.4, 0.6]],
    "x3": [[0.7, 0.3], [0.55, 0.45], [0.45, 0.55], [0.35, 0.65]],
}


def structure(spec: list[NodeSpec]) -> DAG:
    g = DAG([(p, n) for n, _, ps, _ in spec for p in ps])
    g.add_nodes_from([n for n, _, _, _ in spec])
    return g


def network(spec: list[NodeSpec], cpds: list[TabularCPD]) -> list[dict[str, Any]]:
    """Fitted CPDs as omai nodes, parents sorted (pgmpy's order)."""
    by = {str(c.variable): c for c in cpds}
    return [{"name": n, "states": s, "parents": sorted(ps), "cpt": cpd_rows(by[n], sorted(ps))} for n, s, ps, _ in spec]


def lg_data(n: int) -> list[dict[str, float]]:
    g = rng(12)
    x = g.normal(size=n)
    y = 1.5 - 0.7 * x + g.normal(size=n) * 0.5
    z = 0.3 + 0.4 * x + 1.2 * y + g.normal(size=n) * 0.8
    return [{"x": float(a), "y": float(b), "z": float(c)} for a, b, c in zip(x, y, z, strict=True)]


def cases() -> list[Case]:
    out: list[Case] = []
    zero = np.array(0.0)
    rows = sample_rows(STUDENT, 300, 7)
    frame = pd.DataFrame(rows)
    states = {n: s for n, s, _, _ in STUDENT}
    dag = structure(STUDENT)
    mle = DiscreteMLE(state_names=states).fit(dag, frame).parameters_
    out.append(case("mle", "mle", [], zero, rtol=1e-12, data=rows, states=states, network=network(STUDENT, mle)))
    for prior, estimator in [
        ("BDeu", DiscreteBayesianEstimator(state_names=states, prior_type="BDeu", equivalent_sample_size=5)),
        ("K2", DiscreteBayesianEstimator(state_names=states, prior_type="K2")),
        ("dirichlet", DiscreteBayesianEstimator(state_names=states, prior_type="dirichlet", pseudo_counts=2)),
    ]:
        est = estimator.fit(dag, frame).parameters_
        out.append(
            case(
                f"bayesian/{prior}",
                "bayesian",
                [],
                zero,
                data=rows,
                states=states,
                prior=prior,
                network=network(STUDENT, est),
            )
        )
    # EM with h latent: the observed data drop h; the starting CPDs are given for every updatable node.
    full = sample_rows(LATENT, 250, 3)
    observed = [{k: v for k, v in r.items() if k != "h"} for r in full]
    lstates = {n: s for n, s, _, _ in LATENT}
    ldag = structure(LATENT)
    ldag.latents = {"h"}
    init = {
        n: TabularCPD(
            n,
            len(lstates[n]),
            np.array(EM_INIT[n])
            .reshape([*[len(lstates[p]) for p in sorted(ps)], len(lstates[n])])
            .reshape(-1, len(lstates[n]))
            .T,
            evidence=sorted(ps) or None,
            evidence_card=[len(lstates[p]) for p in sorted(ps)] or None,
            state_names={n: [int(x) for x in lstates[n]], **{p: [int(x) for x in lstates[p]] for p in ps}},
        )
        for n, _, ps, _ in LATENT
        if n in EM_INIT
    }
    numeric = pd.DataFrame([{k: int(v) for k, v in r.items()} for r in observed])
    em = DiscreteEM(
        latent_card={"h": 2},
        max_iter=8,
        atol=0.0,
        init_cpds=init,
        show_progress=False,
        state_names={n: [int(x) for x in s] for n, s in lstates.items() if n != "h"},
    ).fit(ldag, numeric)
    fitted = network(LATENT, em.parameters_)
    # The root's starting CPD is pgmpy's: maximum likelihood on the observed data.
    r_mle = DiscreteMLE(state_names={"r": [0, 1]}).fit(structure([LATENT[0]]), numeric[["r"]]).parameters_[0]
    start = [
        {"name": n, "states": s, "parents": sorted(ps), "cpt": [v for row in EM_INIT[n] for v in row]}
        if n in EM_INIT
        else {"name": n, "states": s, "parents": [], "cpt": cpd_rows(r_mle, [])}
        for n, s, ps, _ in LATENT
    ]
    out.append(case("em", "em", [], zero, rtol=1e-10, data=observed, init=start, network=fitted, iterations=8))
    data = lg_data(200)
    lg = LinearGaussianBayesianNetwork([("x", "y"), ("x", "z"), ("y", "z")])
    lg.fit(pd.DataFrame(data))
    nodes = []
    for n in ["x", "y", "z"]:
        cpd = lg.get_cpds(n)
        assert isinstance(cpd, LinearGaussianCPD)
        evidence = [str(e) for e in cpd.evidence]
        nodes.append({"name": n, "parents": evidence, "beta": [float(b) for b in cpd.beta], "std": float(cpd.std)})
    out.append(case("linearGaussian", "linearGaussian", [], zero, rtol=1e-10, data=data, nodes=nodes))
    return out
