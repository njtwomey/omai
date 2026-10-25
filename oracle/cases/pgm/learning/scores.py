"""Structure scores against pgmpy: the discrete local scores (log-likelihood, BIC, AIC, K2, BDeu, BDs) of several
families on data sampled from the student network, whole-structure scores, and the Gaussian scores (LogLikelihoodGauss,
BICGauss, AICGauss) on continuous data."""

import numpy as np
import pandas as pd
import pgmpy
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from mdml_oracle.pgm import STUDENT, sample_rows
from pgmpy.base import DAG
from pgmpy.structure_score import AIC, BIC, K2, AICGauss, BDeu, BDs, BICGauss, LogLikelihood, LogLikelihoodGauss

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__}

FAMILIES: list[tuple[str, list[str]]] = [
    ("grade", ["difficulty", "intelligence"]),
    ("grade", []),
    ("letter", ["grade"]),
    ("sat", ["intelligence", "letter"]),
    ("difficulty", ["sat"]),
]


def cases() -> list[Case]:
    out: list[Case] = []
    rows = sample_rows(STUDENT, 400, 21)
    frame = pd.DataFrame(rows)
    states = {n: s for n, s, _, _ in STUDENT}
    scores = {
        "logLikelihood": LogLikelihood(frame, state_names=states),
        "bic": BIC(frame, state_names=states),
        "aic": AIC(frame, state_names=states),
        "k2": K2(frame, state_names=states),
        "bdeu": BDeu(frame, equivalent_sample_size=10, state_names=states),
        "bds": BDs(frame, equivalent_sample_size=10, state_names=states),
    }
    for name, scorer in scores.items():
        values = [float(scorer.local_score(v, tuple(ps))) for v, ps in FAMILIES]
        out.append(
            case(
                f"local/{name}",
                "local",
                [],
                np.array(values),
                rtol=1e-11,
                data=rows,
                states=states,
                score=name,
                families=FAMILIES,
            )
        )
        true_dag = DAG([(p, n) for n, _, ps, _ in STUDENT for p in ps])
        out.append(
            case(
                f"structure/{name}",
                "structure",
                [],
                np.array(float(scorer.score(true_dag))),
                rtol=1e-11,
                data=rows,
                states=states,
                score=name,
                edges=[[p, n] for n, _, ps, _ in STUDENT for p in ps],
            )
        )
    g = rng(8)
    n = 150
    x = g.normal(size=n)
    y = 0.5 * x + g.normal(size=n)
    z = -x + 0.8 * y + g.normal(size=n) * 0.6
    cont = pd.DataFrame({"x": x, "y": y, "z": z})
    data = [{"x": float(a), "y": float(b), "z": float(c)} for a, b, c in zip(x, y, z, strict=True)]
    gfam: list[tuple[str, list[str]]] = [("z", ["x", "y"]), ("z", []), ("y", ["x"]), ("x", ["z"])]
    for name, scorer in {
        "logLikelihoodGauss": LogLikelihoodGauss(cont),
        "bicGauss": BICGauss(cont),
        "aicGauss": AICGauss(cont),
    }.items():
        values = [float(scorer.local_score(v, tuple(ps))) for v, ps in gfam]
        out.append(
            case(f"gaussian/{name}", "gaussian", [], np.array(values), rtol=1e-10, data=data, score=name, families=gfam)
        )
    return out
