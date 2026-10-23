"""Variable elimination against pgmpy: most probable explanations (map_query over every unobserved variable) and the
probability of evidence, on the Asia, student and sprinkler networks."""

import numpy as np
import pgmpy
from pgmpy.inference import VariableElimination
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.pgm import NETWORKS, definition, evidence_arg, hashables, order_arg, pgmpy_model, stable, ve_query

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__}

EVIDENCE: dict[str, list[dict[str, str]]] = {
    "asia": [{}, {"xray": "yes"}, {"dysp": "yes", "smoke": "no"}, {"xray": "no", "dysp": "yes", "asia": "yes"}],
    "student": [{}, {"letter": "strong"}, {"grade": "C", "sat": "high"}],
    "sprinkler": [{}, {"wet": "yes"}, {"wet": "yes", "sprinkler": "off"}],
}


def cases() -> list[Case]:
    out: list[Case] = []
    for label, spec in NETWORKS.items():
        model = pgmpy_model(spec)
        net = definition(spec)
        names = [n for n, _, _, _ in spec]
        for i, evidence in enumerate(EVIDENCE[label]):
            hidden = [n for n in names if n not in evidence]
            best = VariableElimination(model).map_query(
                hashables(hidden), evidence=evidence_arg(evidence), elimination_order=order_arg([]), show_progress=False
            )
            assignment = {v: str(best[v]) for v in hidden}
            out.append(
                case(
                    f"{label}/mpe/{i}", "mpe", [], np.array(0.0), network=net, evidence=evidence, assignment=assignment
                )
            )
            if evidence:
                observed = list(evidence)
                joint = ve_query(model, observed, {})
                index = np.ravel_multi_index(
                    tuple(next(s for n, s, _, _ in spec if n == v).index(evidence[v]) for v in observed),
                    tuple(len(next(s for n, s, _, _ in spec if n == v)) for v in observed),
                )
                p = stable(np.log(joint[int(index)]))
                out.append(case(f"{label}/evidence/{i}", "logEvidence", [], p, network=net, evidence=evidence))
    return out
