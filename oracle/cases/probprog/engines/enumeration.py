"""Exact inference on discrete programs (probprog/engines/enumeration.ts) against pgmpy: a Bayes network written as a
program (one categorical site per node, observed nodes conditioned) has the posterior marginals of pgmpy's variable
elimination, and the evidence of the network's joint table."""

import numpy as np
import pgmpy
from mdml_oracle.fixture import Case, case
from mdml_oracle.pgm import NETWORKS, definition, joint_table, pgmpy_model, state_index, ve_query

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__}

EVIDENCE: dict[str, list[dict[str, str]]] = {
    "sprinkler": [{"wet": "yes"}, {"wet": "yes", "sprinkler": "off"}],
    "asia": [{"xray": "yes"}, {"dysp": "yes", "smoke": "no"}],
    "student": [{"letter": "strong"}, {"sat": "high", "letter": "weak"}],
}


def cases() -> list[Case]:
    out: list[Case] = []
    for net, evidences in EVIDENCE.items():
        spec = NETWORKS[net]
        model = pgmpy_model(spec)
        names, joint = joint_table(model)
        for i, ev in enumerate(evidences):
            restricted = joint
            for var, state in sorted(ev.items(), key=lambda kv: -names.index(kv[0])):
                restricted = np.take(restricted, state_index(model, var, state), axis=names.index(var))
            evidence_index = {var: state_index(model, var, state) for var, state in ev.items()}
            out.append(
                case(
                    f"{net}/{i}/evidence",
                    "evidence",
                    [],
                    np.array(float(np.log(restricted.sum()))),
                    rtol=1e-12,
                    network=definition(spec),
                    evidence=evidence_index,
                )
            )
            for name, _, _, _ in spec:
                if name in ev:
                    continue
                marg = ve_query(model, [name], ev)
                out.append(
                    case(
                        f"{net}/{i}/{name}",
                        "marginal",
                        [],
                        marg,
                        rtol=1e-12,
                        atol=1e-14,
                        network=definition(spec),
                        evidence=evidence_index,
                        query=name,
                    )
                )
    return out
