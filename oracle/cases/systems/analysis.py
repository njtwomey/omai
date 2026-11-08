"""Controllability and observability matrices and Gramians (with their Cholesky factors) against python-control."""

import control as ct
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from omai_oracle.lti import schur_stable_system, stable_system

REFERENCE: dict[str, str] = {"control": ct.__version__}


def cases() -> list[Case]:
    g = rng(4166)
    out: list[Case] = []
    a, b, c, d = stable_system(g, 4, 2, 3)
    out.append(case("ctrb", "ctrb", [a, b], ct.ctrb(a, b), rtol=1e-12, atol=1e-12))
    out.append(case("obsv", "obsv", [a, c], ct.obsv(a, c), rtol=1e-12, atol=1e-12))
    da, db, dc, dd = schur_stable_system(g, 3, 2, 2)
    for label, (sa, sb, sc, sd, dt) in {"continuous": (a, b, c, d, 0.0), "discrete": (da, db, dc, dd, 0.5)}.items():
        sys = ct.ss(sa, sb, sc, sd, dt)
        for kind in ["c", "o", "cf", "of"]:
            out.append(
                case(
                    f"gram/{label}/{kind}",
                    "gram",
                    [sa, sb, sc, sd],
                    ct.gram(sys, kind),
                    rtol=1e-9,
                    atol=1e-11,
                    type=kind,
                    dt=dt,
                )
            )
    return out
