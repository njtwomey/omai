"""Conditional independence tests against pgmpy: chi-square and G tests (statistic, p-value and degrees of freedom,
stratified with Yates' correction on 2 x 2 strata) on data sampled from the student network, and Fisher's z and the
partial-correlation t test on continuous data."""

from typing import Any, cast

import numpy as np
import pandas as pd
import pgmpy
from pgmpy.ci_tests import ChiSquare, FisherZ, GSq, Pearsonr
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng
from whitebox_oracle.pgm import STUDENT, sample_rows

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__}

DISCRETE: list[tuple[str, str, list[str]]] = [
    ("difficulty", "intelligence", []),
    ("difficulty", "intelligence", ["grade"]),
    ("letter", "intelligence", []),
    ("letter", "intelligence", ["grade"]),
    ("sat", "letter", ["intelligence"]),
    ("sat", "difficulty", ["grade", "letter"]),
]
CONTINUOUS: list[tuple[str, str, list[str]]] = [
    ("x", "z", []),
    ("x", "z", ["y"]),
    ("x", "w", ["y", "z"]),
    ("y", "w", []),
]


def cases() -> list[Case]:
    out: list[Case] = []
    rows = sample_rows(STUDENT, 300, 5)
    frame = pd.DataFrame(rows)
    for name, test in [("chiSquare", ChiSquare(frame)), ("gSquare", GSq(frame))]:
        for x, y, z in DISCRETE:
            stat, p = test.run_test(x, y, z)
            dof = int(cast(Any, test).dof_)  # set by run_test
            out.append(
                case(
                    f"{name}/{x}-{y}|{','.join(z)}",
                    name,
                    [],
                    np.array([float(stat), float(p), float(dof)]),
                    rtol=1e-10,
                    atol=1e-14,
                    data=rows,
                    x=x,
                    y=y,
                    z=z,
                )
            )
    g = rng(6)
    n = 120
    x = g.normal(size=n)
    y = x + g.normal(size=n)
    z = y + g.normal(size=n)
    w = g.normal(size=n) + 0.3 * z
    cont = pd.DataFrame({"x": x, "y": y, "z": z, "w": w})
    data = [{"x": float(a), "y": float(b), "z": float(c), "w": float(d)} for a, b, c, d in zip(x, y, z, w, strict=True)]
    for name, ctest in [("fisherZ", FisherZ(cont)), ("partialCorrelation", Pearsonr(cont))]:
        for a, b, cond in CONTINUOUS:
            stat, p = ctest.run_test(a, b, cond)
            out.append(
                case(
                    f"{name}/{a}-{b}|{','.join(cond)}",
                    name,
                    [],
                    np.array([float(stat), float(p)]),
                    rtol=1e-9,
                    atol=1e-14,
                    data=data,
                    x=a,
                    y=b,
                    z=cond,
                )
            )
    return out
