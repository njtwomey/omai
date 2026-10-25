"""Determinantal point processes against DPPy: the marginal kernel (FiniteDPP.compute_K), elementary symmetric
polynomials of the eigenvalues (dppy.exact_sampling.elementary_symmetric_polynomials), and subset, inclusion and
k-DPP probabilities, conditioning and expected sizes from numpy determinants and inverses."""

import contextlib
import io

import numpy as np
from dppy.exact_sampling import elementary_symmetric_polynomials
from dppy.finite_dpps import FiniteDPP
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"dppy": "0.3.3", "numpy": np.__version__}


def kernel(g: np.random.Generator, n: int) -> FloatArray:
    features = g.normal(size=(n, 3))
    return features @ features.T + 0.1 * np.eye(n)


def cases() -> list[Case]:
    g = rng(404)
    out: list[Case] = []
    for n in [4, 6]:
        lk = kernel(g, n)
        dpp = FiniteDPP("likelihood", L=lk)
        with contextlib.redirect_stdout(io.StringIO()):
            dpp.compute_K()
        k = np.asarray(dpp.K)
        out.append(case(f"marginal/{n}", "marginal", [lk], k, rtol=1e-10, atol=1e-13))
        eig = np.linalg.eigvalsh(lk)
        e = elementary_symmetric_polynomials(eig, n)[:, -1]
        out.append(case(f"esp/{n}", "esp", [eig], e, rtol=1e-10, k=n))
        norm = np.linalg.det(lk + np.eye(n))
        for subset in [[0], [1, 2], [0, 2, 3]]:
            tag = f"{n}/{'-'.join(map(str, subset))}"
            sub = np.ix_(subset, subset)
            p = np.linalg.det(lk[sub]) / norm
            out.append(case(f"log-prob/{tag}", "log-prob", [lk], np.array(np.log(p)), rtol=1e-10, subset=subset))
            out.append(
                case(f"inclusion/{tag}", "inclusion", [lk], np.array(np.linalg.det(k[sub])), rtol=1e-9, subset=subset)
            )
            kp = np.linalg.det(lk[sub]) / e[len(subset)]
            out.append(case(f"k-log-prob/{tag}", "k-log-prob", [lk], np.array(np.log(kp)), rtol=1e-9, subset=subset))
        out.append(case(f"size/{n}", "size", [lk], np.array(np.trace(k)), rtol=1e-11))
        # conditioning on including item 0: L^A = ([(L + I_notA)^{-1}]_notA)^{-1} - I
        a = [0]
        rest = [i for i in range(n) if i not in a]
        ia = np.eye(n)
        ia[0, 0] = 0
        inner = np.linalg.inv(lk + ia)[np.ix_(rest, rest)]
        cond = np.linalg.inv(inner) - np.eye(n - 1)
        out.append(case(f"condition-include/{n}", "condition-include", [lk], cond, rtol=1e-9, atol=1e-12, include=a))
    return out
