"""Closed-form conjugate evidence and posteriors (probprog/validation/exact.ts) against scipy: the evidence of a normal
mean as a multivariate normal density, of a beta probability through the beta function, of a gamma rate through the
gamma function, and of a Dirichlet through the multivariate beta function."""

import importlib.metadata

import numpy as np
import scipy.special as sps
import scipy.stats as st
from omai_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scipy": importlib.metadata.version("scipy")}


def cases() -> list[Case]:
    out: list[Case] = []
    y = np.array([1.2, 0.4, 2.1, -0.3])
    for m0, s0, sigma in [(0.0, 1.0, 1.0), (0.5, 2.0, 0.7)]:
        n = len(y)
        cov = sigma**2 * np.eye(n) + s0**2 * np.ones((n, n))
        evidence = st.multivariate_normal(mean=np.full(n, m0), cov=cov).logpdf(y)
        precision = 1 / s0**2 + n / sigma**2
        mean = (m0 / s0**2 + y.sum() / sigma**2) / precision
        out.append(
            case(
                f"normalNormal/m0={m0}/s0={s0}/sigma={sigma}",
                "normalNormal",
                [y],
                np.array([evidence, mean, np.sqrt(1 / precision)]),
                rtol=1e-11,
                priorMean=m0,
                priorSd=s0,
                sigma=sigma,
            )
        )
    flips = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    for a, b in [(1.0, 1.0), (2.0, 3.0)]:
        k = flips.sum()
        n = len(flips)
        post = st.beta(a + k, b + n - k)
        out.append(
            case(
                f"betaBernoulli/a={a}/b={b}",
                "betaBernoulli",
                [flips],
                np.array([sps.betaln(a + k, b + n - k) - sps.betaln(a, b), post.mean(), post.std()]),
                rtol=1e-11,
                a=a,
                b=b,
            )
        )
    counts = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    for shape, rate in [(2.0, 1.0), (2.0, 1.5)]:
        s = counts.sum()
        n = len(counts)
        post = st.gamma(shape + s, scale=1 / (rate + n))
        evidence = (
            shape * np.log(rate)
            - sps.gammaln(shape)
            + sps.gammaln(shape + s)
            - (shape + s) * np.log(rate + n)
            - sps.gammaln(counts + 1).sum()
        )
        out.append(
            case(
                f"gammaPoisson/shape={shape}/rate={rate}",
                "gammaPoisson",
                [counts],
                np.array([evidence, post.mean(), post.std()]),
                rtol=1e-11,
                shape=shape,
                rate=rate,
            )
        )
    alpha = np.array([1.0, 2.0, 3.0])
    cats = np.array([0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    c = np.array([(cats == k).sum() for k in range(3)], dtype=float)
    post = alpha + c

    def logb(v: np.ndarray) -> float:
        return float(sps.gammaln(v).sum() - sps.gammaln(v.sum()))

    out.append(
        case(
            "dirichletCategorical",
            "dirichletCategorical",
            [alpha, cats],
            np.array([logb(post) - logb(alpha), *(post / post.sum())]),
            rtol=1e-11,
        )
    )
    return out
