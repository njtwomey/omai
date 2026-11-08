"""Divergences (dists/divergences.ts) against torch.distributions.kl_divergence."""

import numpy as np
import torch
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng, spd
from omai_oracle.torch_helpers import to_numpy, to_torch
from torch import distributions as td

REFERENCE: dict[str, str] = {"torch": torch.__version__}


def arr(*values: float) -> FloatArray:
    return np.array(values, dtype=np.float64)


def cases() -> list[Case]:
    g = rng(52)
    t = to_torch
    out: list[Case] = []
    pairs: list[tuple[str, list[FloatArray], list[FloatArray], td.Distribution, td.Distribution]] = []

    def add(name: str, pp: list[FloatArray], qp: list[FloatArray], p: td.Distribution, q: td.Distribution) -> None:
        pairs.append((name, pp, qp, p, q))

    a, b = arr(0.5, -1.0), arr(1.2, 0.7)
    c, d = arr(1.0, 0.3), arr(0.8, 2.0)
    add("normal", [a, b], [c, d], td.Normal(t(a), t(b)), td.Normal(t(c), t(d)))
    add("logNormal", [a, b], [c, d], td.LogNormal(t(a), t(b)), td.LogNormal(t(c), t(d)))
    add("halfNormal", [b], [d], td.HalfNormal(t(b)), td.HalfNormal(t(d)))
    add("gamma", [d, b], [b, d], td.Gamma(t(d), t(b)), td.Gamma(t(b), t(d)))
    add("chi2", [arr(3.0, 5.0)], [arr(4.0, 2.5)], td.Chi2(t(arr(3.0, 5.0))), td.Chi2(t(arr(4.0, 2.5))))
    add("exponential", [b], [d], td.Exponential(t(b)), td.Exponential(t(d)))
    add("exponentialGamma", [b], [d, b], td.Exponential(t(b)), td.Gamma(t(d), t(b)))
    add("beta", [d, b], [b, d], td.Beta(t(d), t(b)), td.Beta(t(b), t(d)))
    alpha_p, alpha_q = arr(0.8, 2.0, 3.0), arr(1.5, 1.0, 2.2)
    add("dirichlet", [alpha_p], [alpha_q], td.Dirichlet(t(alpha_p)), td.Dirichlet(t(alpha_q)))
    pr, qr = arr(0.2, 0.7), arr(0.5, 0.4)
    add("bernoulli", [pr], [qr], td.Bernoulli(probs=t(pr)), td.Bernoulli(probs=t(qr)))
    n = arr(10.0, 10.0)
    add("binomial", [n, pr], [n, qr], td.Binomial(t(n), probs=t(pr)), td.Binomial(t(n), probs=t(qr)))
    cp, cq = arr(0.2, 0.5, 0.3), arr(0.4, 0.4, 0.2)
    add("categorical", [cp], [cq], td.Categorical(probs=t(cp)), td.Categorical(probs=t(cq)))
    add("oneHotCategorical", [cp], [cq], td.OneHotCategorical(probs=t(cp)), td.OneHotCategorical(probs=t(cq)))
    add("poisson", [d], [b], td.Poisson(t(d)), td.Poisson(t(b)))
    add("geometric", [pr], [qr], td.Geometric(probs=t(pr)), td.Geometric(probs=t(qr)))
    add("laplace", [a, b], [c, d], td.Laplace(t(a), t(b)), td.Laplace(t(c), t(d)))
    add("gumbel", [a, b], [c, d], td.Gumbel(t(a), t(b)), td.Gumbel(t(c), t(d)))
    lo_p, hi_p, lo_q, hi_q = arr(0.0, -1.0), arr(1.0, 2.0), arr(-0.5, 0.0), arr(2.0, 3.0)
    add("uniform", [lo_p, hi_p], [lo_q, hi_q], td.Uniform(t(lo_p), t(hi_p)), td.Uniform(t(lo_q), t(hi_q)))
    add(
        "pareto",
        [d, arr(1.5, 2.0)],
        [b, arr(1.0, 1.5)],
        td.Pareto(t(arr(1.5, 2.0)), t(d)),
        td.Pareto(t(arr(1.0, 1.5)), t(b)),
    )
    mp, mq = g.normal(size=3), g.normal(size=3)
    sp, sq = spd(g, 3) / 3, spd(g, 3) / 2
    add(
        "multivariateNormal",
        [mp, sp],
        [mq, sq],
        td.MultivariateNormal(t(mp), covariance_matrix=t(sp)),
        td.MultivariateNormal(t(mq), covariance_matrix=t(sq)),
    )
    f, diag = g.normal(size=(3, 2)), np.exp(g.normal(size=3))
    add(
        "lowRankMultivariateNormal",
        [mp, f, diag],
        [mq, sq],
        td.LowRankMultivariateNormal(t(mp), t(f), t(diag)),
        td.MultivariateNormal(t(mq), covariance_matrix=t(sq)),
    )
    add(
        "independentNormal",
        [a, b],
        [c, d],
        td.Independent(td.Normal(t(a), t(b)), 1),
        td.Independent(td.Normal(t(c), t(d)), 1),
    )
    for name, pp, qp, p, q in pairs:
        out.append(
            case(f"kl/{name}", f"kl.{name}", [*pp, *qp], to_numpy(td.kl_divergence(p, q)), rtol=1e-10, arity=len(pp))
        )
    return out
