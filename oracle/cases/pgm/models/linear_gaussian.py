"""Linear-Gaussian networks and Gaussian Markov random fields against pgmpy (to_joint_gaussian and log_likelihood of a
LinearGaussianBayesianNetwork) and numpy/scipy (Gaussian conditioning, GMRF conditionals and log densities).

pgmpy rounds the joint to 8 decimals, so the networks use weights and variances with short exact expansions."""

import networkx as nx
import numpy as np
import pandas as pd
import pgmpy
import scipy
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork
from scipy.stats import multivariate_normal

REFERENCE: dict[str, str] = {"pgmpy": pgmpy.__version__, "numpy": np.__version__, "scipy": scipy.__version__}

# name, parents, weights, intercept, variance
type LgNode = tuple[str, list[str], list[float], float, float]

NETWORK: list[LgNode] = [
    ("x1", [], [], 1.0, 4.0),
    ("x2", ["x1"], [0.5], -5.0, 4.0),
    ("x3", ["x2"], [-1.0], 4.0, 3.0),
    ("x4", ["x1", "x3"], [0.25, 2.0], 0.0, 1.0),
    ("x5", ["x4"], [-0.5], 2.0, 0.25),
]


def pgmpy_lgbn(spec: list[LgNode]) -> LinearGaussianBayesianNetwork:
    model = LinearGaussianBayesianNetwork([(p, n) for n, ps, _, _, _ in spec for p in ps])
    model.add_nodes_from([n for n, _, _, _, _ in spec])
    for n, ps, ws, b, v in spec:
        model.add_cpds(LinearGaussianCPD(n, [b, *ws], float(np.sqrt(v)), evidence=ps))
    return model


def definition(spec: list[LgNode]) -> list[dict[str, object]]:
    return [{"name": n, "parents": ps, "weights": ws, "intercept": b, "variance": v} for n, ps, ws, b, v in spec]


def cases() -> list[Case]:
    out: list[Case] = []
    model = pgmpy_lgbn(NETWORK)
    net = definition(NETWORK)
    mean, cov = model.to_joint_gaussian()
    order = [str(v) for v in nx.topological_sort(model)]
    out.append(case("joint", "joint", [], np.concatenate([mean, cov.ravel()]), network=net, order=order))
    g = rng(4)
    data = g.normal(size=(6, 5)) * 2
    frame = pd.DataFrame(data, columns=[n for n, _, _, _, _ in NETWORK])
    ll = float(model.log_likelihood(frame))
    out.append(case("loglik", "loglik", [data], np.array(ll), rtol=1e-10, atol=1e-10, network=net))
    # Gaussian conditioning of the joint (numpy), in the network's own variable order.
    names = [n for n, _, _, _, _ in NETWORK]
    perm = [order.index(n) for n in names]
    mu = mean[perm]
    sigma = cov[np.ix_(perm, perm)]
    for t, (query, observed) in enumerate([(["x1"], ["x5"]), (["x2", "x4"], ["x1", "x5"]), (["x3"], ["x2", "x4"])]):
        q = [names.index(v) for v in query]
        o = [names.index(v) for v in observed]
        y = g.normal(size=len(o)) + mu[o]
        s_qo = sigma[np.ix_(q, o)]
        s_oo = sigma[np.ix_(o, o)]
        post_mean = mu[q] + s_qo @ np.linalg.solve(s_oo, y - mu[o])
        post_cov = sigma[np.ix_(q, q)] - s_qo @ np.linalg.solve(s_oo, s_qo.T)
        log_evidence = multivariate_normal(mu[o], s_oo).logpdf(y)
        out.append(
            case(
                f"posterior/{t}",
                "posterior",
                [y],
                np.concatenate([post_mean, post_cov.ravel(), [log_evidence]]),
                rtol=1e-10,
                atol=1e-12,
                network=net,
                query=query,
                observed=observed,
            )
        )
    # A random GMRF: conditionals and log densities.
    n = 6
    a = g.normal(size=(n, n)) * (g.uniform(size=(n, n)) < 0.4)
    precision = a @ a.T + n * np.eye(n) * 0.5
    gmean = g.normal(size=n)
    gnames = [f"v{i}" for i in range(n)]
    observed_idx = [1, 4]
    free = [i for i in range(n) if i not in observed_idx]
    xb = g.normal(size=len(observed_idx))
    cov_full = np.linalg.inv(precision)
    s_ab = cov_full[np.ix_(free, observed_idx)]
    s_bb = cov_full[np.ix_(observed_idx, observed_idx)]
    cond_mean = gmean[free] + s_ab @ np.linalg.solve(s_bb, xb - gmean[observed_idx])
    out.append(
        case(
            "gmrf/conditional",
            "gmrfConditional",
            [precision, gmean, xb],
            cond_mean,
            rtol=1e-10,
            atol=1e-12,
            names=gnames,
            observed=[gnames[i] for i in observed_idx],
        )
    )
    points = g.normal(size=(3, n))
    out.append(
        case(
            "gmrf/logpdf",
            "gmrfLogpdf",
            [precision, gmean, points],
            np.asarray(multivariate_normal(gmean, cov_full).logpdf(points)),
            rtol=1e-10,
            atol=1e-12,
            names=gnames,
        )
    )
    return out
