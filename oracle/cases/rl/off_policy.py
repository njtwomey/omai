"""Off-policy evaluation by the textbook formulas in numpy: IPS, self-normalised IPS (delta-method standard error),
clipped IPS, the direct method and doubly robust (Dudik, Langford and Li 2011) on logged contextual-bandit rounds; and
tabular fitted Q evaluation's fixed point, `(I - gamma P^ Pi) q = r^` over the empirical model of logged transitions,
by `numpy.linalg.solve`.
"""

import numpy as np
import numpy.typing as npt
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__}

type Array = npt.NDArray[np.float64]


def mean_se(terms: Array) -> Array:
    return np.array([terms.mean(), terms.std(ddof=1) / np.sqrt(len(terms))])


def fqe_fixed_point(rows: Array, policy: Array, n_s: int, n_a: int, gamma: float) -> Array:
    counts = np.zeros(n_s * n_a)
    sums = np.zeros(n_s * n_a)
    cont = np.zeros((n_s * n_a, n_s))
    for s, a, r, s2, term in rows:
        i = int(s) * n_a + int(a)
        counts[i] += 1
        sums[i] += r
        if term == 0:
            cont[i, int(s2)] += 1
    seen = counts > 0
    r_hat = np.where(seen, sums / np.maximum(counts, 1), 0.0)
    p_hat = cont / np.maximum(counts, 1)[:, None]
    # M[i, s' * n_a + a'] = P^(s' | i) pi(a' | s').
    m = (p_hat[:, :, None] * policy[None, :, :]).reshape(n_s * n_a, n_s * n_a)
    return np.linalg.solve(np.eye(n_s * n_a) - gamma * m, r_hat)


def cases() -> list[Case]:
    g = rng(11)
    n, n_c, n_a = 30, 3, 3
    behaviour = g.dirichlet(np.ones(n_a), size=n_c)
    target = g.dirichlet(np.ones(n_a), size=n_c)
    model = np.round(g.uniform(size=(n_c, n_a)), 6)
    contexts = g.integers(0, n_c, size=n)
    actions = np.array([g.choice(n_a, p=behaviour[c]) for c in contexts])
    rewards = np.round(g.normal(model[contexts, actions], 0.3), 6)
    propensities = behaviour[contexts, actions]
    w = target[contexts, actions] / propensities
    inputs = [contexts.astype(np.float64), actions.astype(np.float64), rewards, propensities, target, model]

    out: list[Case] = []
    out.append(case("ips", "ips", inputs, mean_se(w * rewards)))
    v = (w * rewards).sum() / w.sum()
    out.append(case("snips", "snips", inputs, np.array([v, np.sqrt((w**2 * (rewards - v) ** 2).sum()) / w.sum()])))
    out.append(case("clippedIps", "clippedIps", inputs, mean_se(np.minimum(w, 2.0) * rewards), maxWeight=2.0))
    dm = (target[contexts] * model[contexts]).sum(axis=1)
    out.append(case("directMethod", "directMethod", inputs, mean_se(dm)))
    resid = rewards - model[contexts, actions]
    out.append(case("doublyRobust", "doublyRobust", inputs, mean_se(dm + w * resid)))
    out.append(
        case("doublyRobust/clipped", "doublyRobust", inputs, mean_se(dm + np.minimum(w, 2.0) * resid), maxWeight=2.0)
    )

    n_s, n_m = 5, 2
    rows = []
    for _ in range(80):
        s = int(g.integers(n_s))
        a = int(g.integers(n_m))
        term = float(g.uniform() < 0.15)
        s2 = int(g.integers(n_s)) if s != 4 else int(g.integers(3, n_s))
        rows.append([s, a, float(np.round(g.normal(0.2 * s - 0.3 * a, 0.5), 6)), s2, term])
    data = np.array(rows, dtype=np.float64)
    policy = g.dirichlet(np.ones(n_m), size=n_s)
    for gamma in (0.5, 0.9):
        q = fqe_fixed_point(data, policy, n_s, n_m, gamma)
        out.append(
            case(f"tabularFqe/gamma{gamma}", "tabularFqe", [data, policy], q, rtol=1e-9, atol=1e-10, gamma=gamma)
        )
    return out
