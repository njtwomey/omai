"""Bandit agents' deterministic update rules replayed over fixed reward sequences, as numpy transcriptions: UCB1 (Auer,
Cesa-Bianchi and Fischer 2002) and KL-UCB (Garivier and Cappe 2011, the index by `scipy.optimize.brentq`) choosing arms
against a fixed reward table; EXP3 probabilities (Auer et al. 2002), gradient-bandit preferences (Sutton and Barto,
section 2.8), constant-step epsilon-greedy estimates, Beta and Gaussian (known and normal-gamma) posteriors over a
fixed list of pulls; LinUCB (Li et al. 2010) choosing arms with ridge estimates from `numpy.linalg.solve`.
"""

import math

import numpy as np
import numpy.typing as npt
import scipy
import scipy.optimize
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}

type Array = npt.NDArray[np.float64]

K = 4
T = 40


def ucb1(table: Array, c: float) -> tuple[Array, Array]:
    n = np.zeros(K)
    mean = np.zeros(K)
    actions = np.zeros(T)
    for t in range(T):
        idx = [math.inf if n[a] == 0 else mean[a] + math.sqrt((c * math.log(max(t, 1))) / n[a]) for a in range(K)]
        a = int(np.argmax(idx))
        r = float(table[t, a])
        n[a] += 1
        mean[a] += (r - mean[a]) / n[a]
        actions[t] = a
    return actions, mean


def kl_bernoulli(p: float, q: float) -> float:
    a = min(max(p, 1e-15), 1 - 1e-15)
    b = min(max(q, 1e-15), 1 - 1e-15)
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def kl_ucb_index(mean: float, n: float, t: int) -> float:
    if n == 0:
        return math.inf
    bound = math.log(max(t, 1)) / n
    if not bound > 0:
        return mean

    def f(q: float) -> float:
        return kl_bernoulli(mean, q) - bound

    if f(1.0) <= 0:
        return 1.0
    lo = min(max(mean, 0.0), 1.0)
    if f(lo) >= 0:
        return lo
    return float(scipy.optimize.brentq(f, lo, 1.0, xtol=1e-12))


def kl_ucb(table: Array) -> tuple[Array, Array]:
    n = np.zeros(K)
    mean = np.zeros(K)
    actions = np.zeros(T)
    for t in range(T):
        a = int(np.argmax([kl_ucb_index(float(mean[i]), float(n[i]), t) for i in range(K)]))
        r = float(table[t, a])
        n[a] += 1
        mean[a] += (r - mean[a]) / n[a]
        actions[t] = a
    return actions, mean


def exp3(actions: Array, rewards: Array, gamma: float) -> Array:
    lw = np.zeros(K)

    def probs(x: Array) -> Array:
        w = np.exp(x - x.max())
        return (1 - gamma) * w / w.sum() + gamma / K

    out = []
    p = probs(lw)
    for a, r in zip(actions.astype(int), rewards, strict=True):
        lw[a] += gamma * r / (p[a] * K)
        p = probs(lw)
        out.append(p.copy())
    return np.array(out)


def gradient(actions: Array, rewards: Array, alpha: float) -> Array:
    h = np.zeros(K)
    baseline = 0.0
    for t, (a, r) in enumerate(zip(actions.astype(int), rewards, strict=True), start=1):
        e = np.exp(h - h.max())
        pi = e / e.sum()
        adv = r - (r if t == 1 else baseline)
        one = np.zeros(K)
        one[a] = 1.0
        h = h + alpha * adv * (one - pi)
        baseline += (r - baseline) / t
    return np.concatenate([h, [baseline]])


def eps_constant(actions: Array, rewards: Array, alpha: float, init: float) -> Array:
    q = np.full(K, init)
    for a, r in zip(actions.astype(int), rewards, strict=True):
        q[a] += alpha * (r - q[a])
    return q


def beta_posterior(actions: Array, rewards: Array) -> Array:
    a0 = np.ones(K)
    b0 = np.ones(K)
    for a, r in zip(actions.astype(int), rewards, strict=True):
        if r == 1:
            a0[a] += 1
        else:
            b0[a] += 1
    return np.concatenate([a0, b0])


def gaussian_known(actions: Array, rewards: Array, variance: float, prior_var: float, mu0: float) -> Array:
    n = np.array([np.sum(actions == a) for a in range(K)], dtype=np.float64)
    s = np.array([rewards[actions == a].sum() for a in range(K)])
    prec = 1 / prior_var + n / variance
    return np.concatenate([(mu0 / prior_var + s / variance) / prec, prec])


def gaussian_unknown(actions: Array, rewards: Array, mu0: float, k0: float, a0: float, b0: float) -> Array:
    out_mean, out_k, out_a, out_b = [], [], [], []
    for a in range(K):
        x = rewards[actions == a]
        n = float(len(x))
        xbar = float(x.mean()) if n > 0 else 0.0
        m2 = float(((x - xbar) ** 2).sum())
        kn = k0 + n
        out_mean.append((k0 * mu0 + n * xbar) / kn)
        out_k.append(kn)
        out_a.append(a0 + n / 2)
        out_b.append(b0 + m2 / 2 + k0 * n * (xbar - mu0) ** 2 / (2 * kn))
    return np.array(out_mean + out_k + out_a + out_b)


def lin_ucb(features: Array, table: Array, alpha: float, lam: float, shared: bool) -> tuple[Array, Array]:
    d = features.shape[2]
    models = 1 if shared else K
    big_a = [lam * np.eye(d) for _ in range(models)]
    b = [np.zeros(d) for _ in range(models)]
    actions = np.zeros(T)
    for t in range(T):
        scores = []
        for a in range(K):
            m = 0 if shared else a
            x = features[t, a]
            theta = np.linalg.solve(big_a[m], b[m])
            scores.append(theta @ x + alpha * math.sqrt(x @ np.linalg.solve(big_a[m], x)))
        a = int(np.argmax(scores))
        m = 0 if shared else a
        x = features[t, a]
        big_a[m] = big_a[m] + np.outer(x, x)
        b[m] = b[m] + table[t, a] * x
        actions[t] = a
    thetas = np.concatenate([np.linalg.solve(big_a[m], b[m]) for m in range(models)])
    return actions, thetas


def cases() -> list[Case]:
    g = rng(7)
    probs = np.array([0.2, 0.4, 0.55, 0.7])
    binary = (g.uniform(size=(T, K)) < probs).astype(np.float64)
    pulls = g.integers(0, K, size=T).astype(np.float64)
    unit_rewards = np.round(g.uniform(size=T), 6)
    normal_rewards = np.round(g.normal(1.0, 2.0, size=T), 6)
    binary_rewards = binary[np.arange(T), pulls.astype(int)]
    features = g.normal(size=(T, K, 3))
    # Unequal lengths keep the first confidence widths from tying.
    features *= g.uniform(0.5, 1.5, size=(T, K, 1)) / np.linalg.norm(features, axis=2, keepdims=True)
    theta = np.array([0.6, -0.3, 0.74])
    linear_table = features @ theta + 0.1 * np.round(g.normal(size=(T, K)), 6)

    out: list[Case] = []
    actions, means = ucb1(binary, 2.0)
    out.append(case("ucb1/actions", "ucb1", [binary], actions, c=2.0))
    out.append(case("ucb1/means", "ucb1", [binary], means, c=2.0))
    actions, means = kl_ucb(binary)
    out.append(case("klUcb/actions", "klUcb", [binary], actions))
    out.append(case("klUcb/means", "klUcb", [binary], means))
    out.append(case("exp3", "exp3", [pulls, unit_rewards], np.ravel(exp3(pulls, unit_rewards, 0.2)), gamma=0.2))
    out.append(case("gradientBandit", "gradientBandit", [pulls, normal_rewards], gradient(pulls, normal_rewards, 0.1)))
    out.append(
        case(
            "epsilonGreedyBandit",
            "epsilonGreedyBandit",
            [pulls, normal_rewards],
            eps_constant(pulls, normal_rewards, 0.1, 5.0),
        )
    )
    out.append(
        case("thompsonBernoulli", "thompsonBernoulli", [pulls, binary_rewards], beta_posterior(pulls, binary_rewards))
    )
    out.append(
        case(
            "thompsonGaussian/known",
            "gaussianPosterior",
            [pulls, normal_rewards],
            gaussian_known(pulls, normal_rewards, 4.0, 2.0, 0.5),
            variance=4.0,
            priorVariance=2.0,
            priorMean=0.5,
        )
    )
    out.append(
        case(
            "thompsonGaussian/estimated",
            "gaussianPosterior",
            [pulls, normal_rewards],
            gaussian_unknown(pulls, normal_rewards, 0.5, 2.0, 1.5, 0.7),
            priorMean=0.5,
            kappa0=2.0,
            alpha0=1.5,
            beta0=0.7,
            rtol=1e-10,
            atol=1e-12,
        )
    )
    for shared in (True, False):
        actions, thetas = lin_ucb(features, linear_table, 0.5, 1.0, shared)
        name = "shared" if shared else "disjoint"
        out.append(case(f"linUcb/{name}/actions", "linUcb", [features, linear_table], actions, model=name))
        out.append(
            case(f"linUcb/{name}/theta", "linUcb", [features, linear_table], thetas, rtol=1e-10, atol=1e-12, model=name)
        )
    return out
