"""Maximum-entropy IRL's dynamic programmes on a small slippery grid world, in numpy and scipy: soft value iteration
(scipy.special.logsumexp backups to a fixed point) with its policy, the finite-horizon expected state visitation
frequencies of Ziebart et al. (2008, algorithm 1, stationary policy) and the discounted occupancy measure
(numpy.linalg.solve), and one MaxEnt IRL gradient step (the feature-expectation gap) for one-hot features."""

import numpy as np
import scipy
import scipy.special
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}

MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))


def grid(n: int, slip: float) -> tuple[FloatArray, FloatArray]:
    """Continuing transitions P[s, a, s'] and terminal probabilities T[s, a] of an n x n grid whose last cell is an
    absorbing goal: moves succeed with probability 1 - 2 slip and slip sideways otherwise; walls block."""
    ns = n * n
    p = np.zeros((ns, 4, ns))
    t = np.zeros((ns, 4))
    goal = ns - 1
    for s in range(ns):
        r, c = divmod(s, n)
        for a in range(4):
            if s == goal:
                t[s, a] = 1.0
                continue
            for m, q in ((a, 1 - 2 * slip), ((a + 1) % 4, slip), ((a + 3) % 4, slip)):
                dr, dc = MOVES[m]
                r2, c2 = r + dr, c + dc
                s2 = r2 * n + c2 if 0 <= r2 < n and 0 <= c2 < n else s
                if s2 == goal:
                    t[s, a] += q
                else:
                    p[s, a, s2] += q
    return p, t


def soft_value_iteration(p: FloatArray, reward: FloatArray, gamma: float) -> tuple[FloatArray, FloatArray]:
    v = np.zeros(p.shape[0])
    for _ in range(100_000):
        q = reward[:, None] + gamma * p @ v
        v_new = scipy.special.logsumexp(q, axis=1)
        done = float(np.max(np.abs(v_new - v))) < 1e-12
        v = v_new
        if done:
            break
    q = reward[:, None] + gamma * p @ v
    v = scipy.special.logsumexp(q, axis=1)
    return v, np.exp(q - v[:, None])


def finite_svf(p: FloatArray, policy: FloatArray, mu0: FloatArray, horizon: int) -> FloatArray:
    d = mu0.copy()
    total = np.zeros_like(mu0)
    for _ in range(horizon):
        total += d
        d = np.einsum("s,sa,sat->t", d, policy, p)
    return total


def discounted_svf(p: FloatArray, policy: FloatArray, mu0: FloatArray, gamma: float) -> FloatArray:
    p_pi = np.einsum("sa,sat->st", policy, p)
    return np.linalg.solve((np.eye(len(mu0)) - gamma * p_pi).T, mu0)


def cases() -> list[Case]:
    g = rng(92)
    out: list[Case] = []
    for i, (n, slip, gamma) in enumerate(((3, 0.1, 0.9), (4, 0.0, 0.95))):
        p, t = grid(n, slip)
        ns = n * n
        mu0 = np.zeros(ns)
        mu0[0] = 1.0
        reward = g.normal(scale=0.5, size=ns) - 0.5
        v, policy = soft_value_iteration(p, reward, gamma)
        inputs = [p, t, mu0, reward]
        out.append(case(f"soft/{i}/V", "softV", inputs, v, rtol=1e-10, atol=1e-10, gamma=gamma))
        out.append(case(f"soft/{i}/policy", "softPolicy", inputs, policy, rtol=1e-10, atol=1e-10, gamma=gamma))
        horizon = 2 * n + 1
        out.append(
            case(
                f"svf/{i}/finite",
                "svfFinite",
                [*inputs, policy],
                finite_svf(p, policy, mu0, horizon),
                gamma=gamma,
                horizon=horizon,
            )
        )
        disc = discounted_svf(p, policy, mu0, gamma)
        out.append(case(f"svf/{i}/discounted", "svfDiscounted", [*inputs, policy], disc, atol=1e-10, gamma=gamma))
        # One MaxEnt IRL step from theta = 0 with one-hot features: lr (f_expert - D), D of the uniform soft policy.
        trajectories = np.array([[0, 1, 2], [0, n, n + 1]], dtype=np.float64)
        f_expert = np.zeros(ns)
        for traj in trajectories.astype(int):
            for k, s in enumerate(traj):
                f_expert[s] += gamma**k / len(trajectories)
        _, uniform_policy = soft_value_iteration(p, np.zeros(ns), gamma)
        step = 0.1 * (f_expert - discounted_svf(p, uniform_policy, mu0, gamma))
        out.append(case(f"irl/{i}/step", "irlStep", [*inputs, trajectories], step, atol=1e-10, gamma=gamma))
    return out
