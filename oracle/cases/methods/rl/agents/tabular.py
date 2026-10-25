"""Tabular reinforcement-learning update rules replayed over fixed transitions, as numpy transcriptions of Sutton and
Barto (2018): Q-learning (6.5), SARSA (6.4), expected SARSA (6.6), n-step SARSA (7.2), Monte Carlo control (5.4),
SARSA(lambda) (12.7), double Q-learning (6.7), TD(0) (6.1), TD(lambda) (12.2) and true online TD(lambda) (12.5), the
Dyna-Q model (8.2) and prioritised sweeping (8.4, with `heapq`).

The transitions are random episodes of a 5-state, 3-action problem; each ends by termination or truncation. Conventions
shared with the TypeScript agents: no bootstrap after termination; after truncation, bootstrap from the epsilon-greedy
expectation `(1 - eps) max Q + eps mean Q`; SARSA-family updates wait for the next transition's action.
"""

import heapq
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__}

type Array = npt.NDArray[np.float64]
type Row = tuple[int, int, float, int, bool, bool]

N_S = 5
N_A = 3


def episodes(seed: int, count: int) -> list[Row]:
    g = rng(seed)
    rows: list[Row] = []
    for e in range(count):
        s = int(g.integers(N_S))
        length = int(g.integers(2, 8))
        for t in range(length):
            a = int(g.integers(N_A))
            s2 = int(g.integers(N_S))
            r = float(np.round(g.normal(), 6))
            last = t == length - 1
            rows.append((s, a, r, s2, last and e % 3 != 2, last and e % 3 == 2))
            s = s2
    return rows


def eg_value(q: Array, s: int, eps: float) -> float:
    return float((1 - eps) * q[s].max() + eps * q[s].mean())


def step_size(alpha: float | str, n: float) -> float:
    return 1.0 / max(n, 1.0) if alpha == "sample-average" else float(alpha)


class Table:
    def __init__(self, init: float, alpha: float | str) -> None:
        self.q = np.full((N_S, N_A), init)
        self.n = np.zeros((N_S, N_A))
        self.alpha = alpha

    def towards(self, s: int, a: int, target: float) -> None:
        self.n[s, a] += 1
        self.q[s, a] += step_size(self.alpha, self.n[s, a]) * (target - self.q[s, a])


def q_learning(rows: list[Row], alpha: float | str, gamma: float, init: float) -> Array:
    t = Table(init, alpha)
    for s, a, r, s2, term, _ in rows:
        t.towards(s, a, r + (0.0 if term else gamma * t.q[s2].max()))
    return t.q


def expected_sarsa(rows: list[Row], alpha: float, gamma: float, eps: float) -> Array:
    t = Table(0.0, alpha)
    for s, a, r, s2, term, _ in rows:
        t.towards(s, a, r + (0.0 if term else gamma * eg_value(t.q, s2, eps)))
    return t.q


def sarsa(rows: list[Row], alpha: float, gamma: float, eps: float) -> Array:
    t = Table(0.0, alpha)
    prev: tuple[int, int, float, int] | None = None
    for s, a, r, s2, term, trunc in rows:
        if prev is not None:
            ps, pa, pr, ps2 = prev
            boot = t.q[s, a] if ps2 == s else eg_value(t.q, ps2, eps)
            t.towards(ps, pa, pr + gamma * boot)
        if term or trunc:
            t.towards(s, a, r + (0.0 if term else gamma * eg_value(t.q, s2, eps)))
            prev = None
        else:
            prev = (s, a, r, s2)
    return t.q


def n_step_sarsa(rows: list[Row], n: int, alpha: float, gamma: float, eps: float, expected: bool) -> Array:
    t = Table(0.0, alpha)
    window: list[tuple[int, int, float, int]] = []

    def update_first(w: list[tuple[int, int, float, int]], boot: float) -> None:
        g, disc = 0.0, 1.0
        for _, _, r, _ in w:
            g += disc * r
            disc *= gamma
        t.towards(w[0][0], w[0][1], g + disc * boot)

    def flush(w: list[tuple[int, int, float, int]], at: int | None) -> None:
        for i in range(len(w)):
            update_first(w[i:], 0.0 if at is None else eg_value(t.q, at, eps))

    for s, a, r, s2, term, trunc in rows:
        if window and window[-1][3] != s:
            flush(window, window[-1][3])
            window = []
        if len(window) == n:
            update_first(window, eg_value(t.q, s, eps) if expected else float(t.q[s, a]))
            window = window[1:]
        window.append((s, a, r, s2))
        if expected and len(window) == n and not (term or trunc):
            update_first(window, eg_value(t.q, s2, eps))
            window = window[1:]
        if term or trunc:
            flush(window, None if term else s2)
            window = []
    return t.q


def monte_carlo(rows: list[Row], alpha: float | str, gamma: float, eps: float, every: bool) -> Array:
    t = Table(0.0, alpha)
    episode: list[tuple[int, int, float]] = []
    for s, a, r, s2, term, trunc in rows:
        episode.append((s, a, r))
        if not (term or trunc):
            continue
        first: dict[tuple[int, int], int] = {}
        for j, (es, ea, _) in enumerate(episode):
            first.setdefault((es, ea), j)
        g = 0.0 if term else eg_value(t.q, s2, eps)
        for j in reversed(range(len(episode))):
            es, ea, er = episode[j]
            g = er + gamma * g
            if every or first[(es, ea)] == j:
                t.towards(es, ea, g)
        episode = []
    return t.q


def sarsa_lambda(rows: list[Row], alpha: float, gamma: float, eps: float, lam: float, replacing: bool) -> Array:
    q = np.zeros((N_S, N_A))
    z = np.zeros((N_S, N_A))

    def process(s: int, a: int, target: float) -> None:
        nonlocal z
        z[s, a] = 1.0 if replacing else z[s, a] + 1.0
        delta = target - q[s, a]
        q[:] = q + alpha * delta * z
        z = gamma * lam * z

    prev: tuple[int, int, float, int] | None = None
    for s, a, r, s2, term, trunc in rows:
        if prev is not None:
            ps, pa, pr, ps2 = prev
            boot = q[s, a] if ps2 == s else eg_value(q, ps2, eps)
            process(ps, pa, pr + gamma * boot)
            if ps2 != s:
                z = np.zeros((N_S, N_A))
        if term or trunc:
            process(s, a, r + (0.0 if term else gamma * eg_value(q, s2, eps)))
            z = np.zeros((N_S, N_A))
            prev = None
        else:
            prev = (s, a, r, s2)
    return q


def double_q(rows: list[Row], coins: list[bool], alpha: float, gamma: float) -> Array:
    qa = np.zeros((N_S, N_A))
    qb = np.zeros((N_S, N_A))
    for (s, a, r, s2, term, _), use_a in zip(rows, coins, strict=True):
        own, other = (qa, qb) if use_a else (qb, qa)
        target = r + (0.0 if term else gamma * other[s2, int(own[s2].argmax())])
        own[s, a] += alpha * (target - own[s, a])
    return np.concatenate([qa.ravel(), qb.ravel(), ((qa + qb) / 2).ravel()])


def dyna_model(rows: list[Row]) -> Array:
    reward = np.zeros(N_S * N_A)
    nxt = np.full(N_S * N_A, -2.0)
    for s, a, r, s2, term, _ in rows:
        reward[s * N_A + a] = r
        nxt[s * N_A + a] = -1.0 if term else s2
    return np.concatenate([reward, nxt])


def prioritized_sweeping(rows: list[Row], alpha: float, gamma: float, planning: int, theta: float) -> Array:
    q = np.zeros(N_S * N_A)
    reward = np.zeros(N_S * N_A)
    nxt = np.full(N_S * N_A, -2)
    preds: list[list[int]] = [[] for _ in range(N_S)]
    queued = np.zeros(N_S * N_A)
    queue: list[tuple[float, int]] = []

    def target(i: int) -> float:
        return float(reward[i] + (0.0 if nxt[i] < 0 else gamma * q[nxt[i] * N_A : nxt[i] * N_A + N_A].max()))

    def push(i: int) -> None:
        p = abs(target(i) - q[i])
        if p > theta and p > queued[i]:
            queued[i] = p
            heapq.heappush(queue, (-p, i))

    for s, a, r, s2, term, _ in rows:
        i0 = s * N_A + a
        reward[i0] = r
        nxt[i0] = -1 if term else s2
        if not term and i0 not in preds[s2]:
            preds[s2].append(i0)
        push(i0)
        done = 0
        while done < planning and queue:
            neg, i = heapq.heappop(queue)
            if -neg != queued[i]:
                continue
            queued[i] = 0.0
            done += 1
            q[i] += alpha * (target(i) - q[i])
            for j in preds[i // N_A]:
                push(j)
    return q


def td0(rows: list[Row], alpha: float, gamma: float) -> Array:
    v = np.zeros(N_S)
    for s, _, r, s2, term, _ in rows:
        v[s] += alpha * (r + (0.0 if term else gamma * v[s2]) - v[s])
    return v


def td_lambda(rows: list[Row], alpha: float, gamma: float, lam: float, kind: str) -> Array:
    v = np.zeros(N_S)
    z = np.zeros(N_S)
    v_old = 0.0
    for s, _, r, s2, term, trunc in rows:
        x = np.zeros(N_S)
        x[s] = 1.0
        vs = float(v[s])
        vs2 = 0.0 if term else float(v[s2])
        delta = r + gamma * vs2 - vs
        if kind == "dutch":
            z = gamma * lam * z + (1 - alpha * gamma * lam * float(z @ x)) * x
            v = v + alpha * (delta + vs - v_old) * z - alpha * (vs - v_old) * x
        else:
            z = gamma * lam * z
            z[s] = 1.0 if kind == "replacing" else z[s] + 1.0
            v = v + alpha * delta * z
        if term or trunc:
            z = np.zeros(N_S)
            v_old = 0.0
        else:
            v_old = vs2
    return v


def cases() -> list[Case]:
    rows = episodes(0, 12)
    data = np.array([[s, a, r, s2, float(te), float(tr)] for s, a, r, s2, te, tr in rows])
    coins = [bool(c) for c in rng(1).integers(0, 2, size=len(rows))]
    exact: dict[str, Callable[[], Array]] = {
        "qLearning/constant": lambda: q_learning(rows, 0.3, 0.9, 0.5),
        "qLearning/sample-average": lambda: q_learning(rows, "sample-average", 0.9, 0.0),
        "sarsa": lambda: sarsa(rows, 0.4, 0.9, 0.2),
        "expectedSarsa/behaviour": lambda: expected_sarsa(rows, 0.4, 0.9, 0.2),
        "expectedSarsa/greedy-target": lambda: expected_sarsa(rows, 0.4, 0.9, 0.0),
        "nStepSarsa/n1": lambda: n_step_sarsa(rows, 1, 0.4, 0.9, 0.2, False),
        "nStepSarsa/n3": lambda: n_step_sarsa(rows, 3, 0.4, 0.9, 0.2, False),
        "nStepSarsa/expected-n2": lambda: n_step_sarsa(rows, 2, 0.4, 0.9, 0.2, True),
        "monteCarloControl/first-visit": lambda: monte_carlo(rows, "sample-average", 0.9, 0.2, False),
        "monteCarloControl/every-visit": lambda: monte_carlo(rows, 0.1, 0.9, 0.2, True),
        "sarsaLambda/accumulating": lambda: sarsa_lambda(rows, 0.3, 0.9, 0.2, 0.7, False),
        "sarsaLambda/replacing": lambda: sarsa_lambda(rows, 0.3, 0.9, 0.2, 0.7, True),
        "doubleQLearning": lambda: double_q(rows, coins, 0.3, 0.9),
        "dynaQ/model": lambda: dyna_model(rows),
        "dynaQ/no-planning": lambda: q_learning(rows, 0.3, 0.9, 0.0),
        "prioritizedSweeping": lambda: prioritized_sweeping(rows, 0.5, 0.9, 3, 1e-3),
        "td0": lambda: td0(rows, 0.2, 0.9),
        "tdLambda/accumulating": lambda: td_lambda(rows, 0.2, 0.9, 0.6, "accumulating"),
        "tdLambda/replacing": lambda: td_lambda(rows, 0.2, 0.9, 0.6, "replacing"),
        "tdLambda/dutch": lambda: td_lambda(rows, 0.2, 0.9, 0.6, "dutch"),
    }
    return [
        case(name, name.split("/")[0], [data, np.array(coins, dtype=np.float64)], np.ravel(f()), rtol=1e-12, atol=1e-12)
        for name, f in exact.items()
    ]
