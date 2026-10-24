"""Exact solutions of tabular MDPs built from gymnasium's `P` tables, by numpy implementations of the Bellman equations.

Value iteration runs to a sup-norm change below 1e-13; policy evaluation solves `(I - gamma P_pi) V = R_pi` with
`numpy.linalg.solve`; occupancy measures solve the transposed system.
"""

from typing import Any, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"gymnasium": gym.__version__, "numpy": np.__version__}

type Array = npt.NDArray[np.float64]


def arrays(env_id: str, kwargs: dict[str, Any]) -> tuple[Array, Array, Array]:
    """Continuation probabilities P[s, a, s'], expected rewards R[s, a] and the initial distribution."""
    env = cast(Any, gym.make(env_id, **kwargs).unwrapped)
    n_s = len(env.P)
    n_a = len(env.P[0])
    p = np.zeros((n_s, n_a, n_s))
    r = np.zeros((n_s, n_a))
    for s in range(n_s):
        for a in range(n_a):
            for prob, s2, reward, terminated in env.P[s][a]:
                r[s, a] += prob * reward
                if not terminated:
                    p[s, a, s2] += prob
    return p, r, np.asarray(env.initial_state_distrib, dtype=np.float64)


def value_iteration(p: Array, r: Array, gamma: float) -> Array:
    v = np.zeros(p.shape[0])
    for _ in range(100_000):
        q = r + gamma * p @ v
        v2 = q.max(axis=1)
        if np.max(np.abs(v2 - v)) < 1e-13:
            return v2
        v = v2
    return v


def evaluate(p: Array, r: Array, policy: Array, gamma: float) -> Array:
    """Exact values of a stochastic policy `policy[s, a]`."""
    p_pi = np.einsum("sa,sat->st", policy, p)
    r_pi = (policy * r).sum(axis=1)
    return np.linalg.solve(np.eye(p.shape[0]) - gamma * p_pi, r_pi)


def occupancy(p: Array, policy: Array, mu: Array, gamma: float) -> Array:
    p_pi = np.einsum("sa,sat->st", policy, p)
    return np.linalg.solve((np.eye(p.shape[0]) - gamma * p_pi).T, mu)


def cases() -> list[Case]:
    out: list[Case] = []
    problems: list[tuple[str, str, dict[str, Any], float]] = [
        ("frozenlake4", "FrozenLake-v1", {}, 0.99),
        ("frozenlake8", "FrozenLake-v1", {"map_name": "8x8"}, 0.99),
        ("cliff", "CliffWalking-v1", {}, 1.0),
        ("cliffslippery", "CliffWalking-v1", {"is_slippery": True}, 0.95),
        ("taxi", "Taxi-v4", {}, 0.9),
    ]
    for name, env_id, kwargs, gamma in problems:
        p, r, mu = arrays(env_id, kwargs)
        v = value_iteration(p, r, gamma)
        q = r + gamma * p @ v
        params: dict[str, Any] = {"env": env_id, "gamma": gamma, **kwargs}
        out.append(case(f"{name}/V", "valueIteration", [], v, rtol=1e-9, atol=1e-10, **params))
        out.append(case(f"{name}/Q", "qFromV", [], q, rtol=1e-9, atol=1e-10, **params))
        n_s, n_a = r.shape
        uniform = np.full((n_s, n_a), 1.0 / n_a)
        g = gamma if gamma < 1 else 0.99
        out.append(
            case(
                f"{name}/uniform",
                "policyEvaluation",
                [],
                evaluate(p, r, uniform, g),
                rtol=1e-10,
                atol=1e-10,
                **{**params, "gamma": g},
            )
        )
        greedy = np.zeros((n_s, n_a))
        greedy[np.arange(n_s), q.argmax(axis=1)] = 1.0
        out.append(
            case(
                f"{name}/greedy",
                "policyEvaluation",
                [],
                evaluate(p, r, greedy, gamma if gamma < 1 else 0.999),
                rtol=1e-9,
                atol=1e-9,
                policy=[int(a) for a in q.argmax(axis=1)],
                **{**params, "gamma": gamma if gamma < 1 else 0.999},
            )
        )
        out.append(
            case(
                f"{name}/occupancy",
                "occupancyMeasure",
                [],
                occupancy(p, uniform, mu, g),
                rtol=1e-10,
                atol=1e-10,
                **{**params, "gamma": g},
            )
        )
    return out
