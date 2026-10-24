"""Textbook worlds against gymnasium: FrozenLake's and CliffWalking's transition tables and initial distributions, and
the taxi's moves through Taxi's map (the movement half of Taxi's table)."""

from typing import Any, cast

import gymnasium as gym
import numpy as np
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"gymnasium": gym.__version__}


def table(env: gym.Env[Any, Any]) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Every outcome of `env.P` as rows `[state, action, probability, next, reward, terminated]`."""
    p_table = cast(dict[int, dict[int, list[tuple[float, int, float, bool]]]], cast(Any, env).P)
    rows: list[list[float]] = []
    for s in sorted(p_table):
        for a in sorted(p_table[s]):
            for p, s2, r, t in p_table[s][a]:
                rows.append([float(s), float(a), float(p), float(s2), float(r), float(t)])
    return np.array(rows, dtype=np.float64)


def taxi_moves() -> list[list[int]]:
    """For each taxi cell and move (south, north, east, west), the cell it reaches: `[cell, action, next]`."""
    env = cast(Any, gym.make("Taxi-v4").unwrapped)
    out: list[list[int]] = []
    for row in range(5):
        for col in range(5):
            s = int(env.encode(row, col, 0, 1))
            for a in range(4):
                (_, s2, _, _) = env.P[s][a][0]
                r2, c2, _, _ = env.decode(s2)
                out.append([row * 5 + col, a, int(r2) * 5 + int(c2)])
    return out


def cases() -> list[Case]:
    out: list[Case] = []
    configs: list[tuple[str, str, dict[str, Any]]] = [
        ("frozenlake/4x4", "FrozenLake-v1", {}),
        ("frozenlake/8x8", "FrozenLake-v1", {"map_name": "8x8"}),
        ("frozenlake/4x4-dry", "FrozenLake-v1", {"is_slippery": False}),
        ("frozenlake/success-0.6", "FrozenLake-v1", {"success_rate": 0.6}),
        ("frozenlake/schedule", "FrozenLake-v1", {"reward_schedule": (10, -5, -1)}),
        ("frozenlake/custom", "FrozenLake-v1", {"desc": ["SFH", "FFF", "HFG"]}),
        ("cliffwalking/dry", "CliffWalking-v1", {}),
        ("cliffwalking/slippery", "CliffWalking-v1", {"is_slippery": True}),
    ]
    for name, env_id, kwargs in configs:
        env = gym.make(env_id, **kwargs).unwrapped
        params: dict[str, Any] = {k: list(v) if isinstance(v, tuple) else v for k, v in kwargs.items()}
        out.append(case(f"{name}/table", "table", [], table(env), env=env_id, **params))
        out.append(
            case(
                f"{name}/initial", "initial", [], np.asarray(cast(Any, env).initial_state_distrib), env=env_id, **params
            )
        )
    out.append(case("taxi/moves", "taxiMoves", [], np.array(0.0), moves=taxi_moves()))
    return out
