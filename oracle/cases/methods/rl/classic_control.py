"""CartPole dynamics against gymnasium, step for step from fixed states."""

from typing import Any, cast

import gymnasium as gym
import numpy as np
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"gymnasium": gym.__version__}


def trajectory(start: FloatArray, actions: list[int]) -> tuple[FloatArray, FloatArray, FloatArray]:
    """States, rewards and termination flags after each action, from `start` (gymnasium's unwrapped dynamics)."""
    env = gym.make("CartPole-v1").unwrapped
    env.reset(seed=0)
    inner = cast(Any, env)  # CartPoleEnv: its state is a public attribute
    inner.state = start.copy()
    states: list[FloatArray] = []
    rewards: list[float] = []
    terminated: list[float] = []
    for a in actions:
        _, r, term, _, _ = env.step(a)
        states.append(np.asarray(inner.state, dtype=np.float64).copy())
        rewards.append(float(r))
        terminated.append(float(term))
        if term:
            break
    return np.stack(states), np.array(rewards), np.array(terminated)


def cases() -> list[Case]:
    g = rng(70)
    out: list[Case] = []
    for i in range(4):
        start = g.uniform(-0.05, 0.05, 4)
        actions = [int(a) for a in g.integers(0, 2, 60)]
        states, rewards, terminated = trajectory(start, actions)
        out.append(
            case(f"cartpole/{i}/states", "cartPoleStates", [start], states, rtol=1e-12, atol=1e-14, actions=actions)
        )
        out.append(case(f"cartpole/{i}/rewards", "cartPoleRewards", [start], rewards, actions=actions))
        out.append(case(f"cartpole/{i}/terminated", "cartPoleTerminated", [start], terminated, actions=actions))
    return out
