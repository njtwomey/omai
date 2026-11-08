"""Classic-control dynamics against gymnasium, step for step from fixed states (the unwrapped environments)."""

from typing import Any, cast

import gymnasium as gym
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"gymnasium": gym.__version__}


def trajectory(
    env_id: str, start: FloatArray, actions: list[Any], dtype: type[np.generic] = np.float64
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Internal states, rewards and termination flags after each action from `start`, stopping at termination."""
    env = gym.make(env_id).unwrapped
    env.reset(seed=0)
    inner = cast(Any, env)  # the classic-control environments keep their state in a public attribute
    inner.state = start.astype(dtype).copy()
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


def emit(
    out: list[Case],
    prefix: str,
    fn: str,
    start: FloatArray,
    actions: list[Any],
    result: tuple[FloatArray, FloatArray, FloatArray],
    tol: float = 1e-12,
) -> None:
    """Three cases (states, rewards, terminated) for one trajectory."""
    states, rewards, terminated = result
    encoded: list[Any] = [np.asarray(a, dtype=np.float64).tolist() if isinstance(a, np.ndarray) else a for a in actions]
    out.append(case(f"{prefix}/states", fn, [start], states, rtol=tol, atol=1e-14, actions=encoded, part="states"))
    out.append(case(f"{prefix}/rewards", fn, [start], rewards, rtol=tol, atol=1e-14, actions=encoded, part="rewards"))
    out.append(case(f"{prefix}/terminated", fn, [start], terminated, actions=encoded, part="terminated"))


def cases() -> list[Case]:
    g = rng(70)
    out: list[Case] = []
    for i in range(4):
        start = g.uniform(-0.05, 0.05, 4)
        actions = [int(a) for a in g.integers(0, 2, 60)]
        emit(out, f"cartpole/{i}", "cartPole", start, actions, trajectory("CartPole-v1", start, actions))
    for i in range(4):
        start = np.array([g.uniform(-1.2, 0.5), g.uniform(-0.07, 0.07)])
        # Mostly pushing right reaches the goal on some trajectories; the left wall is hit on others.
        actions = [int(a) for a in g.choice([0, 1, 2], size=120, p=[0.3, 0.2, 0.5])]
        emit(out, f"mountaincar/{i}", "mountainCar", start, actions, trajectory("MountainCar-v0", start, actions))
    for i in range(4):
        # Continuous MountainCar stores float32 states: start from float32 values.
        start = np.array([g.uniform(-1.2, 0.4), g.uniform(-0.07, 0.07)]).astype(np.float32).astype(np.float64)
        actions = [np.array([g.uniform(-1.0, 1.0)]) for _ in range(120)]
        emit(
            out,
            f"mountaincarcontinuous/{i}",
            "mountainCarContinuous",
            start,
            actions,
            trajectory("MountainCarContinuous-v0", start, actions, np.float32),
        )
    for i in range(4):
        start = np.array([g.uniform(-np.pi, np.pi), g.uniform(-1.0, 1.0)])
        actions = [np.array([g.uniform(-2.5, 2.5)]) for _ in range(60)]
        emit(out, f"pendulum/{i}", "pendulum", start, actions, trajectory("Pendulum-v1", start, actions))
    for i in range(4):
        # Acrobot draws float32 start states.
        start = g.uniform(-0.3, 0.3, 4).astype(np.float32).astype(np.float64)
        actions = [int(a) for a in g.integers(0, 3, 40)]
        # The swing is chaotic, so differences of an ulp in sin and cos grow; 1e-10 still pins every step.
        emit(
            out, f"acrobot/{i}", "acrobot", start, actions, trajectory("Acrobot-v1", start, actions, np.float32), 1e-10
        )
    return out
