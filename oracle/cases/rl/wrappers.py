"""gymnasium's normalising and rescaling wrappers on scripted environments (float64 spaces, so statistics are exact)."""

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.wrappers import NormalizeObservation, NormalizeReward, RescaleAction
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"gymnasium": gym.__version__}

type Array = npt.NDArray[np.float64]


class Scripted(gym.Env[Array, Array]):
    """Replays given observations, rewards and termination flags, and records the actions it receives."""

    def __init__(
        self, observations: Array, rewards: Array, terminated: Array, low: float = -1.0, high: float = 1.0
    ) -> None:
        self.observations = observations
        self.rewards = rewards
        self.terminated = terminated
        self.t = 0
        self.actions: list[Array] = []
        dim = observations.shape[1]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float64)
        self.action_space = gym.spaces.Box(low, high, shape=(2,), dtype=np.float64)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Array, dict[str, Any]]:
        obs = self.observations[self.t]
        self.t += 1
        return obs, {}

    def step(self, action: Array) -> tuple[Array, float, bool, bool, dict[str, Any]]:
        self.actions.append(np.asarray(action, dtype=np.float64))
        obs = self.observations[self.t]
        reward = float(self.rewards[self.t - 1])
        terminated = bool(self.terminated[self.t - 1])
        self.t += 1
        return obs, reward, terminated, False, {}


def cases() -> list[Case]:
    g = rng(73)
    out: list[Case] = []
    for i in range(3):
        n = 40
        obs = g.normal(2.0, 3.0, size=(n + 1, 3))
        rewards = g.normal(1.0, 2.0, size=n)
        terminated = (g.uniform(size=n) < 0.1).astype(np.float64)
        env = NormalizeReward(Scripted(obs, rewards, terminated), gamma=0.9)
        env.reset()
        normalised = [float(env.step(np.zeros(2))[1]) for _ in range(n)]
        out.append(
            case(f"normalizeReward/{i}", "normalizeReward", [rewards, terminated], np.array(normalised), gamma=0.9)
        )
        # NormalizeObservation updates on the reset observation too; the wrapper returns float32 copies, so compare
        # the float64 running statistics exactly and the normalised observations at float32 precision.
        envo = NormalizeObservation(Scripted(obs, rewards, np.zeros(n)))
        first, _ = envo.reset()
        seen = [np.asarray(first, dtype=np.float64)] + [
            np.asarray(envo.step(np.zeros(2))[0], dtype=np.float64) for _ in range(n)
        ]
        rms = envo.obs_rms
        out.append(
            case(f"normalizeObservation/{i}/obs", "normalizeObservation", [obs], np.stack(seen), rtol=1e-6, atol=1e-6)
        )
        out.append(case(f"normalizeObservation/{i}/mean", "normalizeObservationMean", [obs], np.asarray(rms.mean)))
        out.append(case(f"normalizeObservation/{i}/var", "normalizeObservationVar", [obs], np.asarray(rms.var)))
    inner = Scripted(np.zeros((6, 1)), np.zeros(5), np.zeros(5), low=-2.0, high=3.0)
    envr = RescaleAction(inner, np.float64(-1.0), np.float64(1.0))
    envr.reset()
    actions = g.uniform(-1, 1, size=(5, 2))
    for a in actions:
        envr.step(a)
    out.append(case("rescaleAction", "rescaleAction", [actions], np.stack(inner.actions), low=-2.0, high=3.0))
    return out
