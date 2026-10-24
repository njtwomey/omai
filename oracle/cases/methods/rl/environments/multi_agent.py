"""Cooperative navigation against a numpy transcription of the multi-agent particle environment (MPE `core.World.step`
and `scenarios/simple_spread.py` as packaged by PettingZoo; PettingZoo itself is not installed): damped point masses,
softplus contact forces between agents, and the shared coverage reward mixed with collision penalties."""

import numpy as np
import numpy.typing as npt
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "transcription": "MPE core.py and simple_spread.py"}

type Array = npt.NDArray[np.float64]

DT = 0.1
DAMPING = 0.25
SENSITIVITY = 5.0
SIZE = 0.15
CONTACT_FORCE = 1e2
CONTACT_MARGIN = 1e-3
MOVES = np.array([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])


def world_step(pos: Array, vel: Array, u: Array) -> tuple[Array, Array]:
    """MPE's World.step for movable, colliding agents of unit mass."""
    force = u * SENSITIVITY
    n = pos.shape[0]
    for a in range(n):
        for b in range(a + 1, n):
            delta = pos[a] - pos[b]
            dist = float(np.sqrt(np.sum(np.square(delta))))
            dist_min = 2 * SIZE
            k = CONTACT_MARGIN
            penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
            f = CONTACT_FORCE * delta / dist * penetration
            force[a] = force[a] + f
            force[b] = force[b] - f
    vel = vel * (1 - DAMPING) + force * DT
    pos = pos + vel * DT
    return pos, vel


def rewards(pos: Array, landmarks: Array, local_ratio: float) -> Array:
    """simple_spread's per-agent rewards: coverage times (1 - ratio) plus collision penalties times ratio."""
    team = -sum(float(np.min(np.linalg.norm(pos - lm, axis=1))) for lm in landmarks)
    out = []
    for i in range(pos.shape[0]):
        collisions = sum(1 for j in range(pos.shape[0]) if j != i and float(np.linalg.norm(pos[i] - pos[j])) < 2 * SIZE)
        out.append(team * (1 - local_ratio) - collisions * local_ratio)
    return np.array(out)


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(91)
    for n in (2, 3):
        for trial in range(3):
            pos = g.uniform(-0.4, 0.4, size=(n, 2))
            vel = np.zeros((n, 2))
            landmarks = 0.8 * g.uniform(-1, 1, size=(n, 2))
            actions = g.integers(0, 5, size=(8, n))
            rows: list[Array] = []
            start_pos = pos.copy()
            for t in range(actions.shape[0]):
                pos, vel = world_step(pos, vel, MOVES[actions[t]])
                rows.append(np.concatenate([pos.ravel(), vel.ravel(), rewards(pos, landmarks, 0.5)]))
            out.append(
                case(
                    f"spread/{n}/{trial}",
                    "spread",
                    [],
                    np.array(rows),
                    rtol=1e-10,
                    atol=1e-12,
                    positions=start_pos.tolist(),
                    landmarks=landmarks.tolist(),
                    actions=actions.tolist(),
                )
            )
    return out
