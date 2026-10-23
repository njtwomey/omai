"""Returns and advantages: scipy's discounting filter, and numpy transcriptions of stable-baselines3 (GAE) and rlax
(lambda returns, V-trace, Retrace)."""

import numpy as np
import numpy.typing as npt
import scipy
import scipy.signal
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}

type Array = npt.NDArray[np.float64]


def discounted_with_dones(rewards: Array, dones: Array, gamma: float, bootstrap: float) -> Array:
    out = np.zeros_like(rewards)
    g = bootstrap
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * (1 - dones[t]) * g
        out[t] = g
    return out


def sb3_gae(rewards: Array, values: Array, dones: Array, last_value: float, gamma: float, lam: float) -> Array:
    """stable-baselines3's compute_returns_and_advantage for one environment, with `dones[t]` ending step t."""
    adv = np.zeros_like(rewards)
    last = 0.0
    for step in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[step]
        next_values = last_value if step == len(rewards) - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last = delta + gamma * lam * next_non_terminal * last
        adv[step] = last
    return adv


def lambda_returns(rewards: Array, next_values: Array, dones: Array, gamma: float, lam: float) -> Array:
    """rlax.lambda_returns with discount_t = gamma (1 - done)."""
    discounts = gamma * (1 - dones)
    out = np.zeros_like(rewards)
    g = next_values[-1]
    for t in reversed(range(len(rewards))):
        g = rewards[t] + discounts[t] * ((1 - lam) * next_values[t] + lam * g)
        out[t] = g
    return out


def n_step(rewards: Array, values: Array, dones: Array, gamma: float, n: int) -> Array:
    big_t = len(rewards)
    out = np.zeros(big_t)
    for t in range(big_t):
        g, disc = 0.0, 1.0
        for k in range(n):
            i = t + k
            if i >= big_t:
                g += disc * values[big_t]
                break
            g += disc * rewards[i]
            disc *= gamma
            if dones[i]:
                break
        else:
            g += disc * values[t + n]
        out[t] = g
    return out


def vtrace(
    rewards: Array, discounts: Array, values: Array, bootstrap: float, log_rhos: Array, clip_rho: float, lam: float
) -> tuple[Array, Array]:
    """rlax.vtrace_td_error_and_advantage (clip_pg_rho_threshold = 1)."""
    rhos = np.exp(log_rhos)
    clipped = np.minimum(clip_rho, rhos)
    cs = lam * np.minimum(1.0, rhos)
    v_next = np.append(values[1:], bootstrap)
    deltas = clipped * (rewards + discounts * v_next - values)
    acc = 0.0
    vs_minus_v = np.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        acc = deltas[t] + discounts[t] * cs[t] * acc
        vs_minus_v[t] = acc
    vs = vs_minus_v + values
    vs_next = np.append(vs[1:], bootstrap)
    pg = np.minimum(1.0, rhos) * (rewards + discounts * vs_next - values)
    return vs, pg


def retrace(rewards: Array, discounts: Array, v_next: Array, q_next: Array, c: Array) -> Array:
    """rlax.general_off_policy_returns_from_q_and_v."""
    out = np.zeros_like(rewards)
    g = rewards[-1] + discounts[-1] * v_next[-1]
    out[-1] = g
    for t in reversed(range(len(rewards) - 1)):
        g = rewards[t] + discounts[t] * (v_next[t] - c[t] * q_next[t] + c[t] * g)
        out[t] = g
    return out


def cases() -> list[Case]:
    g = rng(74)
    out: list[Case] = []
    for i in range(3):
        n = 25
        rewards = g.normal(size=n)
        dones = (g.uniform(size=n) < 0.15).astype(np.float64)
        values = g.normal(size=n + 1)
        gamma, lam = 0.97, 0.9
        filtered = scipy.signal.lfilter([1.0], [1.0, -gamma], rewards[::-1])[::-1]
        out.append(case(f"discounted/{i}", "discountedReturns", [rewards], np.asarray(filtered), gamma=gamma))
        out.append(
            case(
                f"discountedDones/{i}",
                "discountedReturnsDones",
                [rewards, dones],
                discounted_with_dones(rewards, dones, gamma, float(values[-1])),
                gamma=gamma,
                bootstrap=float(values[-1]),
            )
        )
        adv = sb3_gae(rewards, values[:n], dones, float(values[n]), gamma, lam)
        out.append(case(f"gae/{i}", "gae", [rewards, values, dones], adv, gamma=gamma, lam=lam))
        out.append(
            case(
                f"lambda/{i}",
                "lambdaReturns",
                [rewards, values, dones],
                lambda_returns(rewards, values[1:], dones, gamma, lam),
                gamma=gamma,
                lam=lam,
            )
        )
        for k in (1, 3, 7):
            out.append(
                case(
                    f"nstep/{i}/{k}",
                    "nStepReturns",
                    [rewards, values, dones],
                    n_step(rewards, values, dones, gamma, k),
                    gamma=gamma,
                    n=k,
                )
            )
        discounts = gamma * (1 - dones)
        log_rhos = g.normal(scale=0.5, size=n)
        vs, pg = vtrace(rewards, discounts, values[:n], float(values[n]), log_rhos, 1.0, 0.95)
        out.append(case(f"vtrace/{i}/vs", "vtraceVs", [rewards, discounts, values, log_rhos], vs, lam=0.95))
        out.append(case(f"vtrace/{i}/pg", "vtracePg", [rewards, discounts, values, log_rhos], pg, lam=0.95))
        q_next = g.normal(size=n - 1)
        c = 0.9 * np.minimum(1.0, np.exp(g.normal(scale=0.5, size=n - 1)))
        out.append(
            case(
                f"retrace/{i}",
                "retrace",
                [rewards, discounts, values[1:], q_next, c],
                retrace(rewards, discounts, values[1:], q_next, c),
            )
        )
    return out
