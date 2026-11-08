"""Dual averaging, Stan's warm-up schedule and Welford mass-matrix adaptation against BlackJAX (float64)."""

import importlib.metadata

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.mass_matrix import mass_matrix_adaptation
from blackjax.adaptation.step_size import dual_averaging_adaptation
from blackjax.adaptation.window_adaptation import build_schedule
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

jax.config.update("jax_enable_x64", True)

REFERENCE: dict[str, str] = {
    "blackjax": importlib.metadata.version("blackjax"),
    "jax": importlib.metadata.version("jax"),
}


def dual_averaging_path(acceptance: FloatArray, initial: float, target: float) -> FloatArray:
    """[log step size, log averaged step size] after each update, from BlackJAX's dual averaging."""
    init, update, _ = dual_averaging_adaptation(target)
    state = init(initial)
    rows: list[list[float]] = []
    for a in acceptance:
        state = update(state, float(a))
        rows.append([float(state.log_step_size), float(state.log_step_size_avg)])
    return np.asarray(rows, dtype=np.float64)


def mass_matrix(draws: FloatArray, dense: bool) -> FloatArray:
    """The regularised inverse mass matrix BlackJAX computes from one window of draws."""
    init, update, final = mass_matrix_adaptation(is_diagonal_matrix=not dense)
    state = init(draws.shape[1])
    for x in draws:
        state = update(state, jnp.asarray(x))
    return np.asarray(final(state).inverse_mass_matrix, dtype=np.float64)


def cases() -> list[Case]:
    g = rng(3)
    out: list[Case] = []
    acceptance = g.uniform(0.0, 1.0, size=40)
    for initial, target in [(1.0, 0.8), (0.05, 0.65), (3.0, 0.9)]:
        path = dual_averaging_path(acceptance, initial, target)
        out.append(
            case(
                f"dualAveraging/{initial}/{target}",
                "dualAveraging",
                [acceptance],
                path,
                initialStepSize=initial,
                targetAcceptance=target,
            )
        )
    for n in [0, 10, 19, 20, 37, 100, 150, 151, 249, 500, 1000, 2000]:
        schedule = np.asarray(build_schedule(n), dtype=np.float64).reshape(n, 2)
        out.append(case(f"windowSchedule/{n}", "windowSchedule", [], schedule, numSteps=n))
    draws = g.normal(size=(30, 3)) @ np.array([[1.0, 0.5, 0.0], [0.0, 2.0, 0.3], [0.0, 0.0, 0.7]])
    for dense in [False, True]:
        out.append(
            case(
                f"massMatrix/{'dense' if dense else 'diagonal'}",
                "massMatrix",
                [draws],
                mass_matrix(draws, dense),
                dense=dense,
            )
        )
    return out
