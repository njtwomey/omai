"""Multi-objective study results against Optuna: the best trials (the Pareto front of complete trials, with maximised
objectives) and the hypervolume of the complete trials (`optuna._hypervolume.compute_hypervolume` on the loss
vectors), for random value tables with ties, in two and three objectives."""

from typing import Literal

import numpy as np
import optuna
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from optuna._hypervolume import compute_hypervolume  # pyright: ignore[reportPrivateUsage]
from optuna.trial import TrialState, create_trial

REFERENCE: dict[str, str] = {"optuna": optuna.__version__}

optuna.logging.set_verbosity(optuna.logging.ERROR)


def values(seed: int, n: int, m: int, decimals: int) -> FloatArray:
    return np.asarray(np.round(rng(seed).uniform(0, 1, size=(n, m)), decimals), dtype=np.float64)


def front_case(
    name: str, seed: int, n: int, directions: list[Literal["minimize", "maximize"]], decimals: int, failed: int = 0
) -> Case:
    v = values(seed, n, len(directions), decimals)
    trials = [create_trial(values=[float(x) for x in row]) for row in v]
    trials += [create_trial(state=TrialState.FAIL) for _ in range(failed)]
    study = optuna.create_study(directions=directions)
    study.add_trials(trials)
    best = sorted(t.number for t in study.best_trials)
    signs = np.asarray([1.0 if d == "minimize" else -1.0 for d in directions])
    losses = v * signs
    reference = np.max(losses, axis=0) + 0.1
    hv = float(compute_hypervolume(losses, reference))
    return case(
        name,
        "front",
        [v],
        {"best": best, "hypervolume": hv},
        rtol=1e-12,
        atol=1e-12,
        directions=directions,
        reference=[float(r * s) for r, s in zip(reference, signs, strict=True)],
        failed=failed,
    )


def cases() -> list[Case]:
    return [
        front_case("front/2d", 0, 40, ["minimize", "minimize"], 6),
        front_case("front/2d/ties", 1, 40, ["minimize", "minimize"], 1, failed=2),
        front_case("front/2d/max", 2, 30, ["maximize", "minimize"], 3),
        front_case("front/3d", 3, 50, ["minimize", "maximize", "minimize"], 2),
        front_case("front/4d", 4, 30, ["minimize", "minimize", "minimize", "minimize"], 3),
    ]
