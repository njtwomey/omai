"""Pruning decisions against Optuna's pruners: a fixed matrix of learning curves (one row per trial, reported step by
step, with a few NaN reports) is run through `study.optimize` with each pruner, and the step at which each trial was
pruned (or -1) is recorded. The decisions are deterministic given the reports and the trial order, including
Hyperband's bracket assignment (a CRC-32 hash of the study name and trial number)."""

import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
import optuna
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from optuna.pruners import (
    BasePruner,
    HyperbandPruner,
    MedianPruner,
    PatientPruner,
    PercentilePruner,
    SuccessiveHalvingPruner,
    ThresholdPruner,
)

REFERENCE: dict[str, str] = {"optuna": optuna.__version__}

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

optuna.logging.set_verbosity(optuna.logging.ERROR)


def curves(seed: int, n: int, steps: int, nans: int = 0) -> FloatArray:
    g = rng(seed)
    a = g.uniform(0.5, 3.0, size=(n, 1))
    b = g.uniform(0.05, 0.5, size=(n, 1))
    c = g.uniform(0.0, 1.0, size=(n, 1))
    s = np.arange(steps)[np.newaxis, :]
    out = a * np.exp(-b * s) + c + g.normal(0, 0.02, size=(n, steps))
    for _ in range(nans):
        out[int(g.integers(0, n)), int(g.integers(0, steps))] = np.nan
    return np.asarray(np.round(out, 6), dtype=np.float64)


def decisions(
    pruner: BasePruner, values: FloatArray, direction: Literal["minimize", "maximize"], stride: int, name: str
) -> tuple[list[int], list[str]]:
    pruned_at: list[int] = []

    def objective(trial: optuna.Trial) -> float:
        row = values[trial.number]
        for i, v in enumerate(row):
            step = i * stride
            trial.report(float(v), step)
            if trial.should_prune():
                pruned_at.append(step)
                raise optuna.TrialPruned()
        pruned_at.append(-1)
        return float(row[-1])

    study = optuna.create_study(
        study_name=name, direction=direction, pruner=pruner, sampler=optuna.samplers.RandomSampler(seed=0)
    )
    study.optimize(objective, n_trials=len(values), catch=())
    return pruned_at, [t.state.name.lower() for t in study.trials]


type Make = Callable[[], BasePruner]

CONFIGS: list[tuple[str, Make, dict[str, object]]] = [
    ("median", lambda: MedianPruner(), {"kind": "median"}),
    (
        "median/warmup",
        lambda: MedianPruner(n_startup_trials=2, n_warmup_steps=3, interval_steps=2),
        {"kind": "median", "nStartupTrials": 2, "nWarmupSteps": 3, "intervalSteps": 2},
    ),
    ("percentile/25", lambda: PercentilePruner(25.0), {"kind": "percentile", "percentile": 25}),
    (
        "percentile/75/min",
        lambda: PercentilePruner(75.0, n_startup_trials=3, n_min_trials=3),
        {"kind": "percentile", "percentile": 75, "nStartupTrials": 3, "nMinTrials": 3},
    ),
    (
        "threshold",
        lambda: ThresholdPruner(lower=0.2, upper=2.5, n_warmup_steps=1),
        {"kind": "threshold", "lower": 0.2, "upper": 2.5, "nWarmupSteps": 1},
    ),
    (
        "patient",
        lambda: PatientPruner(None, patience=2, min_delta=0.01),
        {"kind": "patient", "patience": 2, "minDelta": 0.01},
    ),
    (
        "patient/median",
        lambda: PatientPruner(MedianPruner(n_startup_trials=3), patience=1),
        {"kind": "patientMedian", "patience": 1, "nStartupTrials": 3},
    ),
    (
        "halving/1/2",
        lambda: SuccessiveHalvingPruner(min_resource=1, reduction_factor=2),
        {"kind": "halving", "minResource": 1, "reductionFactor": 2},
    ),
    ("halving/auto", lambda: SuccessiveHalvingPruner(), {"kind": "halving", "minResource": "auto"}),
    (
        "halving/2/3/rate/bootstrap",
        lambda: SuccessiveHalvingPruner(
            min_resource=2, reduction_factor=3, min_early_stopping_rate=1, bootstrap_count=2
        ),
        {"kind": "halving", "minResource": 2, "reductionFactor": 3, "minEarlyStoppingRate": 1, "bootstrapCount": 2},
    ),
    (
        "hyperband/27",
        lambda: HyperbandPruner(min_resource=1, max_resource=27, reduction_factor=3),
        {"kind": "hyperband", "minResource": 1, "maxResource": 27, "reductionFactor": 3},
    ),
    ("hyperband/auto", lambda: HyperbandPruner(), {"kind": "hyperband", "maxResource": "auto"}),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for i, (label, make, params) in enumerate(CONFIGS):
        directions: list[Literal["minimize", "maximize"]] = ["minimize", "maximize"]
        for direction in directions:
            stride = 2 if label == "median/warmup" else 1
            steps = 28 if label.startswith("hyperband") else 16
            values = curves(10 + i, 30, steps, nans=3)
            if direction == "maximize":
                values = -values
            name = f"tune-{label}-{direction}"
            pruned, states = decisions(make(), values, direction, stride, name)
            out.append(
                case(
                    f"{label}/{direction}",
                    "prune",
                    [values],
                    {"prunedAt": pruned, "states": states},
                    direction=direction,
                    stride=stride,
                    studyName=name,
                    config=params,
                )
            )
    # The bracket budgets and assignments of Hyperband.
    hb = HyperbandPruner(min_resource=1, max_resource=81, reduction_factor=3)
    study = optuna.create_study(study_name="brackets", pruner=hb)
    hb._try_initialization(study)  # pyright: ignore[reportPrivateUsage]
    budgets = list(hb._trial_allocation_budgets)  # pyright: ignore[reportPrivateUsage]
    frozen = [optuna.trial.create_trial(value=0.0) for _ in range(40)]
    for number, t in enumerate(frozen):
        t.number = number
    brackets = [hb._get_bracket_id(study, t) for t in frozen]  # pyright: ignore[reportPrivateUsage]
    out.append(
        case("hyperband/brackets", "brackets", [], {"budgets": budgets, "brackets": brackets}, studyName="brackets")
    )
    return out
