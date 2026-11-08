"""Learning-rate schedules (optim/schedules.ts) against torch.optim.lr_scheduler and optax schedules."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import torch
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray
from torch.optim import lr_scheduler

REFERENCE: dict[str, str] = {"torch": torch.__version__, "optax": optax.__version__}

jax.config.update("jax_enable_x64", True)

LR = 0.5
STEPS = 40

type SchedulerFactory = Callable[[torch.optim.Optimizer], lr_scheduler.LRScheduler]


def torch_rates(make: SchedulerFactory, steps: int = STEPS, momentum: float = 0.0) -> FloatArray:
    """The learning rate torch reports at steps 0..steps-1."""
    p = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([p], lr=LR, momentum=momentum)
    sched = make(opt)
    out: list[float] = []
    for _ in range(steps):
        out.append(float(sched.get_last_lr()[0]))
        opt.step()
        sched.step()
    return np.asarray(out, dtype=np.float64)


def optax_rates(schedule: optax.Schedule, steps: int = STEPS) -> FloatArray:
    """An optax schedule at steps 0..steps-1."""
    return np.asarray([np.asarray(schedule(jnp.asarray(t)), dtype=np.float64) for t in range(steps)], dtype=np.float64)


TORCH: list[tuple[str, dict[str, Any], SchedulerFactory]] = [
    ("constantFactor", {"init": LR, "factor": 0.25, "steps": 7}, lambda o: lr_scheduler.ConstantLR(o, 0.25, 7)),
    ("stepDecay", {"init": LR, "stepSize": 6, "gamma": 0.5}, lambda o: lr_scheduler.StepLR(o, 6, 0.5)),
    (
        "multiStepDecay",
        {"init": LR, "milestones": [5, 12, 30], "gamma": 0.3},
        lambda o: lr_scheduler.MultiStepLR(o, [5, 12, 30], 0.3),
    ),
    ("exponentialDecay", {"init": LR, "decayRate": 0.9, "steps": 1}, lambda o: lr_scheduler.ExponentialLR(o, 0.9)),
    (
        "linearSchedule",
        {"init": LR / 3, "end": LR, "steps": 10},
        lambda o: lr_scheduler.LinearLR(o, start_factor=1 / 3, end_factor=1.0, total_iters=10),
    ),
    (
        "polynomialDecay",
        {"init": LR, "end": 0, "power": 2.0, "steps": 25},
        lambda o: lr_scheduler.PolynomialLR(o, total_iters=25, power=2.0),
    ),
    (
        "cosineDecay",
        {"init": LR, "decaySteps": 39, "alpha": 0.1},
        lambda o: lr_scheduler.CosineAnnealingLR(o, T_max=39, eta_min=0.05),
    ),
    (
        "cosineWarmRestarts",
        {"init": LR, "t0": 7, "tMult": 1, "etaMin": 0.01},
        lambda o: lr_scheduler.CosineAnnealingWarmRestarts(o, T_0=7, eta_min=0.01),
    ),
    (
        "cosineWarmRestarts",
        {"init": LR, "t0": 3, "tMult": 2, "etaMin": 0.0},
        lambda o: lr_scheduler.CosineAnnealingWarmRestarts(o, T_0=3, T_mult=2),
    ),
    (
        "oneCycle",
        {"maxLr": 1.0, "totalSteps": STEPS},
        lambda o: lr_scheduler.OneCycleLR(o, max_lr=1.0, total_steps=STEPS, cycle_momentum=False),
    ),
    (
        "oneCycle",
        {"maxLr": 1.0, "totalSteps": STEPS, "pctStart": 0.25, "anneal": "linear", "divFactor": 10, "threePhase": True},
        lambda o: lr_scheduler.OneCycleLR(
            o,
            max_lr=1.0,
            total_steps=STEPS,
            pct_start=0.25,
            anneal_strategy="linear",
            div_factor=10,
            three_phase=True,
            cycle_momentum=False,
        ),
    ),
    (
        "cyclical",
        {"baseLr": 0.1, "maxLr": 1.0, "stepSizeUp": 5, "stepSizeDown": 3},
        lambda o: lr_scheduler.CyclicLR(o, 0.1, 1.0, step_size_up=5, step_size_down=3, cycle_momentum=False),
    ),
    (
        "cyclical",
        {"baseLr": 0.1, "maxLr": 1.0, "stepSizeUp": 4, "mode": "triangular2"},
        lambda o: lr_scheduler.CyclicLR(o, 0.1, 1.0, step_size_up=4, mode="triangular2", cycle_momentum=False),
    ),
    (
        "cyclical",
        {"baseLr": 0.1, "maxLr": 1.0, "stepSizeUp": 4, "mode": "exp_range", "gamma": 0.95},
        lambda o: lr_scheduler.CyclicLR(
            o, 0.1, 1.0, step_size_up=4, mode="exp_range", gamma=0.95, cycle_momentum=False
        ),
    ),
]

OPTAX: list[tuple[str, dict[str, Any], optax.Schedule]] = [
    (
        "exponentialDecay",
        {"init": LR, "decayRate": 0.5, "steps": 7, "staircase": True, "transitionBegin": 4, "endValue": 0.05},
        optax.exponential_decay(LR, 7, 0.5, transition_begin=4, staircase=True, end_value=0.05),
    ),
    (
        "cosineDecay",
        {"init": LR, "decaySteps": 30, "alpha": 0.2, "exponent": 2.0},
        optax.cosine_decay_schedule(LR, 30, alpha=0.2, exponent=2.0),
    ),
    (
        "polynomialDecay",
        {"init": LR, "end": 0.01, "power": 1.5, "steps": 20, "begin": 5},
        optax.polynomial_schedule(LR, 0.01, 1.5, 20, transition_begin=5),
    ),
    (
        "linearSchedule",
        {"init": 0.0, "end": LR, "steps": 12, "begin": 3},
        optax.linear_schedule(0.0, LR, 12, transition_begin=3),
    ),
    (
        "warmupCosineDecay",
        {"start": 0.0, "peak": 1.0, "end": 0.1, "warmupSteps": 8, "totalSteps": 35, "exponent": 1.5},
        optax.warmup_cosine_decay_schedule(0.0, 1.0, 8, 35, end_value=0.1, exponent=1.5),
    ),
    (
        "warmupExponentialDecay",
        {"start": 0.0, "peak": 1.0, "warmupSteps": 6, "decayRate": 0.8, "transitionSteps": 3, "transitionBegin": 2},
        optax.warmup_exponential_decay_schedule(0.0, 1.0, 6, 3, 0.8, transition_begin=2),
    ),
    ("warmupConstant", {"start": 0.1, "peak": 1.0, "warmupSteps": 9}, optax.warmup_constant_schedule(0.1, 1.0, 9)),
    (
        "piecewiseInterpolate",
        {"kind": "linear", "init": 1.0, "pairs": [[5, 0.5], [15, 0.1], [30, 2.0]]},
        optax.piecewise_interpolate_schedule("linear", 1.0, {5: 0.5, 15: 0.1, 30: 2.0}),
    ),
    (
        "piecewiseInterpolate",
        {"kind": "cosine", "init": 1.0, "pairs": [[10, 0.2], [25, 0.5]]},
        optax.piecewise_interpolate_schedule("cosine", 1.0, {10: 0.2, 25: 0.5}),
    ),
    (
        "sgdr",
        {"cycles": [[0.0, 1.0, 3, 10, 0.1], [0.0, 0.5, 2, 15, 0.0], [0.1, 0.3, 0, 15, 0.05]]},
        optax.sgdr_schedule(
            [
                {"init_value": 0.0, "peak_value": 1.0, "warmup_steps": 3, "decay_steps": 10, "end_value": 0.1},
                {"init_value": 0.0, "peak_value": 0.5, "warmup_steps": 2, "decay_steps": 15, "end_value": 0.0},
                {"init_value": 0.1, "peak_value": 0.3, "warmup_steps": 0, "decay_steps": 15, "end_value": 0.05},
            ]
        ),
    ),
]


def plateau_case() -> Case:
    """ReduceLROnPlateau on a fixed metric sequence: the rate after each epoch."""
    metrics = np.array([5.0, 4.0, 3.9, 3.9, 3.95, 3.9, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5])
    p = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([p], lr=LR)
    sched = lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, threshold=0.02, cooldown=1, min_lr=0.07)
    rates: list[float] = []
    for m in metrics:
        sched.step(float(m))
        rates.append(float(opt.param_groups[0]["lr"]))
    options = {"factor": 0.5, "patience": 2, "threshold": 0.02, "cooldown": 1, "minLr": 0.07}
    return case("reduceOnPlateau/0", "reduceOnPlateau", [metrics], np.asarray(rates), init=LR, **options)


def cases() -> list[Case]:
    out: list[Case] = []
    for i, (name, params, make) in enumerate(TORCH):
        out.append(
            case(f"{name}/torch/{i}", name, [], torch_rates(make), rtol=1e-12, atol=1e-14, count=STEPS, **params)
        )
    for i, (name, params, schedule) in enumerate(OPTAX):
        out.append(
            case(f"{name}/optax/{i}", name, [], optax_rates(schedule), rtol=1e-12, atol=1e-14, count=STEPS, **params)
        )
    out.append(plateau_case())
    return out
