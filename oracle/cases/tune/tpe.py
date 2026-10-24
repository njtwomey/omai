"""TPE against Optuna's own estimator: for a fixed history of trials over a mixed space (reals, log reals, integers,
log integers, a stepped real and an option), the split into good and bad groups, the log densities of the Parzen
estimators of both groups (multivariate and per parameter, with Optuna's default and alternative settings) at fixed
points, and the multi-objective split with its hypervolume tie-breaking. Everything here is deterministic given the
history, so the comparison is exact up to floating point.

The histories have no repeated values of a parameter: Optuna orders tied observations with numpy's default argsort,
whose order among equal values depends on the platform's sort kernel, and the order decides which tied observation gets
which bandwidth. whitebox-js breaks ties by trial order (a stable sort)."""

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.samplers._tpe.parzen_estimator import (  # pyright: ignore[reportPrivateUsage]
    _ParzenEstimator,  # pyright: ignore[reportPrivateUsage]
    _ParzenEstimatorParameters,  # pyright: ignore[reportPrivateUsage]
)
from optuna.samplers._tpe.sampler import (  # pyright: ignore[reportPrivateUsage]
    _split_trials,  # pyright: ignore[reportPrivateUsage]
    default_gamma,
    default_weights,
)
from optuna.trial import FrozenTrial, TrialState, create_trial
from whitebox_oracle.fixture import Case, Json, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"optuna": optuna.__version__}

optuna.logging.set_verbosity(optuna.logging.ERROR)

OPTIONS = ["a", "b", "c"]
SPACE: dict[str, BaseDistribution] = {
    "x": FloatDistribution(-5.0, 5.0),
    "lr": FloatDistribution(1e-4, 1.0, log=True),
    "k": IntDistribution(1, 1000),
    "kl": IntDistribution(1, 100000, log=True),
    "s": FloatDistribution(0.0, 50.0, step=0.5),
    "act": CategoricalDistribution(OPTIONS),
}

type Params = dict[str, float | int | str]


def draw(g: np.random.Generator) -> Params:
    return {
        "x": float(g.uniform(-5, 5)),
        "lr": float(math.exp(g.uniform(math.log(1e-4), 0.0))),
        "k": int(g.integers(1, 1001)),
        "kl": int(min(100000, math.floor(math.exp(g.uniform(0.0, math.log(100001.0)))))),
        "s": float(0.5 * g.integers(0, 101)),
        "act": OPTIONS[int(g.integers(0, 3))],
    }


def draw_distinct(g: np.random.Generator, seen: dict[str, set[float]]) -> Params:
    """A draw whose numeric values all differ from earlier draws (see the module note on ties)."""
    while True:
        p = draw(g)
        if all(float(p[k]) not in seen[k] for k in ["x", "lr", "k", "kl", "s"]):
            for k in ["x", "lr", "k", "kl", "s"]:
                seen[k].add(float(p[k]))
            return p


def value_of(p: Params) -> float:
    x = float(p["x"])
    lr = float(p["lr"])
    act = str(p["act"])
    return (x - 1) ** 2 + math.log10(lr / 0.01) ** 2 + 1e-3 * abs(int(p["k"]) - 400) + (0.0 if act == "b" else 0.5)


def history(seed: int, n: int, pruned: int = 0) -> list[FrozenTrial]:
    g = rng(seed)
    seen: dict[str, set[float]] = {k: set() for k in ["x", "lr", "k", "kl", "s"]}
    trials: list[FrozenTrial] = []
    for i in range(n):
        p = draw_distinct(g, seen)
        if i < pruned:
            steps = int(g.integers(1, 5))
            iv = {s: value_of(p) + float(g.uniform(0, 3)) for s in range(steps)}
            trials.append(create_trial(params=p, distributions=SPACE, intermediate_values=iv, state=TrialState.PRUNED))
        else:
            trials.append(create_trial(params=p, distributions=SPACE, value=value_of(p)))
    return trials


def study_of(trials: list[FrozenTrial], directions: list[Literal["minimize", "maximize"]]) -> optuna.Study:
    study = optuna.create_study(directions=directions)
    study.add_trials(trials)
    return study


def internal(trials: list[FrozenTrial], names: list[str]) -> dict[str, FloatArray]:
    return {n: np.asarray([SPACE[n].to_internal_repr(t.params[n]) for t in trials], dtype=np.float64) for n in names}


def trial_json(t: FrozenTrial) -> dict[str, Json]:
    return {
        "number": t.number,
        "params": {k: v for k, v in t.params.items()},
        "state": "complete" if t.state == TrialState.COMPLETE else "pruned",
        "values": [float(v) for v in t.values] if t.values is not None else [],
        "intermediate": [{"step": s, "value": float(v)} for s, v in sorted(t.intermediate_values.items())],
    }


def points(seed: int, n: int) -> list[Params]:
    g = rng(seed)
    return [draw(g) for _ in range(n)]


def parameters(
    prior_weight: float = 1.0,
    magic_clip: bool = True,
    endpoints: bool = False,
    weights: Callable[[int], FloatArray] = default_weights,
) -> _ParzenEstimatorParameters:
    return _ParzenEstimatorParameters(
        prior_weight=prior_weight,
        consider_magic_clip=magic_clip,
        consider_endpoints=endpoints,
        weights=weights,
        multivariate=True,
    )


def flat_weights(n: int) -> FloatArray:
    return np.ones(n)


def estimator_case(
    name: str,
    seed: int,
    n: int,
    names: list[str],
    params: _ParzenEstimatorParameters,
    pruned: int = 0,
) -> Case:
    trials = history(seed, n, pruned)
    study = study_of(trials, ["minimize"])
    below, above = _split_trials(study, study.trials, default_gamma(n))
    space = {k: SPACE[k] for k in names}
    l_est = _ParzenEstimator(internal(below, names), space, params)
    g_est = _ParzenEstimator(internal(above, names), space, params)
    at = points(seed + 100, 12)
    samples = internal([create_trial(params=p, distributions=SPACE, value=0.0) for p in at], names)
    lb = l_est.log_pdf(samples)
    la = g_est.log_pdf(samples)
    out: dict[str, Json] = {
        "below": [t.number for t in below],
        "logPdfBelow": [float(v) for v in lb],
        "logPdfAbove": [float(v) for v in la],
    }
    return case(
        name,
        "parzen",
        [[trial_json(t) for t in study.trials], [{k: p[k] for k in names} for p in at]],
        out,
        rtol=1e-10,
        atol=1e-10,
        names=names,
        priorWeight=params.prior_weight,
        considerMagicClip=params.consider_magic_clip,
        considerEndpoints=params.consider_endpoints,
        flatWeights=params.weights is flat_weights,
    )


def mo_values(seed: int, n: int, m: int) -> FloatArray:
    g = rng(seed)
    return np.asarray(np.round(g.uniform(0, 1, size=(n, m)), 3), dtype=np.float64)


def mo_case(
    name: str, seed: int, n: int, m: int, directions: list[Literal["minimize", "maximize"]], n_below: int
) -> Case:
    values = mo_values(seed, n, m)
    g = rng(seed + 1)
    trials = [create_trial(params=draw(g), distributions=SPACE, values=[float(v) for v in row]) for row in values]
    study = study_of(trials, directions)
    below, _ = _split_trials(study, study.trials, n_below)
    return case(
        name,
        "splitMultiObjective",
        [[trial_json(t) for t in study.trials]],
        [t.number for t in below],
        directions=directions,
        nBelow=n_below,
    )


def cases() -> list[Case]:
    names = list(SPACE)
    out: list[Case] = [
        case(
            "gamma",
            "defaultGamma",
            [[0, 1, 9, 10, 11, 100, 249, 250, 251, 1000]],
            [default_gamma(n) for n in [0, 1, 9, 10, 11, 100, 249, 250, 251, 1000]],
        ),
        case("weights/24", "defaultWeights", [], default_weights(24), n=24),
        case("weights/40", "defaultWeights", [], default_weights(40), n=40, rtol=1e-12),
        estimator_case("parzen/multivariate", 0, 30, names, parameters()),
        estimator_case("parzen/multivariate/pruned", 1, 30, names, parameters(), pruned=6),
        estimator_case("parzen/multivariate/mostly-pruned", 9, 30, names, parameters(), pruned=28),
        estimator_case("parzen/endpoints", 2, 20, names, parameters(endpoints=True)),
        estimator_case("parzen/noclip/prior", 3, 20, names, parameters(prior_weight=0.5, magic_clip=False)),
        estimator_case("parzen/flat", 4, 60, names, parameters(weights=flat_weights)),
    ]
    for n in names:
        out.append(estimator_case(f"parzen/univariate/{n}", 5, 40, [n], parameters()))
    out += [
        mo_case("split/mo/2d", 6, 30, 2, ["minimize", "minimize"], 3),
        mo_case("split/mo/2d/max", 7, 30, 2, ["minimize", "maximize"], 5),
        mo_case("split/mo/3d", 8, 40, 3, ["minimize", "minimize", "minimize"], 4),
    ]
    return out
