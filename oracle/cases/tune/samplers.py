"""Sampler orders that the algorithms define (rather than a random stream): scikit-learn's `ParameterGrid` order, and
Optuna's unscrambled quasi-Monte Carlo samplers (`QMCSampler(qmc_type='sobol' | 'halton', scramble=False)`), whose
trials walk the sequence point by point through the search-space transform. Optuna's first trial is drawn at random
(its QMC sampler needs a finished trial to learn the space), so the fixture keeps the trials after it."""

import warnings

import optuna
import sklearn
from optuna.samplers import QMCSampler
from sklearn.model_selection import ParameterGrid
from whitebox_oracle.fixture import Case, Json, case

REFERENCE: dict[str, str] = {"optuna": optuna.__version__, "scikit-learn": sklearn.__version__}

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

optuna.logging.set_verbosity(optuna.logging.ERROR)

type Grid = dict[str, list[str | int | float | bool]]


def grid_case(name: str, grids: list[Grid]) -> Case:
    order: list[Json] = [{k: v for k, v in p.items()} for p in ParameterGrid(grids)]
    inputs: list[Json] = [{k: list(v) for k, v in g.items()} for g in grids]
    return case(name, "grid", [inputs], order)


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -2.0, 3.0)
    y = trial.suggest_float("y", 1e-3, 10.0, log=True)
    k = trial.suggest_int("k", 0, 8)
    c = trial.suggest_categorical("c", ["a", "b", "c", "d"])
    return x + y + k + len(c)


def qmc_case(name: str, kind: str, n: int) -> Case:
    study = optuna.create_study(
        sampler=QMCSampler(qmc_type=kind, scramble=False, seed=0, warn_independent_sampling=False)
    )
    study.optimize(objective, n_trials=n + 1)
    params: list[Json] = [{k: v for k, v in t.params.items()} for t in study.trials[1:]]
    return case(name, kind, [], params, rtol=1e-12, atol=1e-12, n=n)


def cases() -> list[Case]:
    return [
        grid_case("grid/single", [{"b": [1, 2], "a": ["x", "y", "z"], "c": [True, False]}]),
        grid_case(
            "grid/list",
            [{"kernel": ["linear"], "C": [1, 10]}, {"kernel": ["rbf"], "C": [1, 10], "gamma": [0.1, 1.0]}],
        ),
        qmc_case("qmc/sobol", "sobol", 32),
        qmc_case("qmc/halton", "halton", 20),
    ]
