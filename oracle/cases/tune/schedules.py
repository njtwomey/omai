"""Successive-halving schedules against scikit-learn: the iterations, candidates and resources that
`HalvingGridSearchCV` and `HalvingRandomSearchCV` choose (`n_iterations_`, `n_required_iterations_`,
`n_possible_iterations_`, `min_resources_`, `max_resources_`, `n_candidates_`, `n_resources_`), read off fitted searches
of dummy estimators, so the schedule is exactly scikit-learn's whatever the scores."""

import warnings

import numpy as np
import sklearn
from omai_oracle.fixture import Case, Json, case
from omai_oracle.helpers import rng
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_halving_search_cv  # pyright: ignore[reportUnusedImport]  # noqa: F401
from sklearn.linear_model import Ridge
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def attributes(search: HalvingGridSearchCV | HalvingRandomSearchCV) -> dict[str, Json]:
    return {
        "nIterations": int(search.n_iterations_),
        "nRequiredIterations": int(search.n_required_iterations_),
        "nPossibleIterations": int(search.n_possible_iterations_),
        "minResources": int(search.min_resources_),
        "maxResources": int(search.max_resources_),
        "nCandidates": [int(v) for v in search.n_candidates_],
        "nResources": [int(v) for v in search.n_resources_],
    }


def grid_case(
    name: str,
    n_candidates: int,
    n_samples: int,
    min_resources: int | str,
    factor: int,
    aggressive: bool,
    cv: int = 5,
    classes: int = 0,
) -> Case:
    g = rng(n_candidates + n_samples)
    x = g.normal(size=(n_samples, 2))
    if classes > 0:
        y = np.arange(n_samples) % classes
        est = DummyClassifier(strategy="constant", constant=0)
        grid = {"constant": [0], "random_state": list(range(n_candidates))}
    else:
        y = g.normal(size=n_samples)
        est = DummyRegressor(strategy="constant")
        grid = {"constant": [float(c) for c in range(n_candidates)]}
    search = HalvingGridSearchCV(
        est,
        grid,
        factor=factor,
        min_resources=min_resources,  # pyright: ignore[reportArgumentType]
        aggressive_elimination=aggressive,
        cv=cv,
        random_state=0,
    ).fit(x, y)
    return case(
        name,
        "halvingGrid",
        [],
        attributes(search),
        nCandidates=n_candidates,
        nSamples=n_samples,
        minResources=min_resources,
        factor=factor,
        aggressiveElimination=aggressive,
        cv=cv,
        classes=classes,
    )


def random_case(name: str, n_samples: int, factor: int, cv: int, min_resources: int | str) -> Case:
    g = rng(n_samples)
    x = g.normal(size=(n_samples, 2))
    y = g.normal(size=n_samples)
    search = HalvingRandomSearchCV(
        DummyRegressor(strategy="constant"),
        {"constant": list(np.linspace(-1, 1, 50))},
        n_candidates="exhaust",
        factor=factor,
        min_resources=min_resources,  # pyright: ignore[reportArgumentType]
        cv=cv,
        random_state=0,
    ).fit(x, y)
    return case(
        name,
        "halvingRandom",
        [],
        attributes(search),
        nSamples=n_samples,
        factor=factor,
        cv=cv,
        minResources=min_resources,
    )


def resource_case(name: str, n_candidates: int, max_resources: int, factor: int) -> Case:
    g = rng(3)
    x = g.normal(size=(40, 2))
    y = g.normal(size=40)
    search = HalvingGridSearchCV(
        Ridge(solver="sag"),
        {"alpha": [float(a) for a in np.logspace(-2, 2, n_candidates)]},
        resource="max_iter",
        max_resources=max_resources,  # pyright: ignore[reportArgumentType]
        min_resources="exhaust",
        factor=factor,
        cv=3,
    ).fit(x, y)
    return case(
        name,
        "halvingGrid",
        [],
        attributes(search),
        nCandidates=n_candidates,
        resource="max_iter",
        maxResources=max_resources,
        minResources="exhaust",
        factor=factor,
        aggressiveElimination=False,
        cv=3,
        classes=0,
    )


def cases() -> list[Case]:
    return [
        grid_case("halving/exhaust", 27, 1000, "exhaust", 3, False),
        grid_case("halving/smallest", 10, 200, "smallest", 2, False),
        grid_case("halving/fixed/aggressive", 30, 100, 20, 3, True),
        grid_case("halving/fixed/short", 30, 100, 20, 3, False),
        grid_case("halving/classification", 12, 300, "smallest", 3, False, cv=3, classes=3),
        grid_case("halving/power", 81, 2430, "exhaust", 3, False, cv=3),
        random_case("halvingRandom/exhaust", 300, 3, 3, "smallest"),
        random_case("halvingRandom/fixed", 500, 2, 5, 40),
        resource_case("halving/resource", 9, 81, 3),
    ]
