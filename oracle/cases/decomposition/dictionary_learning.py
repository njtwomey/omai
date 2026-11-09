"""Sparse coding and dictionary learning against scikit-learn: `SparseCoder` for each coder (and the positive and
split-sign options), `dict_learning` (batch, from the SVD start, lasso by LARS and by coordinate descent, over a fixed
number of iterations) and `dict_learning_online` (from a given dictionary without shuffling, deterministic, plus the
reconstruction error of a shuffled run from sklearn's own start, which only a loose band can match)."""

import warnings
from importlib.metadata import version

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.decomposition import (
    MiniBatchDictionaryLearning,
    SparseCoder,
    dict_learning,
)

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

EXACT = {"rtol": 1e-8, "atol": 1e-9}
# Coordinate descent stops at a duality gap of 1e-8 |x|^2 in scikit-learn, so its codes carry that much error.
ITERATIVE = {"rtol": 1e-6, "atol": 1e-7}
COORDINATE = {"rtol": 1e-4, "atol": 1e-5}
# Over a dozen alternations the two coordinate descents' stopping errors (each about 1e-6) compound.
CHAINED = {"rtol": 1e-3, "atol": 1e-4}


def unit_rows(a: np.ndarray) -> np.ndarray:  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    return a / np.linalg.norm(a, axis=1, keepdims=True)  # pyright: ignore[reportUnknownVariableType]


def coder_cases() -> list[Case]:
    g = rng(11)
    dictionary = unit_rows(g.normal(size=(10, 8)))
    x = g.normal(size=(6, 8))
    out: list[Case] = []
    options: list[tuple[str, str, dict[str, object], dict[str, object], dict[str, float]]] = [
        ("omp", "omp", {"transform_n_nonzero_coefs": 2}, {"nNonzeroCoefs": 2}, EXACT),
        ("lars", "lars", {"transform_n_nonzero_coefs": 3}, {"nNonzeroCoefs": 3}, EXACT),
        ("lasso_lars", "lassoLars", {"transform_alpha": 0.4}, {"alpha": 0.4}, EXACT),
        ("lasso_cd", "lassoCd", {"transform_alpha": 0.4}, {"alpha": 0.4}, COORDINATE),
        ("threshold", "threshold", {"transform_alpha": 0.5}, {"alpha": 0.5}, EXACT),
    ]
    for algorithm, ours, sk, ts, tol in options:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = SparseCoder(dictionary, transform_algorithm=algorithm, **sk).transform(x)  # pyright: ignore[reportArgumentType]
        out.append(case(f"coder/{algorithm}", "sparseCode", [x, dictionary], code, **tol, algorithm=ours, **ts))  # pyright: ignore[reportArgumentType]
    for algorithm, ours in [("lasso_cd", "lassoCd"), ("threshold", "threshold")]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = SparseCoder(dictionary, transform_algorithm=algorithm, transform_alpha=0.3, positive_code=True).transform(x)
        tol = COORDINATE if algorithm == "lasso_cd" else EXACT
        out.append(
            case(f"coder/{algorithm}-positive", "sparseCode", [x, dictionary], code, **tol, algorithm=ours, alpha=0.3, positive=True)
        )
    code = SparseCoder(dictionary, transform_algorithm="omp", transform_n_nonzero_coefs=2, split_sign=True).transform(x)
    out.append(case("coder/split-sign", "splitCoder", [x, dictionary], code, **EXACT, nNonzeroCoefs=2))
    # More atoms than a plain default: sklearn's default number of atoms, max(n_features / 10, 1), truncated.
    wide = unit_rows(g.normal(size=(30, 25)))
    xw = g.normal(size=(4, 25))
    code = SparseCoder(wide, transform_algorithm="omp").transform(xw)
    out.append(case("coder/omp-default-atoms", "sparseCode", [xw, wide], code, **EXACT, algorithm="omp"))
    return out


def batch_cases() -> list[Case]:
    g = rng(21)
    z = np.zeros((40, 5))
    for i in range(40):
        z[i, g.choice(5, 2, replace=False)] = g.normal(size=2)
    atoms = unit_rows(g.normal(size=(5, 9)))
    x = z @ atoms + 0.05 * g.normal(size=(40, 9))
    out: list[Case] = []
    for method, ours, tol in [("lars", "lassoLars", EXACT), ("cd", "lassoCd", CHAINED)]:
        for iters in [1, 4, 12]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                code, dictionary, errors = dict_learning(
                    x, 5, alpha=0.6, max_iter=iters, tol=0.0, method=method, random_state=0  # pyright: ignore[reportArgumentType]
                )
            params = {"alpha": 0.6, "maxIter": iters, "fitAlgorithm": ours}
            name = f"{method}/{iters}"
            out.append(case(f"{name}/dictionary", "batchDictionary", [x], dictionary, **tol, **params))
            out.append(case(f"{name}/code", "batchCode", [x], code, **tol, **params))
            out.append(case(f"{name}/objective", "batchObjective", [x], np.asarray(errors), **tol, **params))
    return out


def online_cases() -> list[Case]:
    g = rng(31)
    z = np.zeros((60, 6))
    for i in range(60):
        z[i, g.choice(6, 2, replace=False)] = g.normal(size=2)
    atoms = unit_rows(g.normal(size=(6, 10)))
    x = z @ atoms + 0.05 * g.normal(size=(60, 10))
    start = unit_rows(x[:6] + 0.1 * g.normal(size=(6, 10)))
    out: list[Case] = []
    for epochs in [1, 3]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = MiniBatchDictionaryLearning(
                n_components=6,
                alpha=0.5,
                max_iter=epochs,
                fit_algorithm="lars",
                batch_size=7,
                shuffle=False,
                dict_init=start,
                transform_algorithm="lasso_lars",
                transform_alpha=0.5,
                tol=0.0,
                max_no_improvement=None,
                random_state=0,
            ).fit(x)
            code = est.transform(x)
        params = {"alpha": 0.5, "maxIter": epochs, "batchSize": 7, "fitAlgorithm": "lassoLars"}
        out.append(case(f"online/{epochs}/dictionary", "onlineDictionary", [x, start], est.components_, **ITERATIVE, **params))
        out.append(case(f"online/{epochs}/code", "onlineCode", [x, start], code, **ITERATIVE, **params))
        out.append(case(f"online/{epochs}/steps", "onlineSteps", [x, start], np.float64(est.n_steps_), **EXACT, **params))
    # A shuffled run from sklearn's own randomised-SVD start: only the quality of the fit is comparable.
    big = rng(41)
    zb = np.zeros((400, 8))
    for i in range(400):
        zb[i, big.choice(8, 2, replace=False)] = big.normal(size=2)
    atoms_b = unit_rows(big.normal(size=(8, 12)))
    xb = zb @ atoms_b + 0.05 * big.normal(size=(400, 12))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = MiniBatchDictionaryLearning(
            n_components=8, alpha=0.3, max_iter=15, fit_algorithm="lars", batch_size=20, shuffle=True,
            transform_algorithm="lasso_lars", transform_alpha=0.3, tol=0.0, max_no_improvement=None, random_state=0,
        ).fit(xb)  # fmt: skip
        code = est.transform(xb)
    error = float(np.sum((xb - code @ est.components_) ** 2))
    objective = 0.5 * error + 0.3 * float(np.abs(code).sum())
    params = {"alpha": 0.3, "maxIter": 15, "batchSize": 20, "components": 8}
    out.append(case("online/shuffled/error", "onlineBand", [xb], np.float64(error), rtol=0.5, atol=0.5, **params))
    out.append(case("online/shuffled/objective", "onlineObjective", [xb], np.float64(objective), rtol=0.5, atol=0.5, **params))
    return out


def cases() -> list[Case]:
    return [*coder_cases(), *batch_cases(), *online_cases()]
