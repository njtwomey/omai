"""Subset selection against statsmodels: every subset (and every greedy move) scored by the AIC, BIC or adjusted
R-squared of an OLS fit, the searches written out in Python on top of statsmodels' OLS. Compared on the chosen features,
the criterion of the chosen subset and the criteria along the path."""

import itertools

import numpy as np
import numpy.typing as npt
import statsmodels
import statsmodels.api as sm
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}

type Mat = npt.NDArray[np.float64]


def score(x: Mat, y: Mat, subset: tuple[int, ...], criterion: str) -> tuple[float, float]:
    """The natural criterion and the value to minimise."""
    exog = np.column_stack([x[:, list(subset)], np.ones(len(y))])
    res = sm.OLS(y, exog).fit()
    value = {"aic": res.aic, "bic": res.bic, "adjusted-r2": res.rsquared_adj}[criterion]
    return float(value), float(-value if criterion == "adjusted-r2" else value)


def search(x: Mat, y: Mat, method: str, criterion: str) -> tuple[tuple[int, ...], list[float]]:
    d = x.shape[1]
    if method == "best":
        path: list[float] = [score(x, y, (), criterion)[0]]
        best: tuple[int, ...] = ()
        best_score = score(x, y, (), criterion)[1]
        for size in range(1, d + 1):
            winner = min(itertools.combinations(range(d), size), key=lambda s: score(x, y, s, criterion)[1])
            path.append(score(x, y, winner, criterion)[0])
            if score(x, y, winner, criterion)[1] < best_score:
                best, best_score = winner, score(x, y, winner, criterion)[1]
        return best, path
    current: tuple[int, ...] = tuple(range(d)) if method == "backward" else ()
    current_score = score(x, y, current, criterion)[1]
    path = [score(x, y, current, criterion)[0]]
    while True:
        moves: list[tuple[int, ...]] = []
        if method != "backward":
            moves += [tuple(sorted((*current, j))) for j in range(d) if j not in current]
        if method != "forward":
            moves += [tuple(v for v in current if v != j) for j in current]
        if not moves:
            break
        move = min(moves, key=lambda s: score(x, y, s, criterion)[1])
        if not score(x, y, move, criterion)[1] < current_score - 1e-12:
            break
        current, current_score = move, score(x, y, move, criterion)[1]
        path.append(score(x, y, current, criterion)[0])
    return current, path


def cases() -> list[Case]:
    g = rng(1974)
    out: list[Case] = []
    n, d = 70, 6
    x = g.normal(size=(n, d))
    x[:, 1] += 0.7 * x[:, 0]
    y = 1.0 + 2.0 * x[:, 0] - 1.5 * x[:, 2] + 0.8 * x[:, 4] + 0.3 * x[:, 5] + g.normal(size=n)
    for method in ("best", "forward", "backward", "stepwise"):
        for criterion in ("aic", "bic", "adjusted-r2"):
            selected, path = search(x, y, method, criterion)
            out.append(
                case(
                    f"{method}/{criterion}",
                    "subset",
                    [x, y],
                    {
                        "selected": np.array(selected, dtype=np.float64),
                        "criterionValue": np.array(score(x, y, selected, criterion)[0]),
                        "path": np.array(path),
                    },
                    rtol=1e-8,
                    atol=1e-8,
                    method=method,
                    criterion=criterion,
                )
            )
    return out
