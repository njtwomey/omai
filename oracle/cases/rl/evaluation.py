"""Aggregate metrics across runs (rliable's definitions, Agarwal et al. 2021) against the scipy and numpy functions
rliable itself calls: scipy.stats.trim_mean for the IQM, scipy.stats.mannwhitneyu for the probability of improvement,
numpy for the mean, median, optimality gap, performance profiles, percentile intervals and curve interpolation."""

import numpy as np
import numpy.typing as npt
import scipy
import scipy.stats
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}

type Array = npt.NDArray[np.float64]


def probability_of_improvement(x: Array, y: Array) -> float:
    """rliable's metrics.probability_of_improvement: the Mann-Whitney U of each task over n m, averaged over tasks."""
    per: list[float] = []
    for task in range(x.shape[1]):
        if np.array_equal(x[:, task], y[:, task]):
            per.append(0.5)
            continue
        u = float(scipy.stats.mannwhitneyu(x[:, task], y[:, task], alternative="greater").statistic)
        per.append(u / (x.shape[0] * y.shape[0]))
    return float(np.mean(per))


def run_profile(scores: Array, taus: Array) -> Array:
    return np.array([np.mean(scores > tau) for tau in taus])


def average_profile(scores: Array, taus: Array) -> Array:
    means = np.mean(scores, axis=0)
    return np.array([np.mean(means > tau) for tau in taus])


def cases() -> list[Case]:
    g = rng(2021)
    out: list[Case] = []
    for runs, tasks in ((5, 3), (10, 4), (7, 1), (13, 6)):
        s = g.normal(0.6, 0.4, size=(runs, tasks))
        name = f"{runs}x{tasks}"
        out.append(case(f"iqm/{name}", "iqm", [s], np.float64(scipy.stats.trim_mean(s, 0.25, axis=None))))
        out.append(case(f"mean/{name}", "mean", [s], np.float64(np.mean(np.mean(s, axis=0)))))
        out.append(case(f"median/{name}", "median", [s], np.float64(np.median(np.mean(s, axis=0)))))
        for gamma in (1.0, 0.5):
            gap = gamma - np.mean(np.minimum(s, gamma))
            out.append(case(f"gap/{name}/{gamma}", "optimalityGap", [s], np.float64(gap), gamma=gamma))
        taus = np.linspace(-0.5, 1.5, 9)
        out.append(case(f"profile/run/{name}", "runProfile", [s, taus], run_profile(s, taus)))
        out.append(case(f"profile/average/{name}", "averageProfile", [s, taus], average_profile(s, taus)))
        other = g.normal(0.4, 0.5, size=(runs + 2, tasks))
        out.append(case(f"poi/{name}", "poi", [s, other], np.float64(probability_of_improvement(s, other))))
        # Ties: scores on a coarse grid.
        tx = np.round(g.uniform(0, 3, size=(runs, tasks)))
        ty = np.round(g.uniform(0, 3, size=(runs, tasks)))
        out.append(case(f"poi/ties/{name}", "poi", [tx, ty], np.float64(probability_of_improvement(tx, ty))))
        low = g.uniform(-1, 0, size=tasks)
        high = low + g.uniform(1, 2, size=tasks)
        out.append(case(f"normalize/{name}", "normalize", [s, low, high], (s - low) / (high - low)))
    out.append(case("poi/identical", "poi", [np.ones((3, 2)), np.ones((3, 2))], np.float64(0.5)))
    for n, level in ((11, 0.95), (200, 0.9), (57, 0.5)):
        samples = g.normal(size=n)
        tail = (1 - level) / 2
        expected = np.percentile(samples, [100 * tail, 100 * (1 - tail)])
        out.append(case(f"percentile/{n}/{level}", "percentileInterval", [samples], expected, level=level))
    steps = np.sort(g.uniform(0, 100, size=12))
    values = g.normal(size=12)
    grid = np.linspace(steps[0], steps[-1], 20)
    out.append(case("align/interp", "align", [steps, values, grid], np.interp(grid, steps, values)))
    return out
