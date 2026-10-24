"""Chain inference against hmmlearn: log-likelihoods (score), posteriors (predict_proba), Viterbi paths and their log
probabilities (decode), posterior decoding (decode with algorithm="map") and filtered marginals (the last posterior of
each prefix), for categorical and full-covariance Gaussian HMMs with random parameters."""

import hmmlearn
import numpy as np
import numpy.typing as npt
from hmmlearn.hmm import CategoricalHMM, GaussianHMM
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng, spd

REFERENCE: dict[str, str] = {"hmmlearn": hmmlearn.__version__}


def stochastic(g: np.random.Generator, rows: int, cols: int) -> FloatArray:
    """Random probability rows, rounded to 6 decimals and renormalised so they are exact inputs."""
    m = g.dirichlet(np.ones(cols), size=rows)
    m = np.round(m, 6)
    m[:, -1] = 1.0 - m[:, :-1].sum(axis=1)
    return m


def outputs(model: CategoricalHMM | GaussianHMM, x: npt.NDArray[np.generic]) -> dict[str, object]:
    logprob, path = model.decode(x, algorithm="viterbi")
    _, map_path = model.decode(x, algorithm="map")
    filtered = np.array([model.predict_proba(x[: t + 1])[-1] for t in range(len(x))])
    return {
        "score": float(model.score(x)),
        "posteriors": np.asarray(model.predict_proba(x), dtype=np.float64),
        "viterbi": [int(s) for s in path],
        "viterbiLogProb": float(logprob),
        "map": [int(s) for s in map_path],
        "filtered": filtered,
    }


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(31)
    for t, (k, m, n) in enumerate([(2, 3, 8), (3, 4, 15), (4, 2, 20)]):
        model = CategoricalHMM(n_components=k, n_features=m)
        model.startprob_ = stochastic(g, 1, k)[0]
        model.transmat_ = stochastic(g, k, k)
        model.emissionprob_ = stochastic(g, k, m)
        x = g.integers(0, m, size=(n, 1))
        o = outputs(model, x)
        out.append(
            case(
                f"categorical/{t}",
                "categorical",
                [model.startprob_, model.transmat_, model.emissionprob_, x.ravel()],
                np.asarray(o.pop("posteriors")),
                rtol=1e-10,
                atol=1e-12,
                **o,
            )
        )
    for t, (k, d, n) in enumerate([(2, 2, 10), (3, 1, 12)]):
        model = GaussianHMM(n_components=k, covariance_type="full")
        model.startprob_ = stochastic(g, 1, k)[0]
        model.transmat_ = stochastic(g, k, k)
        model.means_ = g.normal(size=(k, d)) * 2
        model.covars_ = np.array([spd(g, d) / d for _ in range(k)])
        x = g.normal(size=(n, d)) * 2
        o = outputs(model, x)
        out.append(
            case(
                f"gaussian/{t}",
                "gaussian",
                [model.startprob_, model.transmat_, model.means_, np.asarray(model.covars_), x],
                np.asarray(o.pop("posteriors")),
                rtol=1e-9,
                atol=1e-12,
                **o,
            )
        )
    return out
