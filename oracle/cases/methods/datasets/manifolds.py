"""Manifold embeddings against sklearn.datasets: our embedding of sklearn's own intrinsic coordinates."""

import numpy as np
import sklearn
from sklearn.datasets import make_s_curve, make_swiss_roll
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    # Without noise the second coordinate is the height, so (t, height) determine the points.
    for hole in [False, True]:
        x, t = make_swiss_roll(n_samples=20, noise=0.0, random_state=0, hole=hole)
        name = "swissRoll/hole" if hole else "swissRoll/plain"
        out.append(case(name, "swissRollEmbedding", [np.asarray(t), np.asarray(x[:, 1])], np.asarray(x)))
    xs, ts = make_s_curve(n_samples=20, noise=0.0, random_state=0)
    out.append(case("sCurve", "sCurveEmbedding", [np.asarray(ts), np.asarray(xs[:, 1])], np.asarray(xs)))
    return out
