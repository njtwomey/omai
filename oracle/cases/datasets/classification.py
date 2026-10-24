"""Deterministic parts of the classification generators against sklearn.datasets."""

import numpy as np
import sklearn
from sklearn.datasets import make_circles, make_gaussian_quantiles, make_hastie_10_2, make_moons
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    for n in [7, 10]:
        x, y = make_moons(n_samples=n, shuffle=False, noise=None)
        out.append(case(f"moons/{n}/x", "moonPoints", [], np.asarray(x), n=n, part="x"))
        out.append(case(f"moons/{n}/y", "moonPoints", [], np.asarray(y), n=n, part="y"))
        xc, yc = make_circles(n_samples=n, shuffle=False, noise=None, factor=0.6)
        out.append(case(f"circles/{n}/x", "circlePoints", [], np.asarray(xc), n=n, factor=0.6, part="x"))
        out.append(case(f"circles/{n}/y", "circlePoints", [], np.asarray(yc), n=n, factor=0.6, part="y"))
    for n, d, k in [(30, 2, 3), (25, 3, 4)]:
        xq, yq = make_gaussian_quantiles(n_samples=n, n_features=d, n_classes=k, random_state=0, shuffle=False)
        out.append(
            case(f"gaussianQuantiles/{n}x{d}/{k}", "gaussianQuantileLabels", [np.asarray(xq)], np.asarray(yq), k=k)
        )
    xh, yh = make_hastie_10_2(n_samples=40, random_state=1)
    out.append(case("hastie10_2/labels", "hastieLabels", [np.asarray(xh)], np.asarray((yh > 0).astype(np.int64))))
    return out
