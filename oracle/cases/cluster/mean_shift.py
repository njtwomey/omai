"""Mean shift against scikit-learn: the cluster centres (most populated first) and the labels, for a flat kernel."""

from importlib.metadata import version

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from sklearn.cluster import MeanShift, estimate_bandwidth

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}


def cases() -> list[Case]:
    g = rng(8)
    x = np.concatenate([g.normal(c, 0.6, (20, 2)) for c in [(0, 0), (5, 1), (1, 6)]])
    out: list[Case] = []
    for bandwidth in [1.5, 2.5, None]:
        bw = float(estimate_bandwidth(x, quantile=0.3)) if bandwidth is None else bandwidth
        ms = MeanShift(bandwidth=bw).fit(x)
        name = "estimated" if bandwidth is None else str(bandwidth)
        out.append(
            case(
                f"{name}/centres",
                "centres",
                [x],
                np.asarray(ms.cluster_centers_, dtype=np.float64),
                rtol=1e-6,
                atol=1e-6,
                bandwidth=bw,
            )
        )
        out.append(case(f"{name}/labels", "labels", [x], np.asarray(ms.labels_, dtype=np.float64), bandwidth=bw))
    out.append(
        case(
            "estimate_bandwidth",
            "bandwidth",
            [x],
            np.array(float(estimate_bandwidth(x, quantile=0.3))),
            rtol=1e-10,
            atol=1e-10,
        )
    )
    return out
