"""Running moments against numpy and scipy.stats (Welford and the parallel merge must agree with the batch formulas),
and exponentially weighted moments against pandas' ewm."""

import numpy as np
import pandas as pd
import scipy
import scipy.stats as st
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}


def cases() -> list[Case]:
    g = rng(31)
    x = g.lognormal(1.0, 0.6, size=40) + 1e3
    s = pd.Series(g.normal(size=30).cumsum())
    moments = np.asarray([x.mean(), x.var(ddof=1), st.skew(x), st.kurtosis(x), x.min(), x.max()])
    out: list[Case] = [
        case("moments/push", "moments", [x], moments),
        case("moments/merge", "momentsMerged", [x], moments, split=13),
    ]
    smoothing: list[tuple[str, dict[str, float]]] = [
        ("alpha", {"alpha": 0.3}),
        ("span", {"span": 5.0}),
        ("com", {"com": 2.5}),
        ("halflife", {"halflife": 3.0}),
    ]
    for name, kw in smoothing:
        for adjust in [True, False]:
            e = s.ewm(adjust=adjust, **kw)  # pyright: ignore[reportArgumentType]
            tag = f"{name}/{'adjust' if adjust else 'recursive'}"
            out += [
                case(f"ewmMean/{tag}", "ewmMean", [s.to_numpy()], e.mean().to_numpy(), adjust=adjust, **kw),
                case(f"ewmVar/{tag}", "ewmVar", [s.to_numpy()], e.var().to_numpy(), adjust=adjust, **kw),
                case(
                    f"ewmVar/biased/{tag}",
                    "ewmVar",
                    [s.to_numpy()],
                    e.var(bias=True).to_numpy(),
                    adjust=adjust,
                    bias=True,
                    **kw,
                ),
                case(f"ewmStd/{tag}", "ewmStd", [s.to_numpy()], e.std().to_numpy(), adjust=adjust, **kw),
            ]
    return out
