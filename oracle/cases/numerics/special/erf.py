"""Error functions against scipy.special."""

import numpy as np
import scipy
import scipy.special as sp
from omai_oracle.fixture import Case
from omai_oracle.helpers import reference, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}


def cases() -> list[Case]:
    g = rng(1)
    x = np.concatenate([g.uniform(-4, 4, 30), [0.0, 0.5, -1.5, 6.0, -6.0]])
    y = np.concatenate([g.uniform(-0.99, 0.99, 30), [0.0, -1.0, 1.0]])
    return [
        reference("erf", "erf", sp.erf, [x]),
        reference("erfc", "erfc", sp.erfc, [x]),
        reference("erfcx", "erfcx", sp.erfcx, [np.concatenate([x, g.uniform(5, 100, 10)])]),
        reference("erfinv", "erfinv", sp.erfinv, [y]),
        reference("erfcinv", "erfcinv", sp.erfcinv, [1 - y]),
        reference("dawsn", "dawsn", sp.dawsn, [np.concatenate([x, g.uniform(-30, 30, 10)])]),
    ]
