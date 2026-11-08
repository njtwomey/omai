"""getWindow against scipy.signal.get_window: names, aliases, parameters, Kaiser betas and suffixes."""

import numpy as np
import scipy
import scipy.signal as S
from omai_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type Spec = str | float | tuple[str, float] | tuple[str, float, float]

SPECS: list[Spec] = [
    "hann",
    "hamm",
    "blackmanharris",
    "nuttall",
    "flattop",
    "boxcar",
    "bartlett",
    "triang",
    "cosine",
    "parzen",
    "bohman",
    "barthann",
    "lanczos",
    "hann_symmetric",
    "hamming_periodic",
    ("kaiser", 8.0),
    ("gaussian", 3.0),
    ("general_gaussian", 1.5, 4.0),
    ("tukey", 0.25),
    ("dpss", 2.5),
    ("exponential", 3.0, 2.0),
    ("general_cosine", 0.5, 0.5),
    ("general_hamming", 0.6),
    ("kbd", 3.0),
    5.0,
]


def cases() -> list[Case]:
    out: list[Case] = []
    for spec in SPECS:
        only = {"exponential": [True], "kbd": [False]}
        for fftbins in only.get(spec[0], [True, False]) if isinstance(spec, tuple) else [True, False]:
            # scipy takes general_cosine coefficients as one sequence parameter.
            arg = ("general_cosine", [0.5, 0.5]) if isinstance(spec, tuple) and spec[0] == "general_cosine" else spec
            w = np.asarray(S.get_window(arg, 12, fftbins=fftbins))
            label = "/".join(str(s) for s in spec) if isinstance(spec, tuple) else str(spec)
            js = list(spec) if isinstance(spec, tuple) else spec
            tol = {"rtol": 1e-9, "atol": 1e-12}
            out.append(case(f"getWindow/{label}/{fftbins}", "getWindow", [], w, **tol, window=js, fftbins=fftbins))
    return out
