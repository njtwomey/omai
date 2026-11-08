"""Mu-law companding against librosa (mu_compress, mu_expand); sinc interpolation as numpy.sinc sums."""

from importlib.metadata import version

import librosa
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "librosa": version("librosa")}


def cases() -> list[Case]:
    g = rng(402)
    out: list[Case] = []
    x = np.concatenate([[-1.0, 0.0, 1.0], g.uniform(-1, 1, size=20)])
    for mu in [255.0, 15.0, 7.0]:
        comp = librosa.mu_compress(x, mu=mu, quantize=False)
        out.append(case(f"muLawEncode/{mu}/float", "muLawEncode", [x], comp, mu=mu, quantize=False))
        codes = np.asarray(librosa.mu_compress(x, mu=mu, quantize=True), dtype=np.float64)
        out.append(case(f"muLawEncode/{mu}/codes", "muLawEncode", [x], codes, mu=mu, quantize=True))
        out.append(
            case(
                f"muLawDecode/{mu}/float",
                "muLawDecode",
                [comp],
                librosa.mu_expand(comp, mu=mu, quantize=False),
                mu=mu,
                quantize=False,
            )
        )
        out.append(
            case(
                f"muLawDecode/{mu}/codes",
                "muLawDecode",
                [codes],
                librosa.mu_expand(codes, mu=mu, quantize=True),
                mu=mu,
                quantize=True,
            )
        )
    s = g.normal(size=16)
    t = np.linspace(2.0, 3.5, 13)
    fs = 4.0
    ref = np.array([np.sum(s * np.sinc(fs * ti - np.arange(16))) for ti in t])
    out.append(case("sincInterpolate", "sincInterpolate", [s, t], ref, fs=fs))
    return out
