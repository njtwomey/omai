"""Decibels against librosa (amplitude_to_db, power_to_db) and phase unwrapping against numpy.unwrap."""

from importlib.metadata import version

import librosa
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "librosa": version("librosa")}


def cases() -> list[Case]:
    g = rng(400)
    out: list[Case] = []
    a = np.abs(g.normal(size=(3, 5))) * 10.0
    a[0, 0] = 0.0
    out.append(case("amplitude", "magnitudeDb", [a], librosa.amplitude_to_db(a, top_db=None)))
    ref = librosa.amplitude_to_db(a, ref=2.0, amin=1e-3, top_db=None)
    out.append(case("amplitude/ref", "magnitudeDb", [a], ref, ref=2.0, amin=1e-3))
    top = librosa.amplitude_to_db(a, ref=np.max, top_db=30.0)
    out.append(case("amplitude/max/top", "magnitudeDb", [a], top, ref="max", topDb=30.0))
    p = a**2
    out.append(case("power", "magnitudeDb", [p], librosa.power_to_db(p, top_db=None), power=True))
    out.append(case("power/top", "magnitudeDb", [p], librosa.power_to_db(p, top_db=80.0), power=True, topDb=80.0))
    ph = np.cumsum(g.uniform(-2.5, 2.5, size=40)) % (2 * np.pi)
    out.append(case("unwrap", "unwrap", [ph], np.unwrap(ph)))
    out.append(case("unwrap/discont", "unwrap", [ph], np.unwrap(ph, discont=4.0), discont=4.0))
    m = (np.cumsum(g.uniform(-3, 3, size=(4, 6)), axis=0)) % 6.0
    out.append(case("unwrap/period/axis0", "unwrap", [m], np.unwrap(m, axis=0, period=6.0), axis=0, period=6.0))
    return out
