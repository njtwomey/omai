"""Estimation against scipy.signal.wiener and librosa.lpc; detector thresholds and ROC curves against scipy.stats;
adaptive filters against padasip; cepstra by numpy (MATLAB's cceps algorithm)."""

from importlib.metadata import version

import librosa
import numpy as np
import padasip as pa
import scipy
import scipy.signal as S
import scipy.stats as st
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from omai_oracle.signal_helpers import smooth_image, tones
from padasip.filters import FilterLMS, FilterNLMS, FilterRLS

REFERENCE: dict[str, str] = {
    "scipy": scipy.__version__,
    "librosa": version("librosa"),
    "padasip": version("padasip"),
    "numpy": np.__version__,
}

TOL = {"rtol": 1e-10, "atol": 1e-12}


def complex_cepstrum(x: FloatArray, n: int) -> tuple[FloatArray, int]:
    """MATLAB cceps / python-acoustics complex_cepstrum."""
    spectrum = np.fft.fft(x, n=n)
    unwrapped = np.unwrap(np.angle(spectrum))
    center = (n + 1) // 2
    ndelay = int(np.round(unwrapped[center] / np.pi))
    unwrapped = unwrapped - np.pi * ndelay * np.arange(n) / center
    log_spectrum = np.log(np.abs(spectrum)) + 1j * unwrapped
    return np.asarray(np.real(np.fft.ifft(log_spectrum)), dtype=np.float64), ndelay


def cases() -> list[Case]:
    g = rng(418)
    out: list[Case] = []
    x = tones(g, 40) + 0.3 * g.normal(size=40)
    out.append(case("wiener/1d", "wiener", [x], S.wiener(x), **TOL))
    out.append(case("wiener/1d/5", "wiener", [x], S.wiener(x, 5, 0.1), **TOL, size=5, noise=0.1))
    img = smooth_image(g, 12, 14) + 0.05 * g.normal(size=(12, 14))
    out.append(case("wiener/2d", "wiener", [img], S.wiener(img, [3, 5]), **TOL, size=[3, 5]))
    for order in [2, 6]:
        out.append(case(f"lpc/{order}", "lpc", [x], librosa.lpc(x, order=order), rtol=1e-9, atol=1e-11, order=order))
    pfa = np.array([1e-4, 1e-2, 0.1, 0.5])
    out.append(
        case("rocKnownSignal", "rocKnownSignal", [pfa], st.norm.sf(st.norm.isf(pfa) - np.sqrt(4.0)), **TOL, d2=4.0)
    )
    thr = st.chi2.isf(pfa, 8)
    out.append(
        case(
            "rocEnergyDetector",
            "rocEnergyDetector",
            [pfa],
            st.ncx2.sf(thr, 8, 8 * 0.7),
            rtol=1e-8,
            atol=1e-12,
            n=8,
            snr=0.7,
        )
    )
    out.append(
        case(
            "energyThreshold",
            "energyThreshold",
            [],
            np.array(0.5 * st.chi2.isf(1e-3, 10)),
            **TOL,
            n=10,
            noiseVariance=0.5,
            pfa=1e-3,
        )
    )
    n_taps = 4
    sig = g.normal(size=60)
    xm = pa.input_from_history(sig, n_taps)
    d = xm @ np.array([0.5, -0.3, 0.2, 0.1]) + 0.01 * g.normal(size=len(xm))
    for name, f in [
        ("lms", FilterLMS(n=n_taps, mu=0.05, w="zeros")),
        ("nlms", FilterNLMS(n=n_taps, mu=0.5, eps=0.01, w="zeros")),
        ("rls", FilterRLS(n=n_taps, mu=0.98, eps=0.5, w="zeros")),
    ]:
        y, e, w = f.run(d, xm)
        res = np.concatenate([y, e, f.w])
        mu = {"lms": 0.05, "nlms": 0.5, "rls": 0.98}[name]
        eps = {"lms": 0.0, "nlms": 0.01, "rls": 0.5}[name]
        out.append(
            case(f"adaptive/{name}", "adaptive", [xm, d], res, rtol=1e-9, atol=1e-11, method=name, mu=mu, eps=eps)
        )
        del w
    for n in [32, 33]:
        c, delay = complex_cepstrum(x[:20], n)
        out.append(
            case(f"complexCepstrum/{n}", "complexCepstrum", [x[:20]], np.concatenate([c, [float(delay)]]), **TOL, n=n)
        )
    out.append(case("realCepstrum", "realCepstrum", [x], np.real(np.fft.ifft(np.log(np.abs(np.fft.fft(x))))), **TOL))
    return out
