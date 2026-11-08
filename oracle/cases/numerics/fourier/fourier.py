"""Fourier transforms against numpy.fft, scipy.fft and scipy.signal (chirp-z, zoom FFT); NUDFTs and NUFFTs by direct
sums (the NUFFT within its requested tolerance relative to the largest value, as FINUFFT's eps)."""

from typing import Literal

import numpy as np
import scipy
import scipy.fft as sf
import scipy.signal as sg
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}

NORMS: list[Literal["backward", "ortho", "forward"]] = ["backward", "ortho", "forward"]
TYPES: list[Literal[1, 2, 3, 4]] = [1, 2, 3, 4]


def pair(z: FloatArray | np.ndarray) -> FloatArray:
    """Complex results as stacked real and imaginary parts."""
    c = np.asarray(z, dtype=np.complex128)
    return np.stack([c.real, c.imag])


def cases() -> list[Case]:
    g = rng(90)
    out: list[Case] = []
    for n in [1, 2, 8, 64, 7, 13, 30, 97]:
        re, im = g.normal(size=n), g.normal(size=n)
        z = re + 1j * im
        for norm in NORMS:
            out.append(
                case(
                    f"fft/{n}/{norm}",
                    "fft",
                    [re, im],
                    pair(np.fft.fft(z, norm=norm)),
                    rtol=1e-10,
                    atol=1e-11,
                    norm=norm,
                )
            )
            out.append(
                case(
                    f"ifft/{n}/{norm}",
                    "ifft",
                    [re, im],
                    pair(np.fft.ifft(z, norm=norm)),
                    rtol=1e-10,
                    atol=1e-11,
                    norm=norm,
                )
            )
    x = g.normal(size=10)
    out.append(
        case(
            "fft/padded",
            "fft",
            [x, np.zeros(10)],
            pair(np.fft.fft(x, n=16)),
            rtol=1e-10,
            atol=1e-11,
            n=16,
            norm="backward",
        )
    )
    out.append(
        case(
            "fft/cropped",
            "fft",
            [x, np.zeros(10)],
            pair(np.fft.fft(x, n=6)),
            rtol=1e-10,
            atol=1e-11,
            n=6,
            norm="backward",
        )
    )
    m = g.normal(size=(3, 5))
    out.append(case("fft/axis0", "fftAxis0", [m], pair(np.fft.fft(m, axis=0)), rtol=1e-10, atol=1e-11))
    for n in [8, 9]:
        s = g.normal(size=n)
        out.append(case(f"rfft/{n}", "rfft", [s], pair(np.fft.rfft(s)), rtol=1e-10, atol=1e-11))
        half = np.fft.rfft(s)
        out.append(
            case(f"irfft/{n}", "irfft", [half.real, half.imag], np.fft.irfft(half, n=n), rtol=1e-10, atol=1e-11, n=n)
        )
        out.append(case(f"hfft/{n}", "hfft", [s[:5]], np.fft.hfft(s[:5], n=n), rtol=1e-10, atol=1e-11, n=n))
    grid = g.normal(size=(4, 6))
    out.append(case("fftn", "fftn", [grid], pair(np.fft.fftn(grid)), rtol=1e-10, atol=1e-11))
    out.append(case("fftn/shape", "fftnShape", [grid], pair(np.fft.fftn(grid, s=(5, 8))), rtol=1e-10, atol=1e-11))
    out.append(case("fftfreq", "fftfreq", [], np.fft.fftfreq(9, 0.1), rtol=1e-14))
    out.append(case("rfftfreq", "rfftfreq", [], np.fft.rfftfreq(10, 0.1), rtol=1e-14))
    out.append(case("fftshift", "fftshift", [grid], np.fft.fftshift(grid), rtol=0))
    out.append(case("ifftshift", "ifftshift", [grid], np.fft.ifftshift(grid), rtol=0))
    lens = np.array([sf.next_fast_len(t) for t in [1, 7, 97, 1000, 1025, 4097]], dtype=np.float64)
    out.append(case("nextFastLen", "nextFastLen", [], lens, rtol=0))
    lens_real = np.array([sf.next_fast_len(t, real=True) for t in [1, 7, 97, 1000, 1025, 4097]], dtype=np.float64)
    out.append(case("nextFastLen/real", "nextFastLenReal", [], lens_real, rtol=0))

    for n in [2, 5, 8]:
        v = g.normal(size=n)
        for t in TYPES:
            if t == 1 and n < 2:
                continue
            for norm in NORMS:
                params = {"type": t, "norm": norm}
                out.append(
                    case(
                        f"dct/{t}/{n}/{norm}",
                        "dct",
                        [v],
                        sf.dct(v, type=t, norm=norm),
                        rtol=1e-10,
                        atol=1e-11,
                        **params,
                    )
                )
                out.append(
                    case(
                        f"idct/{t}/{n}/{norm}",
                        "idct",
                        [v],
                        sf.idct(v, type=t, norm=norm),
                        rtol=1e-10,
                        atol=1e-11,
                        **params,
                    )
                )
                out.append(
                    case(
                        f"dst/{t}/{n}/{norm}",
                        "dst",
                        [v],
                        sf.dst(v, type=t, norm=norm),
                        rtol=1e-10,
                        atol=1e-11,
                        **params,
                    )
                )
                out.append(
                    case(
                        f"idst/{t}/{n}/{norm}",
                        "idst",
                        [v],
                        sf.idst(v, type=t, norm=norm),
                        rtol=1e-10,
                        atol=1e-11,
                        **params,
                    )
                )
    out.append(case("dct/axis0", "dctAxis0", [grid], sf.dct(grid, axis=0), rtol=1e-10, atol=1e-11))

    sig = g.normal(size=12)
    out.append(
        case(
            "czt/dft",
            "czt",
            [sig],
            pair(sg.czt(sig)),
            rtol=1e-9,
            atol=1e-10,
            m=12,
            w_re=float(np.cos(-2 * np.pi / 12)),
            w_im=float(np.sin(-2 * np.pi / 12)),
            a_re=1.0,
            a_im=0.0,
        )
    )
    w = 0.98 * np.exp(-1j * 0.3)
    a = 1.05 * np.exp(1j * 0.2)
    out.append(
        case(
            "czt/spiral",
            "czt",
            [sig],
            pair(sg.czt(sig, m=9, w=w, a=a)),
            rtol=1e-9,
            atol=1e-10,
            m=9,
            w_re=float(w.real),
            w_im=float(w.imag),
            a_re=float(a.real),
            a_im=float(a.imag),
        )
    )
    out.append(case("zoomFft", "zoomFft", [sig], pair(sg.zoom_fft(sig, [0.2, 0.6], m=16, fs=2)), rtol=1e-9, atol=1e-10))

    pts = g.uniform(-np.pi, np.pi, size=20)
    strengths = g.normal(size=20) + 1j * g.normal(size=20)
    k = np.arange(16) - 8
    type1 = np.exp(1j * np.outer(k, pts)) @ strengths
    out.append(
        case("nudft1", "nudft1", [pts, strengths.real, strengths.imag], pair(type1), rtol=1e-11, atol=1e-11, m=16)
    )
    coeffs = g.normal(size=16) + 1j * g.normal(size=16)
    type2 = np.exp(-1j * np.outer(pts, k)) @ coeffs
    out.append(case("nudft2", "nudft2", [pts, coeffs.real, coeffs.imag], pair(type2), rtol=1e-11, atol=1e-11))
    out.extend(nufft_cases())
    return out


def nufft_cases() -> list[Case]:
    """NUFFTs of types 1 and 2 for few and many modes (few modes force an oversampling above 2), both signs."""
    g = rng(91)
    out: list[Case] = []
    for m in [3, 8, 64]:
        pts = g.uniform(-3 * np.pi, 3 * np.pi, size=25)
        strengths = g.normal(size=25) + 1j * g.normal(size=25)
        coeffs = g.normal(size=m) + 1j * g.normal(size=m)
        k = np.arange(m) - m // 2
        for sign in [1, -1]:
            for tol in [1e-6, 1e-12]:
                type1 = np.exp(sign * 1j * np.outer(k, pts)) @ strengths
                type2 = np.exp(-sign * 1j * np.outer(pts, k)) @ coeffs
                for fn, inputs, ref in [
                    ("nufft1", [pts, strengths.real, strengths.imag], type1),
                    ("nufft2", [pts, coeffs.real, coeffs.imag], type2),
                ]:
                    sgn = sign if fn == "nufft1" else -sign
                    out.append(
                        case(
                            f"{fn}/{m}/{sgn:+d}/{tol:g}",
                            fn,
                            inputs,
                            pair(ref),
                            rtol=0.0,
                            atol=tol * float(np.max(np.abs(ref))),
                            m=m,
                            sign=sgn,
                            tolerance=tol,
                        )
                    )
    return out
