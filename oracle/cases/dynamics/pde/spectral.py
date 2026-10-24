"""Spectral differentiation against numpy: FFT derivatives (numpy.fft) and Chebyshev collocation (numpy.polynomial).

The Chebyshev differentiation matrix applied to samples of a polynomial must give the derivative of its Chebyshev
interpolant, which numpy.polynomial.chebyshev computes from the fitted coefficients.
"""

import numpy as np
import numpy.polynomial.chebyshev as cheb
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"numpy": np.__version__}


def periodic(x: FloatArray) -> FloatArray:
    return np.exp(np.sin(x)) + np.cos(3 * x)


def fft_derivative(u: FloatArray, length: float, order: int) -> FloatArray:
    n = u.size
    k = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    mult = (1j * k) ** order
    if order % 2 == 1:
        mult[n // 2] = 0
    return np.real(np.fft.ifft(mult * np.fft.fft(u)))


def cases() -> list[Case]:
    out: list[Case] = []
    for n in [16, 32]:
        x = 2 * np.pi * np.arange(n) / n
        u = periodic(x)
        for order in [1, 2, 3]:
            out.append(
                case(
                    f"fourier/{n}/{order}",
                    "fourier",
                    [u],
                    fft_derivative(u, 2 * np.pi, order),
                    rtol=1e-10,
                    atol=1e-10,
                    order=order,
                )
            )
    for n in [4, 8, 12]:
        t = np.cos(np.pi * np.arange(n + 1) / n)
        f = np.exp(t) * np.sin(2 * t)
        coef = cheb.chebfit(t, f, n)
        out.append(
            case(f"chebyshev/{n}", "chebyshev", [f], cheb.chebval(t, cheb.chebder(coef)), rtol=1e-9, atol=1e-9, n=n)
        )
    return out
