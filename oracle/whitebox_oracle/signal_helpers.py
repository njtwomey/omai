"""Shared inputs for the signal case modules: short test signals, images and complex results as real pairs."""

import numpy as np
import numpy.typing as npt

from .helpers import FloatArray


def pair(z: npt.ArrayLike) -> FloatArray:
    """A complex result as stacked real and imaginary parts, shape `[2, ...]`."""
    c = np.asarray(z, dtype=np.complex128)
    return np.stack([c.real, c.imag])


def tones(
    g: np.random.Generator, n: int, fs: float = 1.0, freqs: tuple[float, ...] = (0.1, 0.23), noise: float = 0.1
) -> FloatArray:
    """A sum of sinusoids at the given frequencies (in the units of fs) plus Gaussian noise."""
    t = np.arange(n) / fs
    x = sum(np.cos(2 * np.pi * f * t + 0.3 * k) / (k + 1) for k, f in enumerate(freqs))
    return np.asarray(x + noise * g.normal(size=n), dtype=np.float64)


def smooth_image(g: np.random.Generator, h: int, w: int) -> FloatArray:
    """A smooth grey image in [0, 1] with a little noise."""
    yy, xx = np.mgrid[:h, :w]
    base = 0.5 + 0.3 * np.sin(yy / 3.0) * np.cos(xx / 4.0) + 0.1 * np.cos((xx + yy) / 5.0)
    return np.asarray(np.clip(base + 0.03 * g.normal(size=(h, w)), 0, 1), dtype=np.float64)


def blocks_image(h: int, w: int) -> FloatArray:
    """A piecewise-constant image: two rectangles and a disc on a dark background."""
    img = np.zeros((h, w))
    img[h // 6 : h // 2, w // 6 : w // 2] = 0.8
    img[h // 2 : 5 * h // 6, w // 2 : 5 * w // 6] = 0.5
    yy, xx = np.mgrid[:h, :w]
    img[(yy - h // 4) ** 2 + (xx - 3 * w // 4) ** 2 < (min(h, w) // 8) ** 2] = 1.0
    return img
