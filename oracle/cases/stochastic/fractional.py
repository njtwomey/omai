"""Fractional processes: the fractional Gaussian noise autocovariance and the fBm covariance (closed forms), and Hurst
exponent estimates against nolds (hurst_rs with least-squares fits, with and without the Anis-Lloyd-Peters
correction, and dfa with half-overlapping or disjoint windows and detrending orders 1 and 2) on fixed series."""

import nolds
import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"nolds": "0.6.3", "numpy": np.__version__}


def fgn_acov(k: FloatArray, h: float) -> FloatArray:
    return 0.5 * (np.abs(k + 1) ** (2 * h) - 2 * np.abs(k) ** (2 * h) + np.abs(k - 1) ** (2 * h))


def cases() -> list[Case]:
    out: list[Case] = []
    k = np.arange(0.0, 8.0)
    for h in [0.3, 0.5, 0.8]:
        out.append(case(f"fgn-acov/{h}", "fgn-acov", [k], fgn_acov(k, h), rtol=1e-12, atol=1e-15, hurst=h))
    t = np.array([0.2, 0.5, 1.0, 1.7])
    h = 0.7
    cov = 0.5 * (
        np.abs(t[:, None]) ** (2 * h) + np.abs(t[None, :]) ** (2 * h) - np.abs(t[:, None] - t[None, :]) ** (2 * h)
    )
    out.append(case("fbm-cov", "fbm-cov", [t], cov, rtol=1e-12, hurst=h))
    g = rng(9)
    white = g.normal(size=600)
    # a persistent series: a moving sum of white noise
    smooth = np.convolve(g.normal(size=640), np.ones(40) / 40, mode="valid")[:600]
    windows = [10, 16, 25, 40, 60, 100]
    for name, x in [("white", white), ("smooth", smooth)]:
        for corrected in [True, False]:
            value = nolds.hurst_rs(x, nvals=windows, fit="poly", corrected=corrected, unbiased=True)
            out.append(
                case(
                    f"hurst-rs/{name}/{corrected}",
                    "hurst-rs",
                    [x],
                    np.array(value),
                    rtol=1e-10,
                    windows=windows,
                    corrected=corrected,
                )
            )
        for overlap, order in [(True, 1), (False, 1), (True, 2)]:
            value = nolds.dfa(x, nvals=windows, overlap=overlap, order=order, fit_trend="poly", fit_exp="poly")
            out.append(
                case(
                    f"dfa/{name}/{overlap}/{order}",
                    "dfa",
                    [x],
                    np.array(value),
                    rtol=1e-9,
                    windows=windows,
                    overlap=overlap,
                    order=order,
                )
            )
    return out
