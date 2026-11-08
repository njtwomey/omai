"""The Kalman filter and RTS smoother against filterpy (KalmanFilter.batch_filter, rts_smoother, log-likelihood).

A constant-velocity model observed in position with a random series; filterpy updates in Joseph form, as omai.
"""

import filterpy
import numpy as np
from filterpy.kalman import KalmanFilter
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"filterpy": filterpy.__version__}


def model() -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    dt = 0.5
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = 0.1 * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    R = np.array([[0.5]])
    x0 = np.array([0.0, 1.0])
    P0 = np.diag([1.0, 2.0])
    return F, H, Q, R, x0, P0


def cases() -> list[Case]:
    g = rng(26_3)
    F, H, Q, R, x0, P0 = model()
    zs = np.cumsum(g.normal(size=12)) * 0.5 + np.arange(12) * 0.5
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F, kf.H, kf.Q, kf.R, kf.x, kf.P = F, H, Q, R, x0.copy(), P0.copy()
    means, covs, pmeans, pcovs = kf.batch_filter(zs.reshape(-1, 1))
    xs, ps, _k, _pp = kf.rts_smoother(means, covs)
    ll = 0.0
    kf2 = KalmanFilter(dim_x=2, dim_z=1)
    kf2.F, kf2.H, kf2.Q, kf2.R, kf2.x, kf2.P = F, H, Q, R, x0.copy(), P0.copy()
    for z in zs:
        kf2.predict()
        kf2.update(np.array([z]))
        ll += kf2.log_likelihood
    inputs = [zs.reshape(-1, 1)]
    out = [
        case(
            "filter/means",
            "means",
            inputs,
            np.asarray(means)[:, :, 0] if np.ndim(means) == 3 else np.asarray(means),
            rtol=1e-10,
            atol=1e-12,
        ),
        case("filter/covariances", "covariances", inputs, np.asarray(covs), rtol=1e-10, atol=1e-12),
        case(
            "filter/predicted",
            "predicted",
            inputs,
            np.asarray(pmeans)[:, :, 0] if np.ndim(pmeans) == 3 else np.asarray(pmeans),
            rtol=1e-10,
            atol=1e-12,
        ),
        case(
            "smoother/means",
            "smoothed",
            inputs,
            np.asarray(xs)[:, :, 0] if np.ndim(xs) == 3 else np.asarray(xs),
            rtol=1e-10,
            atol=1e-12,
        ),
        case("smoother/covariances", "smoothed_cov", inputs, np.asarray(ps), rtol=1e-10, atol=1e-12),
        case("loglik", "loglik", inputs, np.array(ll), rtol=1e-10, atol=1e-12),
    ]
    _ = pcovs
    return out
