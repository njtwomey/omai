"""The unscented and extended Kalman filters against filterpy (UnscentedKalmanFilter with MerweScaledSigmaPoints,
ExtendedKalmanFilter with the Jacobians written out), on a pendulum observed through its horizontal position.
"""

import filterpy
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter, MerweScaledSigmaPoints, UnscentedKalmanFilter
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"filterpy": filterpy.__version__}

DT = 0.1


def fx(x: FloatArray, dt: float = DT) -> FloatArray:
    return np.array([x[0] + dt * x[1], x[1] - dt * 9.81 * np.sin(x[0])])


def f_jac(x: FloatArray) -> FloatArray:
    return np.array([[1.0, DT], [-DT * 9.81 * np.cos(x[0]), 1.0]])


def hx(x: FloatArray) -> FloatArray:
    return np.array([np.sin(x[0])])


def h_jac(x: FloatArray) -> FloatArray:
    return np.array([[np.cos(x[0]), 0.0]])


class PendulumEKF(ExtendedKalmanFilter):
    def predict_x(self, u: object = 0) -> None:
        self.x = fx(self.x)


def cases() -> list[Case]:
    g = rng(26_4)
    Q = np.diag([1e-4, 1e-3])
    R = np.array([[0.01]])
    x0 = np.array([0.5, 0.0])
    P0 = np.diag([0.1, 0.1])
    truth = [x0]
    for _ in range(15):
        truth.append(fx(truth[-1]))
    zs = np.array([hx(x)[0] for x in truth[1:]]) + 0.1 * g.normal(size=15)
    points = MerweScaledSigmaPoints(2, alpha=0.3, beta=2.0, kappa=1.0)
    ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=DT, hx=hx, fx=fx, points=points)
    ukf.x, ukf.P, ukf.Q, ukf.R = x0.copy(), P0.copy(), Q, R
    um, uc = ukf.batch_filter(zs.reshape(-1, 1))
    ekf = PendulumEKF(dim_x=2, dim_z=1)
    ekf.x, ekf.P, ekf.Q, ekf.R = x0.copy(), P0.copy(), Q, R
    em: list[FloatArray] = []
    ec: list[FloatArray] = []
    for z in zs:
        ekf.F = f_jac(ekf.x)
        ekf.predict()
        ekf.update(np.array([z]), h_jac, hx)
        em.append(ekf.x.copy())
        ec.append(ekf.P.copy())
    inputs = [zs.reshape(-1, 1)]
    return [
        case("ukf/means", "ukf_means", inputs, np.asarray(um), rtol=1e-9, atol=1e-11),
        case("ukf/covariances", "ukf_covs", inputs, np.asarray(uc), rtol=1e-9, atol=1e-11),
        case("ekf/means", "ekf_means", inputs, np.asarray(em), rtol=1e-9, atol=1e-11),
        case("ekf/covariances", "ekf_covs", inputs, np.asarray(ec), rtol=1e-9, atol=1e-11),
    ]
