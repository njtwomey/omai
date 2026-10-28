"""The pinhole camera against OpenCV: Rodrigues both ways, projectPoints with five and eight distortion coefficients,
undistortPoints (OpenCV's default five fixed-point iterations, with and without R and P), initUndistortRectifyMap
(float32 maps) and solvePnP on noisy planar and non-planar views (Levenberg-Marquardt reaches the same optimum)."""

from importlib.metadata import version

import cv2
import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}

EXACT = {"rtol": 1e-10, "atol": 1e-10}

K = np.array([[320.0, 0, 161], [0, 315.0, 118], [0, 0, 1]])
D5 = np.array([-0.21, 0.05, 0.001, -0.002, 0.01])
D8 = np.array([-0.21, 0.05, 0.001, -0.002, 0.01, 0.02, -0.01, 0.005])


def cases() -> list[Case]:
    g = rng(4410)
    out: list[Case] = []
    for k, rv in enumerate([np.array([0.1, -0.2, 0.3]), np.array([1.2, 0.4, -2.0]), np.array([0.0, 0.0, 0.0])]):
        r, _ = cv2.Rodrigues(rv)
        out.append(case(f"rodrigues/{k}", "rodrigues", [rv], r, **EXACT))
        back, _ = cv2.Rodrigues(r)
        out.append(case(f"rodriguesVector/{k}", "rodriguesVector", [r], back.ravel(), rtol=1e-9, atol=1e-12))
    X = np.c_[g.uniform(-1, 1, 25), g.uniform(-1, 1, 25), g.uniform(4, 7, 25)]
    rvec = np.array([0.05, -0.1, 0.02])
    tvec = np.array([0.1, -0.2, 0.5])
    for name, d in [("none", np.zeros(5)), ("d5", D5), ("d8", D8)]:
        proj, _ = cv2.projectPoints(X, rvec, tvec, K, d)
        out.append(
            case(
                f"projectPoints/{name}",
                "projectPoints",
                [X, rvec, tvec, K, d],
                proj.reshape(-1, 2),
                **EXACT,
            )
        )
    proj, _ = cv2.projectPoints(X, rvec, tvec, K, D5)
    img = proj.reshape(-1, 2)
    out.append(
        case(
            "undistortPoints/plain",
            "undistortPoints",
            [img, K, D5],
            cv2.undistortPoints(img, K, D5).reshape(-1, 2),
            **EXACT,
        )
    )
    R1, _ = cv2.Rodrigues(np.array([0.01, 0.02, -0.01]))
    P = np.array([[300.0, 0, 150, 0], [0, 300.0, 110, 0], [0, 0, 1, 0]])
    und = cv2.undistortPoints(img, K, D5, R=R1, P=P).reshape(-1, 2)
    out.append(case("undistortPoints/RP", "undistortPoints", [img, K, D5, R1, P], und, **EXACT))
    newk = np.array([[300.0, 0, 30], [0, 300.0, 22], [0, 0, 1]])
    mx, my = cv2.initUndistortRectifyMap(K, D5, R1, newk, (64, 48), cv2.CV_32FC1)
    out.append(
        case(
            "initUndistortRectifyMap",
            "initUndistortRectifyMap",
            [K, D5, R1, newk],
            np.concatenate([mx.ravel(), my.ravel()]).astype(np.float64),
            rtol=1e-6,
            atol=2e-4,
            size=[64, 48],
        )
    )
    board = np.array([[i % 6, i // 6, 0] for i in range(24)], np.float64) * 0.1
    cloud = np.c_[g.uniform(-0.5, 0.5, 15), g.uniform(-0.5, 0.5, 15), g.uniform(-0.3, 0.3, 15)]
    for name, obj, rv, tv in [
        ("planar", board, np.array([0.3, -0.2, 0.1]), np.array([-0.2, -0.1, 1.5])),
        ("cloud", cloud, np.array([-0.1, 0.25, 0.05]), np.array([0.1, 0.05, 2.0])),
    ]:
        p, _ = cv2.projectPoints(obj, rv, tv, K, D5)
        noisy = p.reshape(-1, 2) + g.normal(0, 0.3, (len(obj), 2))
        ok, r_est, t_est = cv2.solvePnP(obj, noisy, K, D5, flags=cv2.SOLVEPNP_ITERATIVE)
        assert ok
        out.append(
            case(
                f"solvePnP/{name}",
                "solvePnP",
                [obj, noisy, K, D5],
                np.concatenate([r_est.ravel(), t_est.ravel()]),
                rtol=1e-6,
                atol=1e-6,
            )
        )
    return out
