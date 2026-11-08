"""Calibrated two-view geometry against OpenCV on noise-free correspondences: the essential matrix (compared up to sign
at unit Frobenius norm; OpenCV runs the five-point solver), the relative pose recovered with the cheirality check,
and linear triangulation of noisy points (dehomogenised)."""

from importlib.metadata import version

import cv2
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}

K = np.array([[320.0, 0, 160], [0, 310.0, 120], [0, 0, 1]])


def canonical(e: FloatArray) -> FloatArray:
    """Unit Frobenius norm, the sign that makes the largest entry positive."""
    e = e / np.linalg.norm(e)
    return e * np.sign(e.ravel()[np.argmax(np.abs(e))])


def cases() -> list[Case]:
    g = rng(4420)
    out: list[Case] = []
    for k, (rv, t) in enumerate(
        [
            (np.array([0.05, -0.2, 0.03]), np.array([-1.0, 0.1, 0.2])),
            (np.array([-0.1, 0.1, 0.0]), np.array([0.3, -0.8, 0.1])),
        ]
    ):
        X = np.c_[g.uniform(-2, 2, 30), g.uniform(-1.5, 1.5, 30), g.uniform(4, 9, 30)]
        r, _ = cv2.Rodrigues(rv)
        a = X @ K.T
        b = (X @ r.T + t) @ K.T
        p1 = a[:, :2] / a[:, 2:]
        p2 = b[:, :2] / b[:, 2:]
        e, _ = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
        out.append(
            case(
                f"findEssentialMat/{k}",
                "findEssentialMat",
                [p1, p2, K],
                canonical(np.asarray(e[:3], dtype=np.float64)),
                rtol=1e-7,
                atol=1e-7,
            )
        )
        _, R, tt, mask = cv2.recoverPose(e[:3], p1, p2, K)
        out.append(
            case(
                f"recoverPose/{k}",
                "recoverPose",
                [e[:3], p1, p2, K],
                np.concatenate([R.ravel(), tt.ravel(), [float((mask > 0).sum())]]),
                rtol=1e-7,
                atol=1e-7,
            )
        )
        P1 = K @ np.c_[np.eye(3), np.zeros(3)]
        P2 = K @ np.c_[r, t]
        n1 = p1 + g.normal(0, 0.3, p1.shape)
        n2 = p2 + g.normal(0, 0.3, p2.shape)
        Xh = cv2.triangulatePoints(P1, P2, n1.T, n2.T)
        out.append(
            case(
                f"triangulatePoints/{k}",
                "triangulatePoints",
                [P1, P2, n1, n2],
                (Xh[:3] / Xh[3]).T,
                rtol=1e-8,
                atol=1e-8,
            )
        )
    return out
