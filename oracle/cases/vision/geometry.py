"""Planar and two-view geometry against OpenCV: getPerspectiveTransform, getAffineTransform, perspectiveTransform,
findHomography (all points with noise, where OpenCV refines by Levenberg-Marquardt to the same least-squares optimum,
and RANSAC with gross outliers, comparing the inlier mask and the fit to the inliers), estimateAffine2D,
findFundamentalMat (the eight-point algorithm on noise-free correspondences, and RANSAC inlier masks),
computeCorrespondEpilines and sampsonDistance."""

from importlib.metadata import version

import cv2
import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from scipy.optimize import least_squares

REFERENCE: dict[str, str] = {"opencv-python-headless": version("opencv-python-headless")}

EXACT = {"rtol": 1e-10, "atol": 1e-9}
FITTED = {"rtol": 1e-6, "atol": 1e-6}


def apply_h(h: FloatArray, pts: FloatArray) -> FloatArray:
    q = np.c_[pts, np.ones(len(pts))] @ h.T
    return q[:, :2] / q[:, 2:]


def reprojection(p: FloatArray, src: FloatArray, dst: FloatArray) -> FloatArray:
    """The reprojection residuals of a homography with H[2][2] = 1 given by its other eight entries."""
    return (apply_h(np.r_[p, 1.0].reshape(3, 3), src) - dst).ravel()


def two_views(g: np.random.Generator, n: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Pixel projections of random 3-D points into two cameras, and the intrinsics."""
    k = np.array([[320.0, 0, 160], [0, 310.0, 120], [0, 0, 1]])
    pts = np.c_[g.uniform(-2, 2, n), g.uniform(-1.5, 1.5, n), g.uniform(4, 9, n)]
    r, _ = cv2.Rodrigues(np.array([0.05, -0.2, 0.03]))
    t = np.array([-1.0, 0.1, 0.2])
    p1 = pts @ k.T
    p2 = (pts @ r.T + t) @ k.T
    return p1[:, :2] / p1[:, 2:], p2[:, :2] / p2[:, 2:], k


def cases() -> list[Case]:
    g = rng(4400)
    out: list[Case] = []
    src4 = np.array([[10.0, 12], [90, 8], [95, 70], [5, 80]])
    dst4 = np.array([[0.0, 0], [100, 5], [110, 90], [-3, 85]])
    out.append(
        case(
            "getPerspectiveTransform",
            "getPerspectiveTransform",
            [src4, dst4],
            cv2.getPerspectiveTransform(src4.astype(np.float32), dst4.astype(np.float32)),
            rtol=1e-6,
            atol=1e-8,
        )
    )
    out.append(
        case(
            "getAffineTransform",
            "getAffineTransform",
            [src4[:3], dst4[:3]],
            cv2.getAffineTransform(src4[:3].astype(np.float32), dst4[:3].astype(np.float32)),
            rtol=1e-6,
            atol=1e-8,
        )
    )
    h = np.array([[1.1, 0.05, 12.0], [-0.03, 0.95, 4.0], [1e-4, -2e-4, 1.0]])
    pts = g.uniform(0, 200, (30, 2))
    out.append(
        case(
            "perspectiveTransform",
            "perspectiveTransform",
            [pts, h],
            cv2.perspectiveTransform(pts.reshape(-1, 1, 2), h).reshape(-1, 2),
            **EXACT,
        )
    )
    noisy = apply_h(h, pts) + g.normal(0, 0.5, pts.shape)
    # OpenCV stops its refinement after ten Levenberg-Marquardt steps (within about 1e-6 of the optimum): the reference
    # is scipy's least squares run to convergence from OpenCV's estimate.
    hh, _ = cv2.findHomography(pts, noisy, 0)
    best = least_squares(reprojection, hh.ravel()[:8], args=(pts, noisy), xtol=1e-15, ftol=1e-15, gtol=1e-15)
    out.append(
        case(
            "findHomography/all",
            "findHomography",
            [pts, noisy],
            np.r_[best.x, 1.0].reshape(3, 3),
            **FITTED,
            method="all",
        )
    )
    outl = noisy.copy()
    outl[::6] += g.uniform(40, 80, outl[::6].shape)
    hr, mask = cv2.findHomography(pts, outl, cv2.RANSAC, 3.0)
    # OpenCV 5 polishes its RANSAC fit its own way; the reference fit is scipy's least squares on its inliers.
    m = mask.ravel() > 0
    fit = least_squares(reprojection, hr.ravel()[:8], args=(pts[m], outl[m]), xtol=1e-15, ftol=1e-15, gtol=1e-15)
    enc = np.concatenate([fit.x, [1.0], mask.ravel().astype(np.float64)])
    out.append(case("findHomography/ransac", "findHomography", [pts, outl], enc, **FITTED, method="ransac"))
    a = np.array([[0.9, -0.2, 5.0], [0.15, 1.1, -3.0]])
    adst = pts @ a[:, :2].T + a[:, 2] + g.normal(0, 0.3, pts.shape)
    aoutl = adst.copy()
    aoutl[::5] += 50
    ma, amask = cv2.estimateAffine2D(pts, aoutl, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    enc = np.concatenate([ma.ravel(), amask.ravel().astype(np.float64)])
    out.append(case("estimateAffine2D/ransac", "estimateAffine2D", [pts, aoutl], enc, **FITTED, method="ransac"))
    # Over every point the affine fit is plain least squares (numpy's).
    sol, *_ = np.linalg.lstsq(np.c_[pts, np.ones(len(pts))], adst, rcond=None)
    out.append(case("estimateAffine2D/all", "estimateAffine2D", [pts, adst], sol.T, **EXACT, method="all"))
    p1, p2, _ = two_views(g, 20)
    f, _ = cv2.findFundamentalMat(p1, p2, cv2.FM_8POINT)
    out.append(case("findFundamentalMat/8point", "findFundamentalMat", [p1, p2], f, **FITTED, method="all"))
    q1, q2, _ = two_views(g, 40)
    q2n = q2 + g.normal(0, 0.1, q2.shape)
    q2n[::7] += g.uniform(30, 60, q2n[::7].shape)
    _, fmask = cv2.findFundamentalMat(q1, q2n, cv2.FM_RANSAC, 3.0, 0.999)
    out.append(
        case(
            "findFundamentalMat/ransac",
            "findFundamentalMat",
            [q1, q2n],
            fmask.ravel().astype(np.float64),
            **EXACT,
            method="ransac",
            threshold=3.0,
        )
    )
    for which in [1, 2]:
        lines = cv2.computeCorrespondEpilines(p1[:6].reshape(-1, 1, 2), which, f).reshape(-1, 3)
        out.append(
            case(
                f"computeCorrespondEpilines/{which}",
                "computeCorrespondEpilines",
                [p1[:6], f],
                lines.astype(np.float64),
                rtol=1e-6,
                atol=1e-7,
                whichImage=which,
            )
        )
    sd = [cv2.sampsonDistance(np.r_[p1[i], 1.0], np.r_[p2[i] + 0.5, 1.0], f) for i in range(6)]
    out.append(case("sampsonDistance", "sampsonDistance", [p1[:6], p2[:6] + 0.5, f], np.asarray(sd), **EXACT))
    return out
