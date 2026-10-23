"""Rotations, alignment and projective geometry against scipy.spatial.transform, scipy.linalg, scipy.spatial and
skimage.transform."""

from importlib.metadata import version
from typing import Literal

import numpy as np
import scipy
import scipy.linalg as sl
import scipy.spatial as ss
from scipy.spatial.transform import Rotation, Slerp
from skimage.transform import FundamentalMatrixTransform, ProjectiveTransform
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "skimage": version("scikit-image")}

type Seq = Literal["xyz", "zyx", "xzx", "zxz", "yxy", "XYZ", "ZYX", "ZXZ", "YZY", "XZY"]

SEQUENCES: list[Seq] = ["xyz", "zyx", "xzx", "zxz", "yxy", "XYZ", "ZYX", "ZXZ", "YZY", "XZY"]


def canonical_quat(q: FloatArray) -> FloatArray:
    return -q if q[3] < 0 else q


def unit_fundamental(f: FloatArray) -> FloatArray:
    f = f / np.linalg.norm(f)
    return -f if f[2, 2] < 0 else f


def cases() -> list[Case]:
    g = rng(160)
    out: list[Case] = []
    quats = g.normal(size=(4, 4))
    vectors = g.normal(size=(5, 3))
    for i, q in enumerate(quats):
        r = Rotation.from_quat(q)
        out.append(case(f"matrix/{i}", "asMatrix", [q], r.as_matrix(), rtol=1e-13, atol=1e-15))
        out.append(case(f"rotvec/{i}", "asRotvec", [q], r.as_rotvec(), rtol=1e-12, atol=1e-14))
        out.append(case(f"apply/{i}", "apply", [q, vectors], r.apply(vectors), rtol=1e-13, atol=1e-14))
        out.append(
            case(f"fromMatrix/{i}", "fromMatrix", [r.as_matrix()], canonical_quat(r.as_quat()), rtol=1e-12, atol=1e-14)
        )
        out.append(
            case(f"fromRotvec/{i}", "fromRotvec", [r.as_rotvec()], canonical_quat(r.as_quat()), rtol=1e-12, atol=1e-14)
        )
        for seq in SEQUENCES:
            out.append(case(f"euler/{seq}/{i}", "asEuler", [q], r.as_euler(seq), rtol=1e-11, atol=1e-12, seq=seq))
    angles = g.uniform(-1, 1, size=3)
    for seq in SEQUENCES:
        out.append(
            case(
                f"fromEuler/{seq}",
                "fromEuler",
                [angles],
                canonical_quat(Rotation.from_euler(seq, angles).as_quat()),
                rtol=1e-12,
                atol=1e-14,
                seq=seq,
            )
        )
    r1 = Rotation.from_quat(quats[0])
    r2 = Rotation.from_quat(quats[1])
    out.append(
        case("compose", "compose", [quats[0], quats[1]], canonical_quat((r1 * r2).as_quat()), rtol=1e-12, atol=1e-14)
    )
    out.append(case("inverse", "inverse", [quats[0]], canonical_quat(r1.inv().as_quat()), rtol=1e-12, atol=1e-14))
    slerp = Slerp([0, 1], Rotation.concatenate([r1, r2]))
    out.append(
        case(
            "slerp",
            "slerp",
            [quats[0], quats[1]],
            canonical_quat(slerp([0.3]).as_quat()[0]),
            rtol=1e-10,
            atol=1e-12,
            t=0.3,
        )
    )

    a = g.normal(size=(8, 3))
    b = a @ Rotation.from_quat(quats[2]).as_matrix().T + 0.01 * g.normal(size=(8, 3))
    rot, scale = sl.orthogonal_procrustes(a, b)
    out.append(case("orthogonalProcrustes", "orthogonalProcrustes", [a, b], rot, rtol=1e-10, atol=1e-12))
    out.append(case("orthogonalProcrustes/scale", "orthogonalProcrustesScale", [a, b], np.array(scale), rtol=1e-12))
    _, m2, disparity = ss.procrustes(a, 2 * b + 1)
    out.append(case("procrustes/mtx2", "procrustesMtx2", [a, 2 * b + 1], m2, rtol=1e-10, atol=1e-12))
    out.append(case("procrustes/disparity", "procrustesDisparity", [a, 2 * b + 1], np.array(disparity), rtol=1e-9))
    target = b + np.array([1.0, -2.0, 0.5])
    aligned, rssd = Rotation.align_vectors(target - target.mean(0), a - a.mean(0))
    out.append(case("kabsch/rotation", "kabsch", [a, target], aligned.as_matrix(), rtol=1e-10, atol=1e-12))
    out.append(case("kabsch/rmsd", "kabschRmsd", [a, target], np.array(rssd / np.sqrt(8)), rtol=1e-9))

    src = g.uniform(0, 10, size=(10, 2))
    h_true = np.array([[1.1, 0.2, 3.0], [-0.1, 0.9, 1.0], [0.01, 0.02, 1.0]])
    hom = np.c_[src, np.ones(10)] @ h_true.T
    dst = hom[:, :2] / hom[:, 2:] + 1e-3 * g.normal(size=(10, 2))
    params = ProjectiveTransform.from_estimate(src, dst).params
    out.append(case("homography", "homography", [src, dst], params / params[2, 2], rtol=1e-8, atol=1e-10))
    x1 = g.uniform(0, 10, size=(12, 2))
    x2 = x1 + np.array([1.0, 0.2]) + 0.05 * g.normal(size=(12, 2))
    fparams = FundamentalMatrixTransform.from_estimate(x1, x2).params
    out.append(case("fundamental", "fundamental", [x1, x2], unit_fundamental(fparams), rtol=1e-7, atol=1e-9))
    return out
