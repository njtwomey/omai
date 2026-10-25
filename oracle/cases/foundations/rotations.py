"""Rotation matrices, Euler angles, quaternions and rotation averaging against scipy.spatial.transform.Rotation."""

from typing import Literal

import numpy as np
import scipy
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from scipy.spatial.transform import Rotation, Slerp

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

type Seq = Literal[
    "xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "xyx", "xzx", "yxy", "yzy", "zxz", "zyz",
    "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX", "XYX", "XZX", "YXY", "YZY", "ZXZ", "ZYZ",
]  # fmt: skip

SEQUENCES: list[Seq] = [
    "xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "xyx", "xzx", "yxy", "yzy", "zxz", "zyz",
    "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX", "XYX", "XZX", "YXY", "YZY", "ZXZ", "ZYZ",
]  # fmt: skip


def canonical(q: FloatArray) -> FloatArray:
    """The sign of a quaternion with w >= 0 (both signs are the same rotation)."""
    return np.where(q[..., 3:] < 0, -q, q)


def cases() -> list[Case]:
    g = rng(1203)
    out: list[Case] = []
    angles = g.uniform(-1.5, 1.5, size=4)
    for axis in "xyz":
        m = Rotation.from_euler(axis, angles[:, None]).as_matrix()
        out.append(case(f"elementary/{axis}", f"elementary{axis.upper()}", [angles], m, rtol=1e-14, atol=1e-15))
    axes = g.normal(size=(4, 3))
    units = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    rotvecs = units * angles[:, None]
    vectors = g.normal(size=(4, 3))
    out.append(
        case("axisAngle", "axisAngleToMatrix", [axes, angles], Rotation.from_rotvec(rotvecs).as_matrix(), atol=1e-15)
    )
    out.append(
        case("rodrigues", "rotateAboutAxis", [vectors, axes, angles], Rotation.from_rotvec(rotvecs).apply(vectors))
    )
    for i, m in enumerate(Rotation.from_rotvec(rotvecs).as_matrix()):
        out.append(case(f"toAxisAngle/{i}", "matrixToRotvec", [m], rotvecs[i], rtol=1e-12, atol=1e-14))

    euler = g.uniform(-1.2, 1.2, size=(3, 3))
    euler[:, 1] = np.abs(euler[:, 1]) + 0.1
    for seq in SEQUENCES:
        r = Rotation.from_euler(seq, euler)
        out.append(case(f"euler/{seq}", "eulerToMatrix", [euler], r.as_matrix(), rtol=1e-13, atol=1e-15, seq=seq))
        out.append(case(f"toEuler/{seq}", "matrixToEuler", [r.as_matrix()[0]], r.as_euler(seq)[0], seq=seq))

    q1 = Rotation.random(5, rng=g).as_quat()
    q2 = Rotation.random(5, rng=g).as_quat()
    r1 = Rotation.from_quat(q1)
    r2 = Rotation.from_quat(q2)
    out.append(case("multiply", "quatMultiply", [q1, q2], canonical((r1 * r2).as_quat()), rtol=1e-12, atol=1e-14))
    out.append(case("rotate", "quatRotate", [q1, vectors[:1].repeat(5, 0)], r1.apply(vectors[0]), atol=1e-14))
    out.append(case("toMatrix", "quatToMatrix", [q1], r1.as_matrix(), rtol=1e-13, atol=1e-15))
    out.append(
        case("fromAxisAngle", "quatFromAxisAngle", [axes, angles], canonical(Rotation.from_rotvec(rotvecs).as_quat()))
    )
    for i, m in enumerate(r1.as_matrix()):
        out.append(case(f"fromMatrix/{i}", "matrixToQuat", [m], canonical(Rotation.from_matrix(m).as_quat())))
    for i in range(3):
        for t in [0.0, 0.25, 0.8]:
            pair = Rotation.concatenate([r1[i], r2[i]])
            want = canonical(Slerp([0.0, 1.0], pair)([t]).as_quat()[0])
            out.append(case(f"slerp/{i}/{t}", "quatSlerp", [q1[i], q2[i]], want, rtol=1e-11, atol=1e-13, t=t))
    out.append(case("distance", "rotationDistance", [q1, q2], (r1.inv() * r2).magnitude(), rtol=1e-12))

    for i, spread in enumerate([0.1, 0.5, 1.0]):
        centre = Rotation.random(rng=g)
        cloud = centre * Rotation.from_rotvec(spread * g.normal(size=(7, 3)))
        weights = g.uniform(0.2, 1.0, size=7)
        out.append(case(f"mean/{i}", "meanRotation", [cloud.as_quat()], canonical(cloud.mean().as_quat()), rtol=1e-10))
        out.append(
            case(
                f"mean/weighted/{i}",
                "meanRotationWeighted",
                [cloud.as_quat(), weights],
                canonical(cloud.mean(weights=weights).as_quat()),
                rtol=1e-10,
            )
        )
    return out
