"""Cameras against numpy: gluLookAt, gluPerspective and glOrtho matrices, projection of points to pixels, pixel rays
recovered by unprojecting the near and far planes through the inverse view-projection matrix, and frustum planes
built from the eight corners of the view volume (independently of the row combinations the engine uses)."""

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__}


def unit(v: FloatArray) -> FloatArray:
    return v / np.linalg.norm(v)


def look_at(eye: FloatArray, target: FloatArray, up: FloatArray) -> FloatArray:
    f = unit(target - eye)
    s = unit(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov: float, aspect: float, near: float, far: float) -> FloatArray:
    f = 1 / np.tan(np.radians(fov) / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def orthographic(left: float, right: float, bottom: float, top: float, near: float, far: float) -> FloatArray:
    m = np.diag([2 / (right - left), 2 / (top - bottom), -2 / (far - near), 1.0])
    m[:3, 3] = [-(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near)]
    return m


def camera(g: np.random.Generator) -> tuple[FloatArray, FloatArray, FloatArray]:
    eye = g.uniform(-4, 4, size=3)
    eye[2] = abs(eye[2]) + 3
    target = g.uniform(-1, 1, size=3)
    up = np.array([0.0, 1.0, 0.0]) + g.uniform(-0.3, 0.3, size=3)
    return eye, target, up


def cases() -> list[Case]:
    g = rng(37)
    out: list[Case] = []
    for i in range(4):
        eye, target, up = camera(g)
        out.append(case(f"lookAt/{i}", "lookAt", [eye, target, up], look_at(eye, target, up), rtol=1e-12, atol=1e-13))
        fov, aspect, near, far = float(g.uniform(20, 120)), float(g.uniform(0.5, 2)), float(g.uniform(0.05, 1)), 50.0
        out.append(
            case(
                f"perspective/{i}",
                "perspective",
                [],
                perspective(fov, aspect, near, far),
                fov=fov,
                aspect=aspect,
                near=near,
                far=far,
            )
        )
        lo = g.uniform(-3, -0.5, size=3)
        hi = g.uniform(0.5, 3, size=3)
        box = [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(-lo[2]), float(hi[2] + 4)]
        out.append(case(f"orthographic/{i}", "orthographic", [], orthographic(*box), box=box))
    for i in range(6):
        eye, target, up = camera(g)
        kind = "pinhole" if i % 2 == 0 else "orthographic"
        width, height = 16 + 4 * i, 12
        aspect = width / height
        fov, size, near, far = float(g.uniform(30, 90)), float(g.uniform(0.5, 2)), 0.1, 40.0
        proj = (
            perspective(fov, aspect, near, far)
            if kind == "pinhole"
            else orthographic(-size * aspect, size * aspect, -size, size, near, far)
        )
        vp = proj @ look_at(eye, target, up)
        points = g.uniform(-2, 2, size=(5, 3))
        clip = np.hstack([points, np.ones((5, 1))]) @ vp.T
        ndc = clip[:, :3] / clip[:, 3:]
        pixels = np.stack([(ndc[:, 0] + 1) * width / 2, (1 - ndc[:, 1]) * height / 2, ndc[:, 2]], axis=1)
        # Pixel rays by unprojection: the points of the near and far planes under the pixel, through inv(P V).
        coords = g.uniform(0, 1, size=(5, 2)) * [width, height]
        inv = np.linalg.inv(vp)
        sx = 2 * coords[:, 0] / width - 1
        sy = 1 - 2 * coords[:, 1] / height
        near_h = np.stack([sx, sy, -np.ones(5), np.ones(5)], axis=1) @ inv.T
        far_h = np.stack([sx, sy, np.ones(5), np.ones(5)], axis=1) @ inv.T
        p_near = near_h[:, :3] / near_h[:, 3:]
        p_far = far_h[:, :3] / far_h[:, 3:]
        directions = (p_far - p_near) / np.linalg.norm(p_far - p_near, axis=1, keepdims=True)
        # A pinhole ray starts at the eye; an orthographic one on the plane through the eye.
        origins = np.tile(eye, (5, 1)) if kind == "pinhole" else p_near - near * directions
        # Frustum planes from the corners of the view volume, oriented towards its centre.
        corner_h = np.array([[x, y, z, 1.0] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)]) @ inv.T
        corners = corner_h[:, :3] / corner_h[:, 3:]
        centre = corners.mean(axis=0)
        faces = [[0, 2, 4], [1, 3, 5], [0, 1, 4], [2, 3, 6], [0, 1, 2], [4, 5, 6]]
        planes: list[list[float]] = []
        for a, b, c in faces:
            n = unit(np.cross(corners[b] - corners[a], corners[c] - corners[a]))
            d = -float(n @ corners[a])
            if n @ centre + d < 0:
                n, d = -n, -d
            planes.append([float(x) for x in n] + [d])
        out.append(
            case(
                f"view/{i}",
                "view",
                [eye, target, up, points, coords],
                pixels,
                rtol=1e-10,
                atol=1e-10,
                kind=kind,
                width=width,
                height=height,
                fov=fov,
                size=size,
                near=near,
                far=far,
                origins=origins.tolist(),
                directions=directions.tolist(),
                planes=planes,
            )
        )
    return out
