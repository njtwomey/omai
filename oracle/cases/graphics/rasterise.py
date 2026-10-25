"""A small reference image for triangle rasterisation, rendered independently of edge functions: vertices through
numpy view and projection matrices, then for every pixel centre and every triangle that faces the camera the
barycentric coordinates by numpy.linalg.solve, inside when all are positive, the depth interpolated linearly in screen
space, the nearest triangle kept, and vertex colours interpolated perspective-correctly (weights divided by w). The
scene is regenerated until no pixel centre lies within 0.005 pixels of an edge, so coverage is unambiguous."""

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__}

WIDTH, HEIGHT = 16, 12
EYE = np.array([0.5, 1.0, 4.0])
FOV, NEAR, FAR = 50.0, 0.1, 50.0


def unit(v: FloatArray) -> FloatArray:
    return v / np.linalg.norm(v)


def view_projection() -> FloatArray:
    f = unit(-EYE)
    s = unit(np.cross(f, [0.0, 1.0, 0.0]))
    u = np.cross(s, f)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = s, u, -f
    view[:3, 3] = -view[:3, :3] @ EYE
    t = 1 / np.tan(np.radians(FOV) / 2)
    proj = np.zeros((4, 4))
    proj[0, 0], proj[1, 1] = t / (WIDTH / HEIGHT), t
    proj[2, 2], proj[2, 3], proj[3, 2] = (FAR + NEAR) / (NEAR - FAR), 2 * FAR * NEAR / (NEAR - FAR), -1
    return proj @ view


def render(positions: FloatArray, colours: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, float] | None:
    vp = view_projection()
    clip = np.hstack([positions, np.ones((len(positions), 1))]) @ vp.T
    w = clip[:, 3]
    screen = np.stack([(clip[:, 0] / w + 1) * WIDTH / 2, (1 - clip[:, 1] / w) * HEIGHT / 2, clip[:, 2] / w], axis=1)
    image = np.zeros((HEIGHT, WIDTH, 3))
    depth = np.full((HEIGHT, WIDTH), np.inf)
    ids = np.full((HEIGHT, WIDTH), -1.0)
    margin = np.inf
    for k in range(len(positions) // 3):
        a, b, c = positions[3 * k : 3 * k + 3]
        if float(np.cross(b - a, c - a) @ (EYE - a)) <= 0:
            continue  # a back face
        p0, p1, p2 = screen[3 * k : 3 * k + 3]
        m = np.array([[p1[0] - p0[0], p2[0] - p0[0]], [p1[1] - p0[1], p2[1] - p0[1]]])
        edges = [(p0, p1), (p1, p2), (p2, p0)]
        for y in range(HEIGHT):
            for x in range(WIDTH):
                q = np.array([x + 0.5, y + 0.5])
                l1, l2 = np.linalg.solve(m, q - p0[:2])
                lam = np.array([1 - l1 - l2, l1, l2])
                for e0, e1 in edges:
                    d = e1[:2] - e0[:2]
                    s = float(np.clip((q - e0[:2]) @ d / (d @ d), 0, 1))
                    margin = min(margin, float(np.linalg.norm(q - e0[:2] - s * d)))
                if np.any(lam <= 0):
                    continue
                z = float(lam @ screen[3 * k : 3 * k + 3, 2])
                if not z < depth[y, x]:
                    continue
                pw = lam / w[3 * k : 3 * k + 3]
                pw = pw / pw.sum()
                depth[y, x], ids[y, x] = z, k
                image[y, x] = pw @ colours[3 * k : 3 * k + 3]
    return image, depth, ids, margin


# Three overlapping triangles facing the camera (counter-clockwise seen from the eye) and one facing away.
BASE = np.array(
    [
        [[-1.4, -1.0, 0.0], [1.2, -1.2, -0.5], [0.0, 1.3, 0.2]],
        [[-0.6, -0.4, 0.8], [1.5, 0.2, -0.2], [-0.8, 1.1, -0.6]],
        [[-1.5, 0.6, -1.0], [-0.2, -1.4, 0.6], [1.3, 1.0, 0.4]],
        [[-1.0, -1.0, 1.0], [0.0, 1.0, 1.0], [1.0, -1.0, 1.0]],
    ]
)


def cases() -> list[Case]:
    g = rng(5150)
    colours = g.uniform(0, 1, size=(12, 3))
    # Jitter the vertices until no pixel centre is within 0.005 pixels of an edge.
    for _ in range(200):
        positions = (BASE + g.uniform(-0.05, 0.05, size=BASE.shape)).reshape(12, 3)
        out = render(positions, colours)
        if out is not None and out[3] > 0.005:
            break
    else:
        raise RuntimeError("rasterise: no unambiguous scene found")
    image, depth, ids, _ = out
    return [
        case(
            "triangles",
            "triangles",
            [positions, colours],
            image,
            rtol=1e-9,
            atol=1e-10,
            ids=ids.ravel().tolist(),
            depth=[float(v) if np.isfinite(v) else None for v in depth.ravel()],
        )
    ]
