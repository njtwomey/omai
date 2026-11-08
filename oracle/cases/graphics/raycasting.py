"""Small reference frames for grid ray casting, rendered independently of the DDA: each column's ray is intersected
with every face of every opaque cell (axis-aligned segments, brute force) to find the nearest wall, then the frame is
drawn the textbook way: fish-eye corrected distance, wall slices of height `f / d`, flat or nearest-sampled textured
walls with darker y sides, floor and ceiling casting, exponential fog and one billboard sprite against the depth
buffer."""

import math

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__}

ROWS = ["#########", "#.......#", "#..#....#", "#.....%.#", "#.......#", "#########"]
OPAQUE = {"#": "wall", "%": "forest"}
WIDTH, HEIGHT, FOV, EYE = 16, 12, math.pi / 3, 0.5
X, Y, HEADING = 1.5, 4.2, -0.6

type Hit = tuple[float, str, str, float]


def nearest_wall(angle: float) -> Hit:
    """Distance, terrain, side ('x' for faces on vertical lines) and texture column of the nearest opaque face."""
    dx, dy = math.cos(angle), math.sin(angle)
    best: Hit = (math.inf, "", "", 0.0)
    for r, row in enumerate(ROWS):
        for c, ch in enumerate(row):
            if ch not in OPAQUE:
                continue
            for xline in (c, c + 1):
                if dx != 0:
                    t = (xline - X) / dx
                    yy = Y + t * dy
                    if 1e-12 < t < best[0] and r <= yy <= r + 1:
                        f = yy - math.floor(yy)
                        best = (t, OPAQUE[ch], "x", f if dx > 0 else 1 - f)
            for yline in (r, r + 1):
                if dy != 0:
                    t = (yline - Y) / dy
                    xx = X + t * dx
                    if 1e-12 < t < best[0] and c <= xx <= c + 1:
                        f = xx - math.floor(xx)
                        best = (t, OPAQUE[ch], "y", 1 - f if dy > 0 else f)
    return best


def nearest(tex: FloatArray, u: float, v: float) -> FloatArray:
    h, w, _ = tex.shape
    return tex[math.floor(v * h) % h, math.floor(u * w) % w]


def render(
    walls: dict[str, FloatArray | None],
    textures: dict[str, FloatArray],
    floor: FloatArray,
    ceiling: FloatArray,
    floor_tex: FloatArray | None,
    fog: tuple[FloatArray, float] | None,
    sprite: tuple[float, float, FloatArray] | None,
) -> tuple[FloatArray, FloatArray]:
    focal = WIDTH / 2 / math.tan(FOV / 2)
    plane = math.tan(FOV / 2)
    fx, fy = math.cos(HEADING), math.sin(HEADING)
    image = np.zeros((HEIGHT, WIDTH, 3))
    depth = np.zeros(WIDTH)

    def fogged(c: FloatArray, d: float) -> FloatArray:
        if fog is None:
            return c
        k = math.exp(-fog[1] * d)
        return c * k + fog[0] * (1 - k)

    for x in range(WIDTH):
        camera_x = 2 * (x + 0.5) / WIDTH - 1
        angle = HEADING + math.atan(camera_x * plane)
        t, terrain, side, u = nearest_wall(angle)
        perp = t * math.cos(angle - HEADING)
        depth[x] = perp
        top, bottom = HEIGHT / 2 - focal * (1 - EYE) / perp, HEIGHT / 2 + focal * EYE / perp
        rx, ry = fx - fy * camera_x * plane, fy + fx * camera_x * plane
        for y in range(HEIGHT):
            yc = y + 0.5
            if top <= yc < bottom:
                flat = walls.get(terrain)
                colour = flat if flat is not None else nearest(textures[terrain], u, (yc - top) / (bottom - top))
                if side == "y":
                    colour = colour * 0.7
                image[y, x] = fogged(colour, perp)
                continue
            below = yc > HEIGHT / 2
            dist = focal * EYE / (yc - HEIGHT / 2) if below else focal * (1 - EYE) / (HEIGHT / 2 - yc)
            px, py = X + dist * rx, Y + dist * ry
            colour = ceiling
            if below:
                colour = floor if floor_tex is None else nearest(floor_tex, px - math.floor(px), py - math.floor(py))
            image[y, x] = fogged(colour, dist)
    if sprite is not None:
        sx, sy, colour = sprite
        ox, oy = sx - X, sy - Y
        d = ox * fx + oy * fy
        lateral = -ox * fy + oy * fx
        size = focal * 0.6 / d
        centre = WIDTH / 2 + focal * lateral / d
        bottom = HEIGHT / 2 + focal * EYE / d
        top, left = bottom - size, centre - size / 2
        for x in range(max(0, math.ceil(left - 0.5)), min(WIDTH, math.ceil(left + size - 0.5))):
            if not d < depth[x]:
                continue
            for y in range(max(0, math.ceil(top - 0.5)), min(HEIGHT, math.ceil(bottom - 0.5))):
                image[y, x] = fogged(colour, d)
    return image, depth


def cases() -> list[Case]:
    g = rng(161)
    grey, sky = np.array([0.3, 0.3, 0.3]), np.array([0.5, 0.6, 0.9])
    flat: dict[str, FloatArray | None] = {"wall": np.array([0.8, 0.7, 0.6]), "forest": np.array([0.1, 0.6, 0.2])}
    plain, depth = render(flat, {}, grey, sky, None, None, None)
    textures = {"wall": g.uniform(0, 1, size=(4, 4, 3)), "forest": g.uniform(0, 1, size=(4, 4, 3))}
    floor_tex = g.uniform(0, 1, size=(4, 4, 3))
    fog = (np.array([1.0, 1.0, 1.0]), 0.1)
    textured, _ = render({"wall": None, "forest": None}, textures, grey, sky, floor_tex, fog, None)
    sprite = (3.2, 3.4, np.array([1.0, 0.8, 0.2]))
    with_sprite, _ = render(flat, {}, grey, sky, None, None, sprite)
    common = {"rows": ROWS, "x": X, "y": Y, "heading": HEADING, "width": WIDTH, "height": HEIGHT, "fov": FOV}
    return [
        case("flat", "frame", [], plain, rtol=1e-10, atol=1e-12, depth=depth.tolist(), **common),
        case(
            "textured",
            "frame",
            [textures["wall"], textures["forest"], floor_tex],
            textured,
            rtol=1e-10,
            atol=1e-12,
            **common,
        ),
        case("sprite", "frame", [], with_sprite, rtol=1e-10, atol=1e-12, sprite=[sprite[0], sprite[1]], **common),
    ]
