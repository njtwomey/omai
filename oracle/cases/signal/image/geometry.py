"""Geometric transforms against scipy.ndimage (map_coordinates, affine_transform, shift, zoom, rotate; orders 0
and 1, every boundary mode) and scikit-image (resize, pyramids, integral images, projective warps)."""

from collections.abc import Callable
from importlib.metadata import version
from typing import Literal, cast

import numpy as np
import scipy
import scipy.ndimage as N
import skimage.transform as T
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.signal_helpers import smooth_image

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "scikit-image": version("scikit-image")}

TOL = {"rtol": 1e-12, "atol": 1e-12}
Mode = Literal["constant", "grid-constant", "nearest", "reflect", "mirror", "grid-wrap"]
MODES: list[Mode] = ["constant", "grid-constant", "nearest", "reflect", "mirror", "grid-wrap"]
SkMode = Literal["constant", "edge", "symmetric", "reflect", "wrap"]
SK_MODES: list[SkMode] = ["constant", "edge", "symmetric", "reflect", "wrap"]
# scikit-image's inferred signature types cval as int (from its default).
resize = cast(Callable[..., FloatArray], T.resize)


def cases() -> list[Case]:
    g = rng(480)
    out: list[Case] = []
    img = smooth_image(g, 7, 9)
    coords = np.stack([g.uniform(-1.5, 8, size=12), g.uniform(-1.5, 10, size=12)])
    orders: list[Literal[0, 1]] = [0, 1]
    for order in orders:
        for mode in MODES:
            r = N.map_coordinates(img, coords, order=order, mode=mode, cval=0.3)
            out.append(
                case(
                    f"mapCoordinates/{order}/{mode}",
                    "mapCoordinates",
                    [img, coords],
                    r,
                    **TOL,
                    order=order,
                    mode=mode,
                    cval=0.3,
                )
            )
    m = np.array([[0.9, 0.2], [-0.15, 1.1]])
    r = N.affine_transform(img, m, offset=[0.5, -0.7], order=1, mode="nearest")
    out.append(
        case("affineTransform", "affineTransform", [img, m], r, **TOL, offset=[0.5, -0.7], order=1, mode="nearest")
    )
    h = np.array([[1.1, 0.1, -0.5], [0.05, 0.9, 0.8], [0, 0, 1]])
    r = N.affine_transform(img, h, output_shape=(6, 10), order=1, mode="reflect")
    out.append(
        case(
            "affineTransform/homogeneous",
            "affineTransform",
            [img, h],
            r,
            **TOL,
            outputShape=[6, 10],
            order=1,
            mode="reflect",
        )
    )
    d = np.array([1.2, 0.8])
    out.append(
        case(
            "affineTransform/diagonal",
            "affineTransform",
            [img, d],
            N.affine_transform(img, d, order=0, mode="mirror"),
            **TOL,
            order=0,
            mode="mirror",
        )
    )
    out.append(
        case(
            "shift",
            "shift",
            [img],
            N.shift(img, [1.5, -0.25], order=1, mode="constant"),
            **TOL,
            shift=[1.5, -0.25],
            order=1,
            mode="constant",
        )
    )
    for grid in [False, True]:
        mode: Mode = "mirror" if grid else "nearest"
        r = N.zoom(img, [1.6, 0.7], order=1, mode=mode, grid_mode=grid)
        out.append(case(f"zoom/{grid}", "zoom", [img], r, **TOL, zoom=[1.6, 0.7], order=1, mode=mode, gridMode=grid))
    for angle in [90.0, 30.0, -47.0]:
        r = N.rotate(img, angle, order=1)
        out.append(case(f"rotate/{angle}", "rotate", [img], r, rtol=1e-10, atol=1e-10, angle=angle, order=1))
    out.append(
        case(
            "rotate/noreshape",
            "rotate",
            [img],
            N.rotate(img, 20.0, reshape=False, order=1, mode="nearest"),
            rtol=1e-10,
            atol=1e-10,
            angle=20.0,
            order=1,
            mode="nearest",
            reshape=False,
        )
    )
    for skmode in SK_MODES:
        r = resize(img, (11, 5), order=1, mode=skmode, cval=0.2)
        out.append(
            case(
                f"resize/{skmode}",
                "resize",
                [img],
                r,
                rtol=1e-10,
                atol=1e-12,
                outputShape=[11, 5],
                order=1,
                mode=skmode,
                cval=0.2,
            )
        )
    r = T.resize(img, (4, 4), order=0, anti_aliasing=False)
    out.append(case("resize/nearest", "resize", [img], r, **TOL, outputShape=[4, 4], order=0, antiAliasing=False))
    big = smooth_image(g, 16, 12)
    out.append(case("pyramidReduce", "pyramidReduce", [big], T.pyramid_reduce(big), rtol=1e-10, atol=1e-12))
    out.append(
        case(
            "pyramidReduce/3",
            "pyramidReduce",
            [big],
            T.pyramid_reduce(big, downscale=3, mode="constant"),
            rtol=1e-10,
            atol=1e-12,
            factor=3,
            mode="constant",
        )
    )
    out.append(case("pyramidExpand", "pyramidExpand", [img], T.pyramid_expand(img), rtol=1e-10, atol=1e-12))
    levels = list(T.pyramid_gaussian(big, max_layer=3))
    out.append(
        case(
            "pyramidGaussian",
            "pyramidGaussian",
            [big],
            np.concatenate([lv.ravel() for lv in levels]),
            rtol=1e-10,
            atol=1e-12,
            maxLayer=3,
        )
    )
    out.append(case("integralImage", "integralImage", [img], T.integral_image(img), **TOL))
    proj = np.array([[1.05, 0.08, -0.6], [-0.04, 0.97, 0.4], [0.002, -0.003, 1.0]])
    warp_modes: list[SkMode] = ["constant", "edge"]
    for skm in warp_modes:
        r = T.warp(img, T.ProjectiveTransform(proj), order=1, mode=skm, cval=0.1)
        out.append(
            case(
                f"warpPerspective/{skm}",
                "warpPerspective",
                [img, proj],
                r,
                rtol=1e-10,
                atol=1e-12,
                order=1,
                mode=skm,
                cval=0.1,
            )
        )
    return out
