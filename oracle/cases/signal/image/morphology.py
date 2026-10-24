"""Mathematical morphology against scipy.ndimage (binary and grey erosion, dilation, opening, closing with
structures, origins and iterations; gradients, top-hats, labelling, the Euclidean distance transform) and
scikit-image (thinning and skeletonisation)."""

from importlib.metadata import version

import numpy as np
import numpy.typing as npt
import scipy
import scipy.ndimage as N
import skimage.morphology as M
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng
from whitebox_oracle.signal_helpers import blocks_image, smooth_image

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "scikit-image": version("scikit-image")}

TOL = {"rtol": 1e-12, "atol": 1e-12}


def f(a: npt.ArrayLike) -> FloatArray:
    return np.asarray(a, dtype=np.float64)


def cases() -> list[Case]:
    g = rng(470)
    out: list[Case] = []
    b = (g.random((10, 12)) > 0.45).astype(float)
    b[3:7, 4:9] = 1
    cross = N.generate_binary_structure(2, 1).astype(float)
    full = N.generate_binary_structure(2, 2).astype(float)
    asym = np.array([[1, 1, 0], [0, 1, 1]], dtype=float)
    out.append(
        case(
            "generateBinaryStructure",
            "generateBinaryStructure",
            [],
            f(N.generate_binary_structure(3, 2)),
            rank=3,
            connectivity=2,
        )
    )
    for name, fn in [
        ("binaryErosion", N.binary_erosion),
        ("binaryDilation", N.binary_dilation),
        ("binaryOpening", N.binary_opening),
        ("binaryClosing", N.binary_closing),
    ]:
        out.append(case(f"{name}", name, [b], f(fn(b)), **TOL))
        out.append(case(f"{name}/full/2", name, [b, full], f(fn(b, full, iterations=2)), **TOL, iterations=2))
        out.append(case(f"{name}/asym", name, [b, asym], f(fn(b, asym)), **TOL))
    out.append(
        case(
            "binaryErosion/origin",
            "binaryErosion",
            [b, asym],
            f(N.binary_erosion(b, asym, origin=(0, -1))),
            **TOL,
            origin=[0, -1],
        )
    )
    out.append(
        case(
            "binaryDilation/origin",
            "binaryDilation",
            [b, asym],
            f(N.binary_dilation(b, asym, origin=(-1, 1))),
            **TOL,
            origin=[-1, 1],
        )
    )
    out.append(
        case("binaryErosion/border", "binaryErosion", [b], f(N.binary_erosion(b, border_value=1)), **TOL, borderValue=1)
    )
    out.append(
        case(
            "binaryDilation/until",
            "binaryDilation",
            [b[:5, :5] * 0 + np.eye(5)],
            f(N.binary_dilation(np.eye(5), iterations=0)),
            **TOL,
            iterations=0,
        )
    )
    img = smooth_image(g, 9, 10)
    for name, fn in [
        ("greyErosion", N.grey_erosion),
        ("greyDilation", N.grey_dilation),
        ("greyOpening", N.grey_opening),
        ("greyClosing", N.grey_closing),
        ("morphologicalGradient", N.morphological_gradient),
        ("whiteTophat", N.white_tophat),
        ("blackTophat", N.black_tophat),
    ]:
        out.append(case(f"{name}/size", name, [img], f(fn(img, size=(3, 2))), **TOL, size=[3, 2]))
        out.append(
            case(
                f"{name}/footprint",
                name,
                [img, asym],
                f(fn(img, footprint=asym, mode="nearest")),
                **TOL,
                mode="nearest",
            )
        )
    out.append(
        case(
            "greyDilation/origin",
            "greyDilation",
            [img, cross],
            f(N.grey_dilation(img, footprint=cross, origin=1, mode="constant", cval=0.3)),
            **TOL,
            origin=1,
            mode="constant",
            cval=0.3,
        )
    )
    labels, n = N.label(b)
    out.append(case("label", "label", [b], np.concatenate([f(labels).ravel(), [float(n)]]), **TOL))
    labels, n = N.label(b, full)
    out.append(case("label/full", "label", [b, full], np.concatenate([f(labels).ravel(), [float(n)]]), **TOL))
    out.append(case("distanceTransformEdt", "distanceTransformEdt", [b], f(N.distance_transform_edt(b)), **TOL))
    out.append(
        case(
            "distanceTransformEdt/sampling",
            "distanceTransformEdt",
            [b],
            f(N.distance_transform_edt(b, sampling=[1.5, 0.7])),
            **TOL,
            sampling=[1.5, 0.7],
        )
    )
    blocks = (blocks_image(16, 16) > 0).astype(float)
    out.append(case("thin", "thin", [blocks], f(M.thin(blocks)), **TOL))
    out.append(case("thin/2", "thin", [blocks], f(M.thin(blocks, max_num_iter=2)), **TOL, maxIter=2))
    # skeletonize is not compared: scikit-image's compiled lookup table departs from Zhang and Suen's published
    # conditions on some neighbourhoods (the engine follows the paper; its property tests check thinness and topology).
    return out
