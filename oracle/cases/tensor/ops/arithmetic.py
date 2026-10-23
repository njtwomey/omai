"""Elementwise arithmetic with broadcasting and dtype promotion, against numpy."""

import numpy as np
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import BINARY_SHAPES, Binary, FloatArray, Unary, rng, uniform

REFERENCE: dict[str, str] = {"numpy": np.__version__}


def py_mod(a: FloatArray, b: FloatArray) -> FloatArray:
    return np.mod(a, b)


BINARY: dict[str, tuple[Binary, tuple[float, float], tuple[float, float]]] = {
    "add": (np.add, (-3, 3), (-3, 3)),
    "sub": (np.subtract, (-3, 3), (-3, 3)),
    "mul": (np.multiply, (-3, 3), (-3, 3)),
    "div": (np.divide, (-3, 3), (0.5, 3)),
    "pow": (np.power, (0.1, 3), (-2, 2)),
    "maximum": (np.maximum, (-3, 3), (-3, 3)),
    "minimum": (np.minimum, (-3, 3), (-3, 3)),
    "mod": (py_mod, (-5, 5), (0.5, 3)),
    "floorDiv": (np.floor_divide, (-5, 5), (0.5, 3)),
    "arctan2": (np.arctan2, (-3, 3), (-3, 3)),
}


def cases() -> list[Case]:
    out: list[Case] = []
    for i, (name, (fn, (alo, ahi), (blo, bhi))) in enumerate(BINARY.items()):
        for sa, sb in BINARY_SHAPES:
            g = rng(200 + i)
            a = uniform(g, sa, alo, ahi)
            b = uniform(g, sb, blo, bhi)
            out.append(case(f"{name}/{list(sa)}x{list(sb)}", name, [a, b], fn(a, b), rtol=1e-13, atol=1e-15))
    unary: list[tuple[str, Unary]] = [("neg", np.negative), ("abs", np.abs), ("sign", np.sign), ("square", np.square)]
    for name, fn in unary:
        x = uniform(rng(300), (2, 3))
        out.append(case(f"{name}/[2, 3]", name, [x], fn(x)))
    ints = np.array([7, -7, 3], dtype=np.int32)
    out.append(
        case("mod/int32", "mod", [ints, np.array([3, 3, -2], dtype=np.int32)], np.mod(ints, np.array([3, 3, -2])))
    )
    out.append(case("div/int32-true-division", "div", [ints, np.array([2, 2, 2], dtype=np.int32)], ints / 2))
    out.append(case("add/int32-plus-float64", "add", [ints, np.array([0.5, 0.5, 0.5])], ints + 0.5))
    special = np.array([np.inf, -np.inf, np.nan, 0.0])
    factor = np.array([2.0, 2.0, 2.0, np.inf])
    with np.errstate(invalid="ignore"):
        out.append(case("mul/non-finite", "mul", [special, factor], special * factor))
    return out
