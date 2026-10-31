"""H-infinity synthesis at fixed levels against SLICOT's SB10FD (through slycot, the routine behind python-control's
hinfsyn): the central controller of the two-Riccati solution, compared by its frequency response (the state
coordinates of the two realisations differ by a similarity).
"""

import numpy as np
import slycot
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"slycot": str(getattr(slycot, "__version__", "0.7"))}


def plant(
    a: FloatArray, b1: FloatArray, b2: FloatArray, c1: FloatArray, c2: FloatArray, d12: FloatArray, d21: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The stacked B = [B1 B2], C = [C1; C2] and D of SB10FD."""
    nz, nw = c1.shape[0], b1.shape[1]
    nu, ny = b2.shape[1], c2.shape[0]
    b = np.hstack([b1, b2])
    c = np.vstack([c1, c2])
    d = np.zeros((nz + ny, nw + nu))
    d[:nz, nw:] = d12
    d[nz:, :nw] = d21
    return b, c, d


def response(ak: FloatArray, bk: FloatArray, ck: FloatArray, dk: FloatArray, w: FloatArray) -> FloatArray:
    """The controller's frequency response at jw, as stacked real and imaginary parts `[2, len(w)]`."""
    out = np.array(
        [(ck @ np.linalg.solve(1j * x * np.eye(ak.shape[0]) - ak, bk) + dk).item() for x in w], dtype=np.complex128
    )
    return np.stack([out.real, out.imag])


def cases() -> list[Case]:
    out: list[Case] = []
    w = np.array([0.0, 0.1, 0.5, 1.0, 3.0, 10.0])
    plants: dict[str, tuple[FloatArray, ...]] = {
        "scalar": (
            np.array([[1.0]]),
            np.array([[1.0, 0.0]]),
            np.array([[1.0]]),
            np.array([[1.0], [0.0]]),
            np.array([[1.0]]),
            np.array([[0.0], [1.0]]),
            np.array([[0.0, 1.0]]),
        ),
        "oscillator": (
            np.array([[0.0, 1.0], [-2.0, -0.2]]),
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([[0.0], [1.0]]),
            np.array([[1.0, 0.0], [0.0, 0.0]]),
            np.array([[1.0, 0.0]]),
            np.array([[0.0], [1.0]]),
            np.array([[0.0, 1.0]]),
        ),
    }
    levels = {"scalar": [3.0, 5.0, 40.0], "oscillator": [20.0, 50.0]}
    for name, (a, b1, b2, c1, c2, d12, d21) in plants.items():
        b, c, d = plant(a, b1, b2, c1, c2, d12, d21)
        n = a.shape[0]
        for gamma in levels[name]:
            ak, bk, ck, dk, _rcond = slycot.sb10fd(
                n, b.shape[1], c.shape[0], b2.shape[1], c2.shape[0], gamma, a, b, c, d
            )
            out.append(
                case(
                    f"sb10fd/{name}/{gamma}",
                    "hinf-controller",
                    [a, b1, b2, c1, c2, d12, d21, w],
                    response(ak, bk, ck, dk, w),
                    rtol=1e-6,
                    atol=1e-8,
                    gamma=gamma,
                )
            )
    return out
