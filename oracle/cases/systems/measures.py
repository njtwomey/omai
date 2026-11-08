"""Step-response measures (control.step_info) and the Pade approximation of a delay (control.pade) against
python-control. step_info is given the same time grid as the TypeScript side, both for a system (final value from the DC
gain) and for sampled response data (final value the last sample).
"""

from typing import cast

import control as ct
import numpy as np
import scipy.signal as sg
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"control": ct.__version__}

KEYS: list[str] = [
    "RiseTime",
    "SettlingTime",
    "SettlingMin",
    "SettlingMax",
    "Overshoot",
    "Undershoot",
    "Peak",
    "PeakTime",
    "SteadyStateValue",
]

SYSTEMS: dict[str, tuple[list[float], list[float]]] = {
    "underdamped": ([4.0], [1.0, 0.8, 4.0]),
    "overdamped": ([2.0], [1.0, 3.0, 2.0]),
    "first-order": ([1.0], [1.0, 1.0]),
    "non-minimum-phase": ([-1.0, 1.0], [1.0, 3.0, 2.0]),
    "negative-gain": ([-3.0], [1.0, 2.0, 5.0]),
}


def measures(info: dict[str, float]) -> FloatArray:
    return np.array([info[k] for k in KEYS], dtype=np.float64)


def cases() -> list[Case]:
    out: list[Case] = []
    t = np.linspace(0.0, 20.0, 4001)
    for name, (num, den) in SYSTEMS.items():
        sys = ct.tf(num, den)
        info = cast(dict[str, float], ct.step_info(sys, T=t))
        out.append(
            case(
                f"step_info/{name}",
                "step-info",
                [np.array(num), np.array(den), t],
                measures(info),
                rtol=1e-9,
                atol=1e-9,
            )
        )
    tt = np.linspace(0.0, 10.0, 1001)
    for name in ["underdamped", "non-minimum-phase"]:
        num, den = SYSTEMS[name]
        _, y = sg.step((num, den), T=tt)
        y = np.asarray(y, dtype=np.float64)
        info = cast(dict[str, float], ct.step_info(y, T=tt))
        out.append(case(f"step_info/data/{name}", "step-data", [tt, y], measures(info), rtol=1e-9, atol=1e-9))
    # Other settling band and rise limits.
    num, den = SYSTEMS["underdamped"]
    info = cast(
        dict[str, float],
        ct.step_info(ct.tf(num, den), T=t, SettlingTimeThreshold=0.05, RiseTimeLimits=(0.05, 0.95)),
    )
    out.append(
        case(
            "step_info/options",
            "step-info",
            [np.array(num), np.array(den), t],
            measures(info),
            rtol=1e-9,
            atol=1e-9,
            band=0.05,
            low=0.05,
            high=0.95,
        )
    )
    for delay, n, numdeg in [(1.0, 3, None), (0.5, 1, None), (2.0, 4, None), (1.0, 3, -2), (0.1, 6, 3), (3.0, 2, 0)]:
        pn, pd = ct.pade(delay, n, numdeg)
        nd = -1 if numdeg is None else numdeg
        out.append(
            case(f"pade/{delay}/{n}/{nd}/num", "pade-num", [], np.array(pn), rtol=1e-12, delay=delay, n=n, numdeg=nd)
        )
        out.append(
            case(f"pade/{delay}/{n}/{nd}/den", "pade-den", [], np.array(pd), rtol=1e-12, delay=delay, n=n, numdeg=nd)
        )
    return out
