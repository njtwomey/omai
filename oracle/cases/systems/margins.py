"""Gain, phase, stability and delay margins against python-control (stability_margins, margin,
phase_crossover_frequencies)."""

import control as ct
import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"control": ct.__version__}

LOOPS: list[tuple[str, list[float], list[float], float]] = [
    ("third-order", [6.0], [1.0, 6.0, 11.0, 6.0], 0.0),
    ("integrator", [2.0], [1.0, 3.0, 2.0, 0.0], 0.0),
    ("lead", [10.0, 10.0], [1.0, 6.0, 5.0, 0.0], 0.0),
    ("resonant", [1.0, 0.2], [1.0, 0.3, 2.0, 0.4], 0.0),
    ("unstable-loop", [20.0], [1.0, 6.0, 11.0, 6.0], 0.0),
    ("discrete", [0.1, 0.05], [1.0, -1.4, 0.45], 0.1),
    ("discrete-third", [0.02, 0.03], [1.0, -2.2, 1.6, -0.38], 0.5),
]


def finite(x: FloatArray) -> FloatArray:
    """Infinite and NaN values as large sentinels the fixture format keeps as strings anyway."""
    return np.asarray(x, dtype=np.float64)


def cases() -> list[Case]:
    out: list[Case] = []
    for name, num, den, dt in LOOPS:
        sys = ct.tf(num, den, dt)
        args = [np.array(num), np.array(den)]
        gm, pm, sm, wpc, wgc, wms = ct.stability_margins(sys)
        summary = np.array([gm, pm, sm, wpc, wgc, wms], dtype=np.float64)
        out.append(case(f"stabilityMargins/{name}", "stabilityMargins", args, summary, rtol=1e-7, atol=1e-9, dt=dt))
        gms, pms, sms, wpcs, wgcs, wmss = ct.stability_margins(sys, returnall=True)
        for field, value in [("gm", gms), ("pm", pms), ("wpc", wpcs), ("wgc", wgcs), ("sm", sms), ("wms", wmss)]:
            out.append(
                case(
                    f"allMargins/{name}/{field}",
                    "allMargins",
                    args,
                    finite(np.asarray(value)),
                    rtol=1e-7,
                    atol=1e-9,
                    dt=dt,
                    field=field,
                )
            )
        # The delay margin at each gain crossover, from python-control's phase margins: (pm mod 360) in radians / wgc.
        dms = np.mod(np.asarray(pms), 360.0) * np.pi / 180 / np.asarray(wgcs)
        out.append(
            case(f"allMargins/{name}/dm", "allMargins", args, finite(dms), rtol=1e-7, atol=1e-9, dt=dt, field="dm")
        )
        gm2, pm2, wcg, wcp = ct.margin(sys)
        out.append(
            case(
                f"margin/{name}",
                "margin",
                args,
                np.array([gm2, pm2, wcg, wcp], dtype=np.float64),
                rtol=1e-7,
                atol=1e-9,
                dt=dt,
            )
        )
        omega, gains = ct.phase_crossover_frequencies(sys)
        out.append(
            case(
                f"phaseCrossover/{name}/omega",
                "phaseCrossover",
                args,
                finite(np.sort(np.asarray(omega, dtype=np.float64))),
                rtol=1e-8,
                atol=1e-10,
                dt=dt,
                part="omega",
            )
        )
        out.append(
            case(
                f"phaseCrossover/{name}/gains",
                "phaseCrossover",
                args,
                finite(np.asarray(gains, dtype=np.float64)[np.argsort(omega)]),
                rtol=1e-8,
                atol=1e-10,
                dt=dt,
                part="gains",
            )
        )
    return out
