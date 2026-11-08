"""Frequency responses (freqresp, dfreqresp, bode, dbode, findfreqs) against scipy.signal, and dcgain, evalfr, MIMO
responses and Nyquist encirclement counts against python-control."""

import control as ct
import numpy as np
import scipy
import scipy.signal as sg
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from omai_oracle.lti import parts, schur_stable_system, stable_system

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "control": ct.__version__}

TOL: dict[str, float] = {"rtol": 1e-10, "atol": 1e-12}


def cases() -> list[Case]:
    g = rng(4162)
    out: list[Case] = []
    third = ([1.0, 2.0], [1.0, 0.5, 1.0, 0.3])
    w = np.logspace(-1, 1, 9)
    for name, system in [("tf", third), ("zpk", sg.tf2zpk(*third)), ("ss", sg.tf2ss(*third))]:
        _, h = sg.freqresp(system, w=w)
        out.append(
            case(
                f"freqresp/{name}",
                "freqresp-tf",
                [np.array(third[0]), np.array(third[1]), w],
                parts(h),
                **TOL,
                form=name,
            )
        )
    _, h = sg.freqresp(third, n=20)
    wd, _ = sg.freqresp(third, n=20)
    out.append(case("freqresp/default-w", "freqresp-w", [np.array(third[0]), np.array(third[1])], wd, rtol=1e-12, n=20))
    out.append(
        case("freqresp/default-h", "freqresp-h", [np.array(third[0]), np.array(third[1])], parts(h), **TOL, n=20)
    )
    wb, mag, phase = sg.bode(third, n=15)
    out.append(case("bode/w", "bode", [np.array(third[0]), np.array(third[1])], wb, rtol=1e-12, n=15, part="w"))
    out.append(case("bode/mag", "bode", [np.array(third[0]), np.array(third[1])], mag, **TOL, n=15, part="mag"))
    out.append(case("bode/phase", "bode", [np.array(third[0]), np.array(third[1])], phase, **TOL, n=15, part="phase"))

    dnum, dden, dt = [0.2, 0.1], [1.0, -1.1, 0.5], 0.1
    ws = np.linspace(0.05, 3.0, 11)  # radians per sample
    _, h = sg.dfreqresp((dnum, dden, dt), w=ws)
    dargs = [np.array(dnum), np.array(dden), ws / dt]
    out.append(case("dfreqresp", "dfreqresp", dargs, parts(h), **TOL, dt=dt))
    wdb, mag, phase = sg.dbode((dnum, dden, dt), w=ws)
    out.append(case("dbode/mag", "dbode", dargs, mag, **TOL, dt=dt, part="mag"))
    out.append(case("dbode/phase", "dbode", dargs, phase, **TOL, dt=dt, part="phase"))
    out.append(case("dbode/w", "dbode", dargs, wdb, rtol=1e-12, dt=dt, part="w"))

    a, b, c, d = stable_system(g, 4, 2, 3)
    wm = np.logspace(-1, 1, 6)
    resp = ct.ss(a, b, c, d)(1j * wm)  # [outputs, inputs, nw]
    out.append(case("freqresp/mimo", "freqresp-ss", [a, b, c, d, wm], parts(np.moveaxis(resp, -1, 0).ravel()), **TOL))
    da, db, dc, dd = schur_stable_system(g, 3, 2, 2)
    resp = ct.ss(da, db, dc, dd, 0.5)(np.exp(1j * wm * 0.5))
    out.append(
        case(
            "freqresp/mimo-discrete",
            "freqresp-ss-d",
            [da, db, dc, dd, wm],
            parts(np.moveaxis(resp, -1, 0).ravel()),
            **TOL,
            dt=0.5,
        )
    )
    out.append(case("dcgain/mimo", "dcgain-ss", [a, b, c, d], ct.dcgain(ct.ss(a, b, c, d)), **TOL))
    out.append(
        case(
            "dcgain/mimo-discrete",
            "dcgain-ss-d",
            [da, db, dc, dd],
            ct.dcgain(ct.ss(da, db, dc, dd, 0.5)),
            **TOL,
            dt=0.5,
        )
    )
    out.append(
        case(
            "dcgain/tf",
            "dcgain-tf",
            [np.array([3.0, 1.0]), np.array([2.0, 1.0, 4.0])],
            np.array(ct.dcgain(ct.tf([3.0, 1.0], [2.0, 1.0, 4.0]))),
            **TOL,
        )
    )
    point = 0.3 + 0.7j
    value = ct.evalfr(ct.tf(*third), point)
    out.append(
        case(
            "evalfr/tf",
            "evalfr",
            [np.array(third[0]), np.array(third[1])],
            parts(value),
            **TOL,
            re=point.real,
            im=point.imag,
        )
    )

    butter_b, butter_a = sg.butter(3, 2.0, analog=True)
    filters: list[tuple[str, list[float], list[float]]] = [
        ("butter", [float(v) for v in butter_b], [float(v) for v in butter_a]),
        ("lead", [1.0, 0.5], [1.0, 5.0]),
        ("integrator", [1.0], [1.0, 1.0, 0.0]),
        ("resonant", [1.0, 0.0, 4.0], [1.0, 0.2, 9.0, 0.3]),
    ]
    for name, bb, aa in filters:
        out.append(
            case(
                f"findfreqs/{name}",
                "findfreqs",
                [np.array(bb), np.array(aa)],
                sg.findfreqs(bb, aa, 12),
                rtol=1e-12,
                n=12,
            )
        )

    loops: list[tuple[str, list[float], list[float], float]] = [
        ("stable", [2.0], [1.0, 1.0], 0.0),
        ("cubic-stable", [3.0], [1.0, 3.0, 3.0, 1.0], 0.0),
        ("cubic-unstable", [10.0], [1.0, 3.0, 3.0, 1.0], 0.0),
        ("open-loop-unstable", [3.0], [1.0, -1.0], 0.0),
        ("integrator", [1.0], [1.0, 1.0, 0.0], 0.0),
        ("discrete", [0.5], [1.0, -0.5], 0.1),
        ("discrete-unstable", [3.0, 0.0], [1.0, -1.2, 0.5], 1.0),
    ]
    for name, bb, aa, ldt in loops:
        response = ct.nyquist_response(ct.tf(bb, aa, ldt), warn_encirclements=False)
        out.append(case(f"nyquist/{name}", "nyquist", [np.array(bb), np.array(aa)], np.array(response.count), dt=ldt))
    return out
