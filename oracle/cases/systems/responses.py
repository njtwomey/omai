"""Time responses (lsim, dlsim, step, dstep, impulse, dimpulse, initial_response) against scipy.signal and control."""

from typing import cast

import control as ct
import numpy as np
import scipy
import scipy.signal as sg
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from mdml_oracle.lti import schur_stable_system, stable_system

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "control": ct.__version__}

TOL: dict[str, float] = {"rtol": 1e-9, "atol": 1e-11}


def cases() -> list[Case]:
    g = rng(4161)
    out: list[Case] = []
    a, b, c, d = stable_system(g, 3, 2, 2)
    t = np.linspace(0.0, 5.0, 51)
    u = np.stack([np.sin(t), np.cos(2 * t) + 0.5 * t], axis=1)
    x0 = g.normal(size=3)
    for interp in [True, False]:
        _, y, x = sg.lsim((a, b, c, d), u, t, X0=x0, interp=interp)
        args = [a, b, c, d, u, t, x0]
        out.append(case(f"lsim/mimo/{interp}/y", "lsim", args, y, **TOL, interp=interp, part="y"))
        out.append(case(f"lsim/mimo/{interp}/x", "lsim", args, x, **TOL, interp=interp, part="x"))
    # A transfer function (scipy realises it by tf2ss), starting after t = 0.
    num, den = [1.0, 2.0], [1.0, 3.0, 2.0]
    ts = np.linspace(0.5, 4.0, 36)
    _, y, x = sg.lsim((num, den), np.sin(ts), ts, X0=[0.3, -0.2])
    tf_args = [np.array(num), np.array(den), np.sin(ts), ts, np.array([0.3, -0.2])]
    out.append(case("lsim/tf/late-start/y", "lsim-tf", tf_args, y, **TOL, part="y"))
    out.append(case("lsim/tf/late-start/x", "lsim-tf", tf_args, x, **TOL, part="x"))

    da, db, dc, dd = schur_stable_system(g, 3, 2, 1)
    du = g.normal(size=(12, 2))
    dx0 = g.normal(size=3)
    _, y, x = sg.dlsim((da, db, dc, dd, 0.1), du, x0=dx0)
    dargs = [da, db, dc, dd, du, dx0]
    out.append(case("dlsim/y", "dlsim", dargs, y, **TOL, dt=0.1, part="y"))
    out.append(case("dlsim/x", "dlsim", dargs, x, **TOL, dt=0.1, part="x"))
    # Given times off the sample grid: the input is interpolated onto it.
    tt = np.linspace(0.0, 1.3, 7)
    ut = np.stack([np.sin(tt), tt**2], axis=1)
    _, y, x = sg.dlsim((da, db, dc, dd, 0.1), ut, t=tt)
    out.append(case("dlsim/interpolated/y", "dlsim-t", [da, db, dc, dd, ut, tt], y, **TOL, dt=0.1, part="y"))

    systems = {"oscillator": ([1.0], [1.0, 0.4, 1.0]), "zero": ([2.0, 1.0], [1.0, 3.0, 2.5])}
    for name, (num, den) in systems.items():
        tt, y = sg.step((num, den))
        out.append(case(f"step/{name}/t", "step-t", [np.array(num), np.array(den)], tt, rtol=1e-12))
        out.append(case(f"step/{name}/y", "step", [np.array(num), np.array(den)], y, **TOL))
        _, y = sg.impulse((num, den), N=40)
        out.append(case(f"impulse/{name}/y", "impulse", [np.array(num), np.array(den)], y, **TOL, n=40))
    sa, sb, sc, sd = stable_system(g, 2, 1, 1)
    sx0 = g.normal(size=2)
    tt = np.linspace(0, 3, 31)
    _, y = sg.step((sa, sb, sc, sd), X0=sx0, T=tt)
    out.append(case("step/ss/x0", "step-ss", [sa, sb, sc, sd, sx0, tt], y, **TOL))
    # scipy adds X0 to the column B, so pass it as a column too.
    _, y = sg.impulse((sa, sb, sc, sd), X0=sx0[:, None], T=tt)
    out.append(case("impulse/ss/x0", "impulse-ss", [sa, sb, sc, sd, sx0, tt], y, **TOL))

    for inp in range(2):
        _, ys = sg.dstep((da, db, dc, dd, 0.1), n=20)
        out.append(case(f"dstep/input{inp}", "dstep", [da, db, dc, dd], ys[inp], **TOL, dt=0.1, n=20, input=inp))
        _, ys = sg.dimpulse((da, db, dc, dd, 0.1), n=20)
        out.append(case(f"dimpulse/input{inp}", "dimpulse", [da, db, dc, dd], ys[inp], **TOL, dt=0.1, n=20, input=inp))

    ia, ib, ic, id_ = stable_system(g, 3, 1, 2)
    ix0 = g.normal(size=3)
    tt = np.linspace(0, 2, 21)
    r = cast(ct.TimeResponseData, ct.initial_response(ct.ss(ia, ib, ic, id_), T=tt, X0=ix0))
    out.append(case("initial/continuous", "initial", [ia, ib, ic, id_, ix0, tt], np.asarray(r.outputs).T, **TOL))
    r = cast(ct.TimeResponseData, ct.initial_response(ct.ss(da, db, dc, dd, 0.1), T=np.arange(15) * 0.1, X0=dx0))
    out.append(
        case("initial/discrete", "initial-d", [da, db, dc, dd, dx0], np.asarray(r.outputs).ravel(), **TOL, dt=0.1, n=15)
    )
    return out
