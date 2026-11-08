"""FIR design against scipy.signal: firwin, firwin2, remez, kaiserord and minimum_phase."""

import numpy as np
import scipy
import scipy.signal as S
from omai_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

TOL = {"rtol": 1e-10, "atol": 1e-13}


def cases() -> list[Case]:
    out: list[Case] = []
    out.append(case("firwin/low", "firwin", [], S.firwin(21, 0.3), **TOL, numtaps=21, cutoff=0.3))
    hp = S.firwin(21, 0.3, pass_zero=False)
    out.append(case("firwin/high", "firwin", [], hp, **TOL, numtaps=21, cutoff=0.3, passZero=False))
    bp = S.firwin(31, [0.2, 0.5], pass_zero="bandpass", window="blackman")
    out.append(
        case(
            "firwin/band",
            "firwin",
            [],
            bp,
            **TOL,
            numtaps=31,
            cutoff=[0.2, 0.5],
            passZero="bandpass",
            window="blackman",
        )
    )
    bs = S.firwin(31, [0.2, 0.5, 0.7], fs=4.0, scale=False)
    out.append(case("firwin/multi", "firwin", [], bs, **TOL, numtaps=31, cutoff=[0.2, 0.5, 0.7], fs=4.0, scale=False))
    kw = S.firwin(25, 100.0, width=40.0, fs=1000.0)
    out.append(case("firwin/kaiser", "firwin", [], kw, **TOL, numtaps=25, cutoff=100.0, width=40.0, fs=1000.0))
    nt, beta = S.kaiserord(60.0, 0.05)
    out.append(case("kaiserord", "kaiserord", [], np.array([nt, beta]), **TOL, ripple=60.0, width=0.05))
    out.append(case("kaiserBeta", "kaiserBeta", [], np.array([S.kaiser_beta(a) for a in [10.0, 30.0, 70.0]]), **TOL))
    out.append(case("kaiserAtten", "kaiserAtten", [], np.array(S.kaiser_atten(31, 0.1)), **TOL))
    f2 = S.firwin2(31, [0.0, 0.3, 0.3, 0.6, 1.0], [1.0, 1.0, 0.5, 0.2, 0.0])
    out.append(
        case(
            "firwin2/steps",
            "firwin2",
            [],
            f2,
            **TOL,
            numtaps=31,
            freq=[0.0, 0.3, 0.3, 0.6, 1.0],
            gain=[1.0, 1.0, 0.5, 0.2, 0.0],
        )
    )
    f2e = S.firwin2(20, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0], window=None)
    out.append(
        case(
            "firwin2/even",
            "firwin2",
            [],
            f2e,
            **TOL,
            numtaps=20,
            freq=[0.0, 0.5, 1.0],
            gain=[1.0, 1.0, 0.0],
            window=None,
        )
    )
    f3 = S.firwin2(21, [0.0, 2.5, 5.0], [0.0, 1.0, 0.0], antisymmetric=True, fs=10.0)
    out.append(
        case(
            "firwin2/type3",
            "firwin2",
            [],
            f3,
            **TOL,
            numtaps=21,
            freq=[0.0, 2.5, 5.0],
            gain=[0.0, 1.0, 0.0],
            antisymmetric=True,
            fs=10.0,
        )
    )
    f4 = S.firwin2(16, [0.0, 0.5, 1.0], [0.0, 0.5, 1.0], antisymmetric=True, nfreqs=65)
    out.append(
        case(
            "firwin2/type4",
            "firwin2",
            [],
            f4,
            **TOL,
            numtaps=16,
            freq=[0.0, 0.5, 1.0],
            gain=[0.0, 0.5, 1.0],
            antisymmetric=True,
            nfreqs=65,
        )
    )
    rt = {"rtol": 1e-8, "atol": 1e-10}
    for nt in [21, 22]:
        r = S.remez(nt, [0, 0.1, 0.2, 0.5], [1, 0])
        out.append(case(f"remez/low/{nt}", "remez", [], r, **rt, numtaps=nt, bands=[0, 0.1, 0.2, 0.5], desired=[1, 0]))
    bp = S.remez(41, [0, 50, 100, 200, 250, 500], [0, 1, 0], weight=[2, 1, 2], fs=1000.0)
    out.append(
        case(
            "remez/band",
            "remez",
            [],
            bp,
            **rt,
            numtaps=41,
            bands=[0, 50, 100, 200, 250, 500],
            desired=[0, 1, 0],
            weight=[2, 1, 2],
            fs=1000.0,
        )
    )
    for nt in [31, 30]:
        hb = S.remez(nt, [0.05, 0.45], [1], type="hilbert")
        out.append(
            case(
                f"remez/hilbert/{nt}",
                "remez",
                [],
                hb,
                **rt,
                numtaps=nt,
                bands=[0.05, 0.45],
                desired=[1],
                type="hilbert",
            )
        )
        df = S.remez(nt, [0, 0.4], [1], type="differentiator")
        out.append(
            case(
                f"remez/diff/{nt}",
                "remez",
                [],
                df,
                **rt,
                numtaps=nt,
                bands=[0, 0.4],
                desired=[1],
                type="differentiator",
            )
        )
    lin = S.remez(31, [0, 0.15, 0.25, 0.5], [1, 0])
    mp = S.minimum_phase(lin, method="homomorphic", n_fft=1024)
    out.append(case("minimumPhase/homomorphic", "minimumPhase", [lin], mp, rtol=1e-8, atol=1e-10, nFft=1024))
    # The Hilbert method takes logarithms of magnitudes floored at 1e-10 at the stopband zeros, where rounding in
    # the FFT moves the square roots by 1e-8; the reference itself is only stable to about 1e-2.
    mh = S.minimum_phase(lin, method="hilbert", n_fft=1024)
    out.append(
        case("minimumPhase/hilbert", "minimumPhase", [lin], mh, rtol=2e-2, atol=2e-3, method="hilbert", nFft=1024)
    )
    mp = S.minimum_phase(lin, half=False)
    out.append(
        case("minimumPhase/full", "minimumPhase", [lin], mp, rtol=1e-8, atol=1e-10, method="homomorphic", half=False)
    )
    return out
