"""Bayes factors against pingouin; the Bayesian A/B comparison against its Beta-posterior integrals by scipy.integrate
(no library computes it), and the highest-density interval against its definition."""

import numpy as np
import pingouin as pg
import scipy
import scipy.stats as st
from scipy.integrate import quad
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"pingouin": pg.__version__, "scipy": scipy.__version__}


def ab(ka: int, na: int, kb: int, nb: int) -> list[float]:
    a = st.beta(1 + ka, 1 + na - ka)
    b = st.beta(1 + kb, 1 + nb - kb)
    p, _ = quad(lambda x: a.pdf(x) * b.sf(x), 0, 1, epsabs=0, epsrel=1e-12)
    loss_b, _ = quad(lambda y: b.pdf(y) * quad(lambda x: (x - y) * a.pdf(x), y, 1)[0], 0, 1, epsabs=0, epsrel=1e-10)
    return [float(p), float(loss_b)]


def hdi(draws: FloatArray, level: float) -> list[float]:
    s = np.sort(draws)
    k = max(1, int(np.floor(level * s.size)))
    widths = s[k - 1 :] - s[: s.size - k + 1]
    i = int(np.argmin(widths))
    return [float(s[i]), float(s[i + k - 1])]


def cases() -> list[Case]:
    out: list[Case] = []
    for t, nx, ny, paired in [
        (2.5, 20, None, False),
        (-1.2, 15, 18, False),
        (0.4, 30, 30, True),
        (4.1, 12, None, False),
    ]:
        bf = pg.bayesfactor_ttest(t, nx, ny, paired=paired)
        out.append(
            case(
                f"ttest/{t}/{nx}/{ny}/{paired}",
                "ttest",
                [],
                np.asarray(bf),
                rtol=1e-8,
                t=t,
                nx=nx,
                ny=ny or 0,
                paired=paired,
            )
        )
    for k, n, p in [(115, 200, 0.5), (3, 20, 0.3), (40, 50, 0.7)]:
        out.append(
            case(
                f"binomial/{k}/{n}",
                "binomial",
                [],
                np.asarray(pg.bayesfactor_binom(k, n, p)),
                rtol=1e-10,
                k=k,
                n=n,
                p=p,
            )
        )
    # pingouin's one-sided Ly factors need mpmath, which the oracle does not use; the two-sided ones are closed form.
    for r, n in [(0.3, 40), (-0.5, 15), (0.7, 25)]:
        bf = pg.bayesfactor_pearson(r, n)
        out.append(case(f"pearson/{r}/{n}", "pearson", [], np.asarray(bf), rtol=1e-9, r=r, n=n))
        bf = pg.bayesfactor_pearson(r, n, method="wetzels")
        out.append(
            case(f"pearson/{r}/{n}/wetzels", "pearson", [], np.asarray(bf), rtol=1e-8, r=r, n=n, method="wetzels")
        )
    out.append(case("abTest", "abTest", [], np.asarray(ab(20, 100, 30, 100)), rtol=1e-8))
    draws = rng(4).gamma(2.0, 1.0, size=500)
    out.append(case("hdi", "hdi", [draws, draws], np.asarray(hdi(draws, 0.9))))
    return out
