"""f-, Renyi, alpha, beta and Bregman divergences against scipy and sklearn (numpy formulas where none exists)."""

import numpy as np
import scipy
import scipy.integrate as si
import scipy.spatial.distance as ssd
import scipy.special as sp
import scipy.stats as st
import sklearn
from sklearn.decomposition._nmf import _beta_divergence  # pyright: ignore[reportPrivateUsage]
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "scikit-learn": sklearn.__version__}


def beta_div(p: FloatArray, q: FloatArray, beta: float) -> float:
    """sklearn's beta-divergence of NMF, with X = p, W = [[1]] and H = q."""
    return float(_beta_divergence(p[None, :], np.ones((1, 1)), q[None, :], beta))


def renyi_divergence(p: FloatArray, q: FloatArray, alpha: float) -> float:
    if alpha == np.inf:
        s = p > 0
        return float(np.log(np.max(p[s] / q[s])))
    return float(np.log(np.sum(p**alpha * q ** (1 - alpha))) / (alpha - 1))


def renyi_entropy(p: FloatArray, alpha: float) -> float:
    if alpha == np.inf:
        return float(-np.log(np.max(p)))
    if alpha == 0:
        return float(np.log(np.count_nonzero(p)))
    return float(np.log(np.sum(p[p > 0] ** alpha)) / (1 - alpha))


def alpha_divergence(p: FloatArray, q: FloatArray, alpha: float) -> float:
    return float(np.sum(p**alpha * q ** (1 - alpha) - alpha * p + (alpha - 1) * q) / (alpha * (alpha - 1)))


def cases() -> list[Case]:
    g = rng(410)
    out: list[Case] = []
    p = g.uniform(0.05, 1, 7)
    q = g.uniform(0.05, 1, 7)
    p, q = p / p.sum(), q / q.sum()
    f_refs: dict[str, float] = {
        "kl": float(np.sum(sp.rel_entr(p, q))),
        "reverseKl": float(np.sum(sp.rel_entr(q, p))),
        "totalVariation": float(0.5 * ssd.cityblock(p, q)),
        "squaredHellinger": float(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)),
        "pearson": float(st.chisquare(p, q).statistic),
        "neyman": float(st.chisquare(q, p).statistic),
        "jensenShannon": float(ssd.jensenshannon(p, q) ** 2),
        "triangular": float(np.sum((p - q) ** 2 / (p + q))),
    }
    for name, value in f_refs.items():
        out.append(case(f"fDivergence/{name}", "fDivergence", [p, q], np.array(value), f=name))
    out.append(case("fDivergence/kl/base=2", "fDivergence", [p, q], np.array(f_refs["kl"] / np.log(2)), f="kl", base=2))
    # Densities on a grid: the trapezoidal rule of scipy.integrate.
    x = np.linspace(-6, 6, 241)
    dp = st.norm.pdf(x, 0.0, 1.0)
    dq = st.norm.pdf(x, 0.5, 1.3)
    kl_grid = si.trapezoid(sp.rel_entr(dp, dq), x)
    out.append(case("fDivergence/kl/grid-dx", "fDivergence", [dp, dq], np.array(kl_grid), f="kl", dx=0.05))
    out.append(case("fDivergence/kl/grid-x", "fDivergence", [dp, dq, x], np.array(kl_grid), f="kl"))
    tv_grid = si.trapezoid(0.5 * np.abs(dp - dq), x)
    out.append(
        case("fDivergence/totalVariation/grid", "fDivergence", [dp, dq], np.array(tv_grid), f="totalVariation", dx=0.05)
    )
    pz = p.copy()
    pz[0] = 0.0
    pz = pz / pz.sum()
    for alpha in [0.0, 0.5, 2.0, 3.0, np.inf]:
        tag = "inf" if alpha == np.inf else str(alpha)
        a = "inf" if alpha == np.inf else alpha
        out.append(
            case(
                f"renyiDivergence/{tag}", "renyiDivergence", [pz, q], np.array(renyi_divergence(pz, q, alpha)), alpha=a
            )
        )
        out.append(case(f"renyiEntropy/{tag}", "renyiEntropy", [pz], np.array(renyi_entropy(pz, alpha)), alpha=a))
    out.append(case("renyiDivergence/1", "renyiDivergence", [pz, q], np.array(np.sum(sp.rel_entr(pz, q))), alpha=1.0))
    out.append(case("renyiEntropy/1", "renyiEntropy", [pz], np.array(st.entropy(pz)), alpha=1.0))
    out.append(
        case(
            "renyiDivergence/2/base=2",
            "renyiDivergence",
            [pz, q],
            np.array(renyi_divergence(pz, q, 2.0) / np.log(2)),
            alpha=2.0,
            base=2,
        )
    )
    # Unnormalised positive measures for the alpha- and beta-divergences.
    m1 = g.uniform(0.1, 3, 8)
    m2 = g.uniform(0.1, 3, 8)
    for alpha in [-1.0, 0.5, 2.0, 3.5]:
        out.append(
            case(
                f"alphaDivergence/{alpha}",
                "alphaDivergence",
                [m1, m2],
                np.array(alpha_divergence(m1, m2, alpha)),
                alpha=alpha,
            )
        )
    out.append(case("alphaDivergence/1", "alphaDivergence", [m1, m2], np.array(np.sum(sp.kl_div(m1, m2))), alpha=1.0))
    out.append(case("alphaDivergence/0", "alphaDivergence", [m1, m2], np.array(np.sum(sp.kl_div(m2, m1))), alpha=0.0))
    for beta in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        out.append(
            case(f"betaDivergence/{beta}", "betaDivergence", [m1, m2], np.array(beta_div(m1, m2, beta)), beta=beta)
        )
    out.append(
        case(
            "bregmanDivergence/squaredEuclidean",
            "bregmanDivergence",
            [m1, m2],
            np.array(ssd.sqeuclidean(m1, m2)),
            phi="squaredEuclidean",
        )
    )
    out.append(
        case(
            "bregmanDivergence/negativeEntropy",
            "bregmanDivergence",
            [m1, m2],
            np.array(np.sum(sp.kl_div(m1, m2))),
            phi="negativeEntropy",
        )
    )
    out.append(
        case(
            "bregmanDivergence/itakuraSaito",
            "bregmanDivergence",
            [m1, m2],
            np.array(beta_div(m1, m2, 0.0)),
            phi="itakuraSaito",
        )
    )
    return out
