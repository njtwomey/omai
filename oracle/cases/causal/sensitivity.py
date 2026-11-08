"""Rosenbaum bounds against scipy: the signed-rank bound at Gamma 1 is scipy's one-sided Wilcoxon test with the normal
approximation (asserted while generating) and at larger Gamma the same statistic with the tilted sign probability
Gamma / (1 + Gamma); the binary bounds are scipy's exact binomial tails at that probability."""

import numpy as np
import scipy
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from scipy.stats import binom, norm, rankdata, wilcoxon

REFERENCE: dict[str, str] = {"scipy": scipy.__version__}

GAMMAS = np.array([1.0, 1.25, 1.5, 2.0, 3.0])


def signed_rank_case(name: str, seed: int, n: int, shift: float, ties: bool) -> Case:
    g = rng(seed)
    control = g.normal(size=n)
    treated = control + shift + g.normal(size=n) * 0.8
    if ties:
        treated, control = np.round(treated, 1), np.round(control, 1)
    d = treated - control
    d = d[d != 0]
    ranks = rankdata(np.abs(d))
    t_plus = float(np.sum(ranks[d > 0]))

    def p(gamma: float, upper: bool) -> float:
        q = gamma / (1 + gamma) if upper else 1 / (1 + gamma)
        mu = q * float(np.sum(ranks))
        sd = float(np.sqrt(q * (1 - q) * np.sum(ranks**2)))
        return float(norm.sf((t_plus - mu) / sd))

    upper = np.array([p(float(x), True) for x in GAMMAS])
    lower = np.array([p(float(x), False) for x in GAMMAS])
    scipy_p = float(wilcoxon(treated, control, alternative="greater", method="approx", correction=False).pvalue)  # pyright: ignore[reportUnknownMemberType]
    assert abs(scipy_p - upper[0]) < 1e-12 * max(1.0, abs(scipy_p)) + 1e-14, (scipy_p, upper[0])
    return case(
        name,
        "signedRank",
        [treated, control],
        {"pUpper": upper, "pLower": lower},
        rtol=1e-8,
        atol=1e-14,
    )


def binary_case(name: str, seed: int, n: int, p_treated: float, p_control: float) -> Case:
    g = rng(seed)
    treated = (g.uniform(size=n) < p_treated).astype(np.float64)
    control = (g.uniform(size=n) < p_control).astype(np.float64)
    b = int(np.sum((treated == 1) & (control == 0)))
    m = b + int(np.sum((treated == 0) & (control == 1)))
    upper = np.array([float(binom.sf(b - 1, m, x / (1 + x))) for x in GAMMAS])
    lower = np.array([float(binom.sf(b - 1, m, 1 / (1 + x))) for x in GAMMAS])
    return case(name, "mcnemar", [treated, control], {"pUpper": upper, "pLower": lower}, rtol=1e-8, atol=1e-14)


def cases() -> list[Case]:
    out: list[Case] = [
        signed_rank_case("signed-rank/small", 401, 30, 0.6, False),
        signed_rank_case("signed-rank/large", 402, 400, 0.15, False),
        signed_rank_case("signed-rank/ties", 403, 200, 0.2, True),
        binary_case("mcnemar/a", 411, 200, 0.55, 0.4),
        binary_case("mcnemar/b", 412, 500, 0.35, 0.3),
    ]
    return out
