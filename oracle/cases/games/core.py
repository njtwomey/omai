"""The core and the nucleolus. The nucleolus of a bankruptcy game is the Talmud rule (Aumann and Maschler 1985):
constrained equal awards on half the claims when the estate is at most half the total claim, and constrained equal
losses on half the claims otherwise, computed here in closed form (the three estates of the Talmud's marriage
contract, 100, 200 and 300 against claims 100, 200 and 300, give 33 1/3 each, (50, 75, 75) and (50, 100, 150)), plus
textbook games (the glove game, whose core is a single point, and the majority game, whose core is empty). The least
core's epsilon comes from scipy.optimize.linprog for the same games and random ones."""

import numpy as np
import scipy
import scipy.optimize as so
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "definition": "Aumann and Maschler (1985), the Talmud rule"}


def equal_awards(caps: list[float], amount: float) -> list[float]:
    """Constrained equal awards: min(cap_i, lam) summing to `amount` (0 <= amount <= sum of caps)."""
    lo, hi = 0.0, max(caps)
    for _ in range(200):
        lam = (lo + hi) / 2
        if sum(min(c, lam) for c in caps) < amount:
            lo = lam
        else:
            hi = lam
    lam = (lo + hi) / 2
    return [min(c, lam) for c in caps]


def talmud(estate: float, claims: list[float]) -> list[float]:
    """The Talmud rule."""
    half = [c / 2 for c in claims]
    total = sum(claims)
    if estate <= total / 2:
        return equal_awards(half, estate)
    losses = equal_awards(half, total - estate)
    return [c - loss for c, loss in zip(claims, losses, strict=True)]


def bankruptcy_worths(estate: float, claims: list[float]) -> FloatArray:
    """The bankruptcy game's worths by bitmask: max(0, E - the others' claims)."""
    n = len(claims)
    total = sum(claims)
    out = np.zeros(2**n)
    for mask in range(1, 2**n):
        own = sum(c for i, c in enumerate(claims) if mask >> i & 1)
        out[mask] = max(0.0, estate - (total - own))
    return out


def least_core(n: int, v: FloatArray) -> float:
    """min eps subject to x(S) >= v(S) - eps for every proper coalition, x(N) = v(N)."""
    full = 2**n - 1
    a_ub = [[-(mask >> i & 1) for i in range(n)] + [-1] for mask in range(1, full)]
    b_ub = [-v[mask] for mask in range(1, full)]
    c = [0.0] * n + [1.0]
    r = so.linprog(
        c,
        A_ub=np.array(a_ub, dtype=np.float64),
        b_ub=np.array(b_ub),
        A_eq=np.array([[1.0] * n + [0.0]]),
        b_eq=np.array([v[full]]),
        bounds=[(None, None)] * (n + 1),
    )
    return float(np.asarray(r.x, dtype=np.float64)[n])


def cases() -> list[Case]:
    out: list[Case] = []
    bankruptcies: list[tuple[float, list[float]]] = [
        (100.0, [100.0, 200.0, 300.0]),
        (200.0, [100.0, 200.0, 300.0]),
        (300.0, [100.0, 200.0, 300.0]),
    ]
    for seed, n in enumerate([3, 4, 5, 4]):
        g = rng(1100 + seed)
        claims = [float(c) for c in g.uniform(1, 10, size=n)]
        bankruptcies.append((float(g.uniform(0.1, 0.9) * sum(claims)), claims))
    for estate, claims in bankruptcies:
        n = len(claims)
        v = bankruptcy_worths(estate, claims)
        out.append(
            case(
                f"bankruptcy/{estate:.4g}/{'-'.join(f'{c:.3g}' for c in claims)}",
                "nucleolus",
                [v],
                np.array(talmud(estate, claims)),
                n=n,
                leastCore=least_core(n, v),
                atol=1e-9,
            )
        )
    glove = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    majority = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    for name, f, nuc in (("glove-1-2", glove, [1.0, 0.0, 0.0]), ("majority-3", majority, [1 / 3, 1 / 3, 1 / 3])):
        v = np.array(f)
        out.append(case(name, "nucleolus", [v], np.array(nuc), n=3, leastCore=least_core(3, v), atol=1e-9))
    for seed, n in enumerate([3, 4, 5]):
        v = np.concatenate([[0.0], rng(1200 + seed).uniform(0, 1, size=2**n - 1)])
        out.append(case(f"random-{n}", "leastCore", [v], np.array(least_core(n, v)), n=n, atol=1e-8))
    return out
