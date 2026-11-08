"""Knowledge-graph scoring functions and filtered ranking against numpy and scipy: TransE (L1 and L2 distance),
DistMult (a trilinear sum), ComplEx (the real part of the Hermitian product, in numpy complex arithmetic) and RotatE
(the modulus of the rotated head minus the tail, with the relation a phase), at random embeddings; and the filtered rank
of a true entity among scored candidates with ties sharing the average rank (scipy.stats.rankdata)."""

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from scipy import __version__ as scipy_version
from scipy.stats import rankdata

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy_version}


def scores(model: str, norm: int, e: FloatArray, r: FloatArray, triples: list[tuple[int, int, int]]) -> FloatArray:
    out: list[float] = []
    for h, rel, t in triples:
        if model == "transe":
            d = e[h] + r[rel] - e[t]
            out.append(float(-np.sum(np.abs(d)) if norm == 1 else -np.sqrt(np.sum(d**2) + 1e-12)))
        elif model == "distmult":
            out.append(float(np.sum(e[h] * r[rel] * e[t])))
        else:
            d = e.shape[1] // 2
            hc = e[h, :d] + 1j * e[h, d:]
            tc = e[t, :d] + 1j * e[t, d:]
            if model == "complex":
                rc = r[rel, :d] + 1j * r[rel, d:]
                out.append(float(np.real(np.sum(hc * rc * np.conj(tc)))))
            else:
                rot = np.exp(1j * r[rel])
                out.append(float(-np.sum(np.sqrt(np.abs(hc * rot - tc) ** 2 + 1e-12))))
    return np.asarray(out)


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(11)
    triples = [(0, 0, 1), (2, 1, 3), (4, 2, 0), (1, 0, 4), (3, 2, 2), (4, 1, 1)]
    for model, norm in [("transe", 1), ("transe", 2), ("distmult", 1), ("complex", 1), ("rotate", 1)]:
        d = 4
        width = 2 * d if model in ("complex", "rotate") else d
        rwidth = 2 * d if model == "complex" else d
        e = g.normal(size=(5, width))
        r = g.normal(size=(3, rwidth))
        out.append(
            case(
                f"score/{model}/{norm}",
                "kgScore",
                [e, r, np.asarray(triples, dtype=np.int64)],
                scores(model, norm, e, r, triples),
                rtol=1e-10,
                atol=1e-12,
                model=model,
                norm=norm,
            )
        )
    for seed in range(4):
        gg = rng(100 + seed)
        n = 12
        s = np.round(gg.normal(size=n), 1)  # rounding creates ties
        true = int(gg.integers(0, n))
        known = sorted({int(x) for x in gg.integers(0, n, size=3)} - {true})
        keep = [i for i in range(n) if i == true or i not in known]
        rank = float(rankdata(-s[keep], method="average")[keep.index(true)])
        out.append(case(f"filtered-rank/{seed}", "filteredRank", [s], rank, true=true, known=known))
    return out
