"""Weak supervision against numpy references written from the papers (no oracle library implements them):
Dawid-Skene EM with full confusion matrices and with the one-coin model, from the majority-vote start for a fixed number
of iterations; and confident learning's thresholds, confident joint, calibrated joint, noise-transition anchors and the
prune-by-noise-rate / prune-by-class / confident-joint issue sets."""

import numpy as np
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__}


def votes_matrix(g: np.random.Generator, n: int, k: int, accuracies: list[float], coverage: float) -> FloatArray:
    """Votes `[n, m]` from sources of the given accuracies (errors uniform over the other classes); -1 abstains."""
    truth = g.integers(0, k, size=n)
    out = np.full((n, len(accuracies)), -1.0)
    for j, a in enumerate(accuracies):
        correct = g.uniform(size=n) < a
        wrong = (truth + g.integers(1, k, size=n)) % k
        vote = np.where(correct, truth, wrong)
        out[:, j] = np.where(g.uniform(size=n) < coverage, vote, -1)
    return out


def majority_shares(v: FloatArray, k: int) -> FloatArray:
    post = np.zeros((v.shape[0], k))
    for c in range(k):
        post[:, c] = (v == c).sum(axis=1)
    total = post.sum(axis=1, keepdims=True)
    return np.where(total > 0, post / np.maximum(total, 1), 1.0 / k)


def dawid_skene(v: FloatArray, k: int, mode: str, steps: int) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """EM from the majority-vote posterior: `steps + 1` M-steps, each followed by an E-step."""
    n, m = v.shape
    post = majority_shares(v, k)
    conf = np.zeros((m, k, k))
    priors = np.zeros(k)
    ll = 0.0
    for _ in range(steps + 1):
        priors = post.sum(axis=0) / n
        for j in range(m):
            if mode == "full":
                counts = np.zeros((k, k))
                for lab in range(k):
                    counts[:, lab] = post[v[:, j] == lab].sum(axis=0)
                z = counts.sum(axis=1, keepdims=True)
                conf[j] = np.where(z > 0, counts / np.maximum(z, 1e-300), 1.0 / k)
            else:
                voted = v[:, j] >= 0
                right = sum(post[i, int(v[i, j])] for i in np.flatnonzero(voted))
                acc = right / max(voted.sum(), 1)
                conf[j] = np.full((k, k), (1 - acc) / (k - 1))
                np.fill_diagonal(conf[j], acc)
        logp = np.tile(np.log(np.maximum(priors, 1e-300)), (n, 1))
        for j in range(m):
            for i in np.flatnonzero(v[:, j] >= 0):
                logp[i] += np.log(np.maximum(conf[j][:, int(v[i, j])], 1e-300))
        top = logp.max(axis=1, keepdims=True)
        z = top + np.log(np.exp(logp - top).sum(axis=1, keepdims=True))
        ll = float(z.sum())
        post = np.exp(logp - z)
    return conf, priors, post, ll


def confident_learning(p: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Thresholds, confident-joint counts, the calibrated joint and the per-item guesses (Northcutt et al. 2021)."""
    n, k = p.shape
    labels = y.astype(int)
    thr = np.array([p[labels == c, c].mean() for c in range(k)])
    counts = np.zeros((k, k))
    guess = np.full(n, -1.0)
    for i in range(n):
        above = p[i] >= thr - 1e-6
        if above.any():
            g = int(np.argmax(np.where(above, p[i], -np.inf)))
            guess[i] = g
            counts[labels[i], g] += 1
    sizes = np.bincount(labels, minlength=k).astype(float)
    cal = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-300) * sizes[:, None]
    cal = cal / cal.sum() * sizes.sum()
    return thr, counts, cal / n, guess


def prune_by_noise_rate(p: FloatArray, y: FloatArray, joint: FloatArray) -> list[int]:
    n, k = p.shape
    labels = y.astype(int)
    chosen: set[int] = set()
    cal = np.round(joint * n).astype(int)
    for g in range(k):
        rows = np.flatnonzero(labels == g)
        for t in range(k):
            if t == g:
                continue
            margin = p[rows, t] - p[rows, g]
            order = rows[np.lexsort((rows, -margin))]
            chosen.update(int(i) for i in order[: cal[g, t]])
    selfc = p[np.arange(n), labels]
    return [int(i) for i in sorted(chosen, key=lambda i: (selfc[i], i))]


def anchors(p: FloatArray, percentile: float) -> FloatArray:
    n, k = p.shape
    t = np.zeros((k, k))
    for c in range(k):
        order = np.lexsort((np.arange(n), p[:, c]))
        anchor = order[min(n - 1, max(0, int(np.ceil(percentile / 100 * n)) - 1))]
        row = p[anchor]
        t[c] = row / row.sum()
    return t


def cases() -> list[Case]:
    g = rng(31102026)
    out: list[Case] = []
    for name, k, accs in [("binary", 2, [0.9, 0.8, 0.7, 0.6]), ("three", 3, [0.85, 0.8, 0.7, 0.65, 0.6])]:
        v = votes_matrix(g, 400, k, accs, 0.8)
        for mode in ["full", "one-coin"]:
            conf, priors, post, ll = dawid_skene(v, k, mode, 30)
            out.append(
                case(
                    f"dawid-skene/{mode}/{name}",
                    "dawid_skene",
                    [v],
                    {
                        "confusions": conf.reshape(len(accs), -1),
                        "priors": priors,
                        "posterior": post,
                        "logLikelihood": np.array([ll]),
                    },
                    rtol=1e-8,
                    atol=1e-9,
                    mode=mode,
                    iterations=30,
                    classes=k,
                )
            )
    for name, k in [("binary", 2), ("three", 3), ("four", 4)]:
        n = 500
        y = g.integers(0, k, size=n).astype(np.float64)
        logits = g.normal(size=(n, k)) * 1.5
        logits[np.arange(n), y.astype(int)] += 2.0
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        thr, counts, joint, guess = confident_learning(p, y)
        out.append(
            case(
                f"confident/{name}",
                "confident",
                [p, y],
                {
                    "thresholds": thr,
                    "counts": counts,
                    "joint": joint,
                    "guess": guess,
                    "issues": np.array(prune_by_noise_rate(p, y, joint), dtype=np.float64),
                    "anchor": anchors(p, 97.0),
                },
                rtol=1e-10,
                atol=1e-12,
            )
        )
    return out
