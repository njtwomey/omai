"""Online linear learners against scikit-learn: the perceptron (plain and averaged, as SGD with the perceptron loss on
the data in order), passive-aggressive PA-I and PA-II classifiers and the epsilon-insensitive regressor (SGD with the
passive-aggressive learning rates), and the voted, pocket and kernel perceptrons formed from their definitions with
numpy loops."""

from typing import Any

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.linear_model import Perceptron, SGDClassifier, SGDRegressor
from sklearn.metrics.pairwise import rbf_kernel

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


# scikit-learn accepts None for these (no tolerance, no penalty); its stubs type them as float and str.
NONE: Any = None


def data(g: np.random.Generator, n: int, noise: float) -> tuple[FloatArray, FloatArray]:
    x = g.normal(size=(n, 3))
    y = (x @ np.array([1.0, -1.0, 0.5]) + noise * g.normal(size=n) > 0).astype(np.float64)
    return x, y


def perceptron_path(x: FloatArray, y: FloatArray, epochs: int) -> tuple[list[FloatArray], list[float], int]:
    """The weights and intercept after every example, and the mistakes, of the plain perceptron in data order."""
    ys = 2 * y - 1
    w = np.zeros(x.shape[1])
    b = 0.0
    ws: list[FloatArray] = []
    bs: list[float] = []
    mistakes = 0
    for _ in range(epochs):
        for i in range(len(ys)):
            if ys[i] * (x[i] @ w + b) <= 0:
                w = w + ys[i] * x[i]
                b += float(ys[i])
                mistakes += 1
            ws.append(w.copy())
            bs.append(b)
    return ws, bs, mistakes


def voted(x: FloatArray, y: FloatArray, epochs: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The vectors of the voted perceptron, their intercepts and survival counts (the zero vector dropped)."""
    ys = 2 * y - 1
    w = np.zeros(x.shape[1])
    b = 0.0
    vecs: list[FloatArray] = [w.copy()]
    bias: list[float] = [0.0]
    counts: list[int] = [0]
    for _ in range(epochs):
        for i in range(len(ys)):
            if ys[i] * (x[i] @ w + b) <= 0:
                w = w + ys[i] * x[i]
                b += float(ys[i])
                vecs.append(w.copy())
                bias.append(b)
                counts.append(1)
            else:
                counts[-1] += 1
    keep = [k for k, c in enumerate(counts) if c > 0]
    return (
        np.array([vecs[k] for k in keep]),
        np.array([bias[k] for k in keep]),
        np.array([counts[k] for k in keep], dtype=np.float64),
    )


def pocket(x: FloatArray, y: FloatArray, epochs: int) -> tuple[FloatArray, float]:
    """Gallant's pocket algorithm with ratchet, cycling through the data."""
    ys = 2 * y - 1
    n = len(ys)
    w = np.zeros(x.shape[1])
    b = 0.0
    run = 0
    pw = w.copy()
    pb = 0.0
    prun = 0
    perr = n
    for t in range(epochs * n):
        i = t % n
        if ys[i] * (x[i] @ w + b) > 0:
            run += 1
            if run > prun:
                errors = int(np.sum(ys * (x @ w + b) <= 0))
                if errors < perr:
                    pw, pb, prun, perr = w.copy(), b, run, errors
        else:
            w = w + ys[i] * x[i]
            b += float(ys[i])
            run = 0
        if perr == 0:
            break
    return pw, pb


def kernel_perceptron(x: FloatArray, y: FloatArray, gamma: float, epochs: int) -> tuple[FloatArray, float, int]:
    ys = 2 * y - 1
    n = len(ys)
    k = rbf_kernel(x, gamma=gamma)
    alpha = np.zeros(n)
    b = 0.0
    mistakes = 0
    for t in range(epochs * n):
        i = t % n
        score = (alpha * ys) @ k[:, i] + b
        if ys[i] * score <= 0:
            alpha[i] += 1
            b += float(ys[i])
            mistakes += 1
    return alpha, b, mistakes


def cases() -> list[Case]:
    g = rng(1969)
    out: list[Case] = []
    x, y = data(g, 40, 0.3)
    q = g.normal(size=(10, 3))
    for epochs in (1, 2, 4):
        p = Perceptron(shuffle=False, max_iter=epochs, tol=NONE, eta0=1.0).fit(x, y)
        mistakes = perceptron_path(x, y, epochs)[2]
        out.append(
            case(
                f"perceptron/plain/{epochs}",
                "perceptron",
                [x, y, q],
                {
                    "coef": np.asarray(p.coef_)[0],
                    "intercept": np.asarray(p.intercept_)[0],
                    "decision": p.decision_function(q),
                    "predict": p.predict(q),
                },
                rtol=1e-12,
                atol=1e-12,
                variant="plain",
                epochs=epochs,
                mistakes=mistakes,
            )
        )
        a = SGDClassifier(
            loss="perceptron",
            learning_rate="constant",
            eta0=1.0,
            penalty=NONE,
            average=True,
            shuffle=False,
            max_iter=epochs,
            tol=NONE,
        ).fit(x, y)
        out.append(
            case(
                f"perceptron/averaged/{epochs}",
                "perceptron",
                [x, y, q],
                {
                    "coef": np.asarray(a.coef_)[0],
                    "intercept": np.asarray(a.intercept_)[0],
                    "decision": a.decision_function(q),
                    "predict": a.predict(q),
                },
                rtol=1e-11,
                atol=1e-11,
                variant="averaged",
                epochs=epochs,
            )
        )
    # Noisy labels, so the pocket and the voted perceptron have something to do.
    xn, yn = data(g, 50, 1.5)
    qn = g.normal(size=(12, 3))
    vecs, bias, counts = voted(xn, yn, 3)
    total = counts.sum()
    votes = np.array([np.sum(counts * np.where(vecs @ row + bias > 0, 1.0, -1.0)) / total for row in qn])
    out.append(
        case(
            "perceptron/voted",
            "perceptron",
            [xn, yn, qn],
            {"decision": votes, "predict": (votes > 0).astype(np.float64)},
            rtol=1e-12,
            atol=1e-12,
            variant="voted",
            epochs=3,
        )
    )
    pw, pb = pocket(xn, yn, 6)
    out.append(
        case(
            "perceptron/pocket",
            "perceptron",
            [xn, yn, qn],
            {"coef": pw, "intercept": np.array(pb), "decision": qn @ pw + pb},
            rtol=1e-12,
            atol=1e-12,
            variant="pocket",
            epochs=6,
        )
    )
    # XOR needs the kernel.
    xx = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.1, 0.0], [0.9, 1.1], [0.1, 0.9], [1.1, 0.1]])
    yx = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    qx = g.uniform(-0.2, 1.2, size=(9, 2))
    alpha, b, mistakes = kernel_perceptron(xx, yx, 2.0, 30)
    ys = 2 * yx - 1
    decision = (alpha * ys) @ rbf_kernel(xx, qx, gamma=2.0) + b
    out.append(
        case(
            "perceptron/kernel",
            "kernelPerceptron",
            [xx, yx, qx],
            {"intercept": np.array(b), "decision": decision, "mistakes": np.array(float(mistakes))},
            rtol=1e-12,
            atol=1e-12,
            gamma=2.0,
            epochs=30,
        )
    )
    for variant, rate, c in [("pa1", "pa1", 0.5), ("pa2", "pa2", 0.5), ("pa1", "pa1", 100.0)]:
        m = SGDClassifier(
            loss="hinge", penalty=NONE, learning_rate=rate, eta0=c, shuffle=False, max_iter=3, tol=NONE
        ).fit(x, y)
        out.append(
            case(
                f"pa/classifier/{variant}/{c}",
                "pa",
                [x, y, q],
                {
                    "coef": np.asarray(m.coef_)[0],
                    "intercept": np.asarray(m.intercept_)[0],
                    "decision": m.decision_function(q),
                    "predict": m.predict(q),
                },
                rtol=1e-11,
                atol=1e-11,
                variant=variant,
                c=c,
                epochs=3,
            )
        )
    t = x @ np.array([2.0, -1.0, 0.5]) + 1.0 + 0.1 * g.normal(size=len(x))
    for variant, rate, c, eps in [("pa1", "pa1", 0.5, 0.1), ("pa2", "pa2", 1.0, 0.3)]:
        r = SGDRegressor(
            loss="epsilon_insensitive",
            epsilon=eps,
            penalty=NONE,
            learning_rate=rate,
            eta0=c,
            shuffle=False,
            max_iter=3,
            tol=NONE,
        ).fit(x, t)
        out.append(
            case(
                f"pa/regressor/{variant}",
                "paRegressor",
                [x, t, q],
                {"coef": r.coef_, "intercept": np.asarray(r.intercept_)[0], "predict": r.predict(q)},
                rtol=1e-11,
                atol=1e-11,
                variant=variant,
                c=c,
                epsilon=eps,
                epochs=3,
            )
        )
    return out
